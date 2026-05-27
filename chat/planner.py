"""Fast planner (llama3.2): checklist + which tools to run before the main answer LLM."""
from __future__ import annotations

import json
from typing import Any

from pathlib import Path

from ..config import PLANNER_LLM, USE_PLANNER_LLM, PLANNER_NUM_GPU
from ..models.llm import LLM, OllamaError
from .preflight import plan_needs_android_docs
from .research import query_needs_codebase_research

PLANNER_SYSTEM = """\
You are a fast planning assistant for an Android codebase agent.
Output a single JSON object only — no markdown fences.

Shape:
{
  "checklist": [
    {"id": 1, "task": "...", "status": "done"|"pending"},
    ...
  ],
  "tools": [
    {"tool": "android_docs", "args": {"topic": "room"}},
    ...
  ]
}

Rules:
- If a project doc path is already pre-loaded, mark that checklist item "done" — do NOT add docs_lookup or read_file for it.
- Keep checklist short (3–6 items). Last item is always "Write final answer for the user".

When to add android_docs:
  YES — user asks for implementation strategy, platform API guidance, or "supporting documentation".
  NO  — user asks to search the codebase, inspect files, create/write a README, or explore the project.
        Codebase tasks use grep + semantic_search, NOT android_docs. Adding android_docs to a README
        or codebase-search task wastes ~900 tokens of context with irrelevant platform docs.

NEVER add write_file or edit_file to the tools list.
Writing happens in the answer phase after all research is complete, not during planning.
"""


def _default_plan(
    user_query: str,
    preloaded_files: list[str],
    project_root: Path | None = None,
) -> dict[str, Any]:
    checklist: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    n = 1
    q = user_query.lower()

    for path in preloaded_files:
        checklist.append({
            "id": n,
            "task": f"Use project doc {path} (already loaded in context)",
            "status": "done",
        })
        n += 1

    if query_needs_codebase_research(user_query):
        readme = project_root / "README.md" if project_root else None
        if readme and readme.is_file():
            checklist.append({
                "id": n,
                "task": "Use README.md as project summary anchor",
                "status": "done",
            })
            n += 1
        checklist.append({
            "id": n,
            "task": "Search codebase: semantic_search + grep (+ graph for key files)",
            "status": "pending",
        })
        n += 1
        tools.append({
            "tool": "semantic_search",
            "args": {"query": "Android application main activities Room database", "k": 8},
        })
        tools.append({
            "tool": "grep",
            "args": {
                "pattern": r"class\s+\w+(Activity|Fragment|ViewModel|Repository)\b",
                "path": ".",
                "regex": True,
                "max_results": 40,
            },
        })
        if "readme" in q and any(w in q for w in ("create", "write", "inspect", "search")):
            checklist.append({
                "id": n,
                "task": "Draft README.md at project root from graph + search findings",
                "status": "pending",
            })
            n += 1

    if plan_needs_android_docs(user_query):
        checklist.append({
            "id": n,
            "task": "Fetch and explain relevant Android platform documentation (Room)",
            "status": "pending",
        })
        n += 1
        tools.append({"tool": "android_docs", "args": {"topic": "room"}})

    final_task = "Write final answer for the user"
    if plan_needs_android_docs(user_query):
        final_task = "Write final answer: implementation strategy + documentation links"
    elif query_needs_codebase_research(user_query):
        final_task = "Write final answer from codebase research (and README draft if requested)"

    checklist.append({"id": n, "task": final_task, "status": "pending"})
    return {"checklist": checklist, "tools": tools}


def _merge_plan(default: dict, llm_plan: dict) -> dict[str, Any]:
    """Keep deterministic preloaded 'done' items; merge tools (default + LLM)."""
    out = {"checklist": list(default.get("checklist") or []), "tools": []}
    merged: dict[str, dict] = {}
    for t in (default.get("tools") or []) + (llm_plan.get("tools") or []):
        if isinstance(t, dict) and t.get("tool"):
            merged[t["tool"]] = {"tool": t["tool"], "args": t.get("args") or {}}
    out["tools"] = list(merged.values())
    # Append any LLM checklist items not duplicating preloaded paths
    existing = {c.get("task", "") for c in out["checklist"]}
    for c in llm_plan.get("checklist") or []:
        if isinstance(c, dict) and c.get("task") and c["task"] not in existing:
            out["checklist"].append(c)
    return out


def run_planner(
    user_query: str,
    graph_summary: str,
    preloaded_files: list[str],
    planner_llm: LLM | None = None,
    project_root: Path | None = None,
    use_llm: bool | None = None,
) -> dict[str, Any]:
    """Return {checklist, tools} for this turn."""
    default = _default_plan(user_query, preloaded_files, project_root)
    if use_llm is None:
        use_llm = USE_PLANNER_LLM
    if not use_llm:
        return default
    if planner_llm is None:
        planner_llm = LLM(model=PLANNER_LLM, timeout=90, num_gpu=PLANNER_NUM_GPU)

    prompt = f"""User question:
{user_query}

Pre-loaded project docs (full text is already in the answer prompt — do not re-fetch):
{json.dumps(preloaded_files) if preloaded_files else "(none)"}

Graph summary (abbreviated):
{graph_summary[:1200]}

Produce the plan JSON."""

    try:
        raw = planner_llm.generate(
            prompt=prompt,
            system=PLANNER_SYSTEM,
            json_mode=True,
            num_predict=400,
            temperature=0.0,
        )
        data = json.loads(raw)
        if isinstance(data, dict):
            return _merge_plan(default, data)
    except OllamaError:
        # Model missing (404) — deterministic default plan still runs android_docs etc.
        return default
    except (json.JSONDecodeError, TypeError):
        pass
    return default


def resolve_planner_llm() -> LLM:
    """Pick first available Ollama model matching planner preferences.

    Respects PLANNER_NUM_GPU so the caller doesn't have to pass it explicitly.
    Set CODESCOPE_PLANNER_NUM_GPU=0 to run llama on CPU (keeps qwen VRAM warm).
    """
    prefs = [PLANNER_LLM, "llama3.2:latest", "llama3.2:3b", "llama3.2"]
    try:
        probe = LLM(model=PLANNER_LLM, timeout=15)
        available = probe.list_models()
        for pref in prefs:
            for name in available:
                if name == pref or name.split(":")[0] == pref.split(":")[0]:
                    return LLM(model=name, timeout=90, num_gpu=PLANNER_NUM_GPU)
    except Exception:
        pass
    return LLM(model=PLANNER_LLM, timeout=90, num_gpu=PLANNER_NUM_GPU)


def format_checklist(plan: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in plan.get("checklist") or []:
        if not isinstance(item, dict):
            continue
        status = item.get("status", "pending")
        mark = "x" if status == "done" else " "
        lines.append(f"[{mark}] {item.get('id', '?')}. {item.get('task', '')}")
    return "\n".join(lines) if lines else "(empty checklist)"
