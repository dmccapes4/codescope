"""Two-stage write flow.

When the user asks the agent to write to a file (e.g. "implement the
Composables and write to PatientActivity.kt"), a single LLM pass tends to
burn most of its num_predict budget rehashing the review prose before it
even starts emitting code — and qwen2.5-coder:14b reliably truncates
mid-code-block.

The two-stage flow splits that work:

  Stage 1 (PLAN, ~1500 tokens, markdown):
      Same answer model produces a compact markdown implementation plan
      (no full code yet). Pretty-printed to the terminal so the user can
      see the plan before stage 2 starts (the slow one).

      Qwen has a strong bias to wrap structured output in JSON envelopes
      (`{"action":"final_answer","content":"…"}`). We accept that and
      unwrap it transparently — the user always sees clean markdown.

  Stage 2 (IMPLEMENT, full ANSWER_NUM_PREDICT_WRITE, JSON):
      Re-call the answer model with the plan injected and a strict system
      directive: emit the COMPLETE new file body for each modified file
      as a fenced code block under a `### <path>` heading. No plan recap,
      no review — just headings + code blocks. The same
      `{"action":"final_answer","content":"…"}` JSON envelope is still
      returned so all the existing post-processing (`_maybe_write_file`,
      truncation salvage, `coerce_answer_text`) keeps working.

Toggle with CODESCOPE_TWO_STAGE_WRITE (default on for workstation profile,
off elsewhere — see config.py).
"""
from __future__ import annotations

import json
import re
from typing import Callable

from ..models.llm import LLM, OllamaError


_PLAN_SYSTEM = (
    "You are a senior Android engineer planning a file modification. The user "
    "has asked for an implementation change and the relevant source files have "
    "already been retrieved for you (see PRE-READ SOURCE FILES). Produce a "
    "compact implementation plan in markdown — NO full file bodies, NO long "
    "code blocks. Cover, in order:\n"
    "  1. **Files to modify** — one heading per file with its project-relative path.\n"
    "  2. **Changes per file** — bullet list of edits (add import X, replace function Y "
    "with new Composable Z, etc.) tied to the user's request.\n"
    "  3. **New imports** — group them by file.\n"
    "  4. **Key types / signatures** — function signatures, parameter types, return types.\n"
    "     Show only signatures, never full bodies.\n"
    "  5. **Order of operations** — which file to touch first if there are dependencies.\n"
    "  6. **Open questions / assumptions** — if anything is unclear, list it briefly. "
    "Otherwise write 'None.'\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "  • Return RAW markdown ONLY.\n"
    "  • Your VERY FIRST CHARACTER must be '#' (a markdown heading). NEVER '{'.\n"
    "  • DO NOT wrap the response in JSON. DO NOT emit "
    '{"action":"final_answer", ...}. DO NOT include ```json fences.\n'
    "  • Length budget: 60 lines max. Be concrete but terse — the next LLM pass "
    "will execute this plan and write the actual code."
)


# Stage-2 system suffix appended to the existing answer_system. Tells qwen to
# treat the injected plan as a design contract and emit full file bodies under
# project-relative-path headings — exactly what `_maybe_write_file` and
# `_parse_file_writes` can pick up downstream.
STAGE2_SYSTEM_SUFFIX = (
    "\n\n"
    "=== TWO-STAGE WRITE FLOW: IMPLEMENTATION PASS ===\n"
    "An IMPLEMENTATION PLAN block was prepended to the user prompt. Treat it as "
    "an authoritative design contract — do NOT re-summarise it, do NOT add a "
    "'Review' or 'Overview' section.\n\n"
    "Your `content` field MUST contain:\n"
    "  1. A short title (one line, e.g. '## Implementation').\n"
    "  2. For EACH file listed under 'Files to modify' in the plan, EXACTLY one "
    "section in this shape:\n\n"
    "        ### <project-relative path of the file>\n"
    "        ```kotlin\n"
    "        <COMPLETE updated contents of that file>\n"
    "        ```\n\n"
    "  3. Optionally one short closing line explaining how to validate (build "
    "command, where to look, etc.). No more than 3 lines of prose total outside "
    "code blocks.\n\n"
    "Rules:\n"
    "  • Emit FULL file bodies, not diffs or snippets. Include the package "
    "declaration, every existing import that should remain, and every existing "
    "declaration that should remain unchanged.\n"
    "  • Use the correct language tag in the fence: ```kotlin / ```kt / ```java "
    "/ ```xml / ```kts.\n"
    "  • Close every fence with ``` on its own line. Never leave a code block open.\n"
    "  • Wrap everything inside the normal "
    '{"action":"final_answer","content":"…"} JSON envelope just like a regular '
    "answer. Code blocks live inside `content`.\n"
)


def _strip_json_envelope(s: str) -> str:
    """If qwen wrapped the markdown plan in a JSON envelope
    `{"action":"final_answer","content":"…"}` (which it does ~50% of the time
    despite system instructions), unwrap it. Otherwise return the input.
    """
    if not s:
        return s
    text = s.strip()
    if not text.startswith("{"):
        return text
    # Try full JSON first.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to salvage with a regex when there's trailing junk.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return text
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return text
    if not isinstance(data, dict):
        return text
    content = data.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return text


def _pretty_plan(plan_md: str, print_fn: Callable[[str], None]) -> None:
    """Render the markdown plan via Rich if available, else plain text."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel

        console = Console()
        body = Markdown(plan_md)
        console.print(
            Panel(
                body,
                title="[bold cyan]Implementation plan (stage 1/2)[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
    except Exception:
        print_fn("\n=== Implementation plan (stage 1/2) ===")
        print_fn(plan_md)
        print_fn("========================================\n")


def run_plan_stage(
    llm: LLM,
    base_prompt: str,
    print_fn: Callable[[str], None],
    plan_num_predict: int = 1500,
) -> str | None:
    """Run stage 1 — produce + pretty-print the implementation plan.

    Returns the (unwrapped, JSON-stripped) plan markdown, or None on failure.
    """
    print_fn(
        f"[codescope] Stage 1/2: drafting implementation plan with "
        f"{llm.model} (num_predict={plan_num_predict})…"
    )
    try:
        raw = llm.generate(
            prompt=base_prompt,
            system=_PLAN_SYSTEM,
            json_mode=False,        # markdown, not JSON — but qwen often emits JSON anyway
            num_predict=plan_num_predict,
            temperature=0.15,
        )
    except OllamaError as e:
        print_fn(f"[codescope] Plan stage failed: {e}. Falling back to single-stage write.")
        return None
    plan_md = _strip_json_envelope(raw or "").strip()
    if not plan_md:
        print_fn("[codescope] Plan stage returned empty output. Falling back to single-stage write.")
        return None
    _pretty_plan(plan_md, print_fn)
    return plan_md


def inject_plan_into_prompt(base_prompt: str, plan_md: str) -> str:
    """Prepend the plan to the answer prompt and rewrite the trailing JSON
    instruction so stage 2 is unambiguous about emitting file bodies."""
    plan_block = (
        "=== IMPLEMENTATION PLAN (stage 1 — follow this exactly; do not "
        "re-summarise, do not deviate) ===\n"
        + plan_md
        + "\n\n"
    )
    # Replace the boring instruction line at the very end of base_prompt
    # (added by build_answer_prompt) with a stage-2-specific one. We append our
    # imperative even if the trailing line wasn't found — extra reinforcement.
    sep = (
        "\n\n=== STAGE 2 INSTRUCTION ===\n"
        "Execute the plan above. Respond with JSON:\n"
        '  {"action":"final_answer","content":"…"}\n'
        "Inside `content`, for EVERY file in the plan emit exactly:\n"
        "    ### <project-relative path>\n"
        "    ```kotlin\n"
        "    <COMPLETE updated file contents>\n"
        "    ```\n"
        "No plan recap. No 'Review' section. No prose between code blocks beyond a "
        "1-line transition. Close every fence with ``` on its own line.\n"
    )
    return plan_block + base_prompt + sep


def build_stage2_system(answer_system: str) -> str:
    """Append the stage-2 directive to the existing answer system prompt."""
    return answer_system.rstrip() + STAGE2_SYSTEM_SUFFIX
