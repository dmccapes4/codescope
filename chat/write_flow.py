"""Two-stage write flow.

When the user asks the agent to write to a file (e.g. "implement the
Composables and write to the file"), a single LLM pass tends to burn most of
its num_predict budget rehashing the review prose before it even starts
emitting code — and qwen2.5-coder:14b reliably truncates mid-code-block.

The two-stage flow splits that work:

  Stage 1 (PLAN, ~1500 tokens, markdown):
      Same answer model produces a compact markdown implementation plan
      (no full code yet). Pretty-printed to the terminal so the user can
      see the plan before stage 2 starts (the slow one).

  Stage 2 (IMPLEMENT, full ANSWER_NUM_PREDICT_WRITE, JSON):
      Re-call the answer model with the plan from stage 1 injected into
      the prompt. Stage 2 produces the normal {"action":"final_answer",
      "content":"…"} JSON envelope — so all the existing post-processing
      (truncation salvage, _maybe_write_file, etc.) keeps working.

Toggle with CODESCOPE_TWO_STAGE_WRITE (default on for workstation profile,
off elsewhere — see config.py).
"""
from __future__ import annotations

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
    "Length budget: 60 lines max. The next LLM pass will execute this plan and write "
    "the actual code, so be concrete but terse. Return ONLY the markdown plan — no "
    "preamble, no JSON wrapper."
)


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
    Returns the plan markdown, or None if the LLM call failed / empty."""
    print_fn(
        f"[codescope] Stage 1/2: drafting implementation plan with "
        f"{llm.model} (num_predict={plan_num_predict})…"
    )
    try:
        plan_md = llm.generate(
            prompt=base_prompt,
            system=_PLAN_SYSTEM,
            json_mode=False,        # markdown, not JSON
            num_predict=plan_num_predict,
            temperature=0.15,
        )
    except OllamaError as e:
        print_fn(f"[codescope] Plan stage failed: {e}. Falling back to single-stage write.")
        return None
    plan_md = (plan_md or "").strip()
    if not plan_md:
        print_fn("[codescope] Plan stage returned empty output. Falling back to single-stage write.")
        return None
    _pretty_plan(plan_md, print_fn)
    return plan_md


def inject_plan_into_prompt(base_prompt: str, plan_md: str) -> str:
    """Prepend the plan to the answer prompt so stage 2 treats it as authoritative."""
    plan_block = (
        "=== IMPLEMENTATION PLAN (stage 1 — follow this exactly; do not "
        "re-summarise the review, do not deviate from the plan) ===\n"
        + plan_md
        + "\n\n"
    )
    return plan_block + base_prompt
