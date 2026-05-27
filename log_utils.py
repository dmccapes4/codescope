"""
Logging, terminal UX, and HITL governance utilities.

Design mirrors FullMetalPacket conventions:
  ▶️  phase start     [HH:MM:SS]
  ✅  phase complete  [HH:MM:SS]  elapsed
  ❌  failure         [HH:MM:SS]
  ⚠️  warning         [HH:MM:SS]
  📋  info            [HH:MM:SS]
  💬  user prompt

Governance gates: [C]ontinue / [A]bort / [Q]uestion / [S]kip / [R]etry
Streaming runner: subprocess → terminal + log file simultaneously
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Tuple, Optional


# ── Emoji vocabulary ──────────────────────────────────────────────────────────

ARROW   = "▶️ "
CHECK   = "✅"
CROSS   = "❌"
WARN    = "⚠️ "
INFO    = "📋"
CHAT    = "💬"
THINK   = "🧠"
SPARKLE = "✨"
PACKAGE = "📦"
SEARCH  = "🔍"
CLOCK   = "⏱️ "


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _elapsed(start: datetime) -> str:
    secs = (datetime.now(timezone.utc) - start).total_seconds()
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


# ── log_event ─────────────────────────────────────────────────────────────────

def log_event(phase: str, msg: str, emoji: str = INFO, file=None) -> None:
    line = f"[{_ts()}] {emoji} {phase} | {msg}"
    print(line, flush=True, file=file)


def log_start(phase: str, msg: str = "") -> datetime:
    detail = f" — {msg}" if msg else ""
    print(f"[{_ts()}] {ARROW} {phase}{detail}", flush=True)
    return datetime.now(timezone.utc)


def log_ok(phase: str, msg: str = "", start: datetime | None = None) -> None:
    elapsed = f"  ({_elapsed(start)})" if start else ""
    detail  = f" — {msg}" if msg else ""
    print(f"[{_ts()}] {CHECK} {phase}{detail}{elapsed}", flush=True)


def log_fail(phase: str, msg: str = "") -> None:
    detail = f" — {msg}" if msg else ""
    print(f"[{_ts()}] {CROSS} {phase}{detail}", flush=True)


def log_warn(phase: str, msg: str) -> None:
    print(f"[{_ts()}] {WARN} {phase} | {msg}", flush=True)


def log_info(phase: str, msg: str) -> None:
    print(f"[{_ts()}] {INFO} {phase} | {msg}", flush=True)


# ── Banners ───────────────────────────────────────────────────────────────────

def phase_banner(title: str, width: int = 68) -> None:
    border = "═" * width
    pad    = max(0, (width - len(title) - 2) // 2)
    print(f"\n{border}")
    print(f"{'':>{pad}}  {title}")
    print(f"{border}\n", flush=True)


def section_line(width: int = 68) -> None:
    print("─" * width, flush=True)


def print_banner(project: str, version: str = "0.1.0") -> None:
    W = 66  # inner width (between ║ and ║)

    def _row(content: str) -> str:
        # Truncate if too long, then left-pad to W
        return f"║{content[:W]:<{W}}║"

    title   = "  c o d e s c o p e"
    tagline = f"  Agentic code-ingestion + search  │  v{version}"
    proj    = f"  project: {project}"

    lines = [
        "╔" + "═" * W + "╗",
        _row(title),
        _row(tagline),
        _row(proj),
        "╚" + "═" * W + "╝",
    ]
    print("\n" + "\n".join(lines) + "\n", flush=True)


# ── safe_input ────────────────────────────────────────────────────────────────

def safe_input(prompt: str) -> str:
    """
    Read a line from the user.  Works even when stdin is not a TTY by falling
    back to /dev/tty on POSIX systems.  Retries once on EOFError.
    """
    for attempt in range(2):
        try:
            if getattr(sys.stdin, "isatty", lambda: True)():
                return input(prompt)
        except AttributeError:
            pass

        if os.path.exists("/dev/tty"):
            try:
                sys.stdout.write(prompt)
                sys.stdout.flush()
                with open("/dev/tty", "r") as tty:
                    return tty.readline().rstrip("\n")
            except OSError:
                pass

        try:
            return input(prompt)
        except EOFError:
            if attempt == 0:
                continue
            return ""
    return ""


# ── Governance gates ──────────────────────────────────────────────────────────

def governance_gate(
    phase_name: str,
    description: str,
    allow_skip: bool = False,
    allow_retry: bool = False,
) -> str:
    """
    Print a governance checkpoint and wait for user choice.

    Returns one of: "continue", "abort", "skip", "retry"
    """
    phase_banner(f"{THINK}  GOVERNANCE CHECKPOINT: {phase_name}")
    print(description.strip())
    print()

    opts = [("[C] Continue", "C")]
    if allow_retry:
        opts.append(("[R] Retry", "R"))
    if allow_skip:
        opts.append(("[S] Skip", "S"))
    opts.append(("[A] Abort", "A"))

    opt_str = "  ".join(o[0] for o in opts)
    valid   = {o[1] for o in opts}

    while True:
        print(f"{CHAT}  {opt_str}")
        choice = safe_input("Select: ").strip().upper()
        if choice in valid:
            break
        print(f"  Please choose one of: {', '.join(sorted(valid))}")

    if choice == "C":
        return "continue"
    if choice == "R":
        return "retry"
    if choice == "S":
        return "skip"
    # A
    log_warn(phase_name, "Aborted by user.")
    return "abort"


def confirm(prompt: str, default: bool = False) -> bool:
    """Simple [y/N] or [Y/n] confirmation."""
    hint = "[Y/n]" if default else "[y/N]"
    ans  = safe_input(f"{prompt} {hint}: ").strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes")


# ── Streaming subprocess runner ───────────────────────────────────────────────

def run_streaming(
    cmd: List[str],
    log_path: Path | None = None,
    cwd: Path | None = None,
    extra_env: dict | None = None,
    line_cb: Callable[[str], None] | None = None,
) -> Tuple[bool, str]:
    """
    Run *cmd* streaming stdout+stderr to the terminal AND to *log_path*.

    Returns (success: bool, last_snippet: str) where last_snippet is the
    final ~2 000 chars of output — useful to show on failure.
    """
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if extra_env:
        env.update(extra_env)

    last_lines: List[str] = []

    try:
        lf = open(log_path, "w", encoding="utf-8") if log_path else None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(cwd) if cwd else None,
                env=env,
            )
            assert proc.stdout
            for line in proc.stdout:
                print(line, end="", flush=True)
                if lf:
                    lf.write(line)
                    lf.flush()
                if line_cb:
                    line_cb(line)
                last_lines.append(line)
                # Keep snippet ≤ 2 000 chars
                while last_lines and sum(len(x) for x in last_lines) > 2_000:
                    last_lines.pop(0)
            proc.wait()
            success = proc.returncode == 0
            return success, "".join(last_lines).strip()
        finally:
            if lf:
                lf.close()
    except Exception as e:
        return False, str(e)


# ── run_meta.json ─────────────────────────────────────────────────────────────

def write_run_meta(
    path: Path,
    slug: str,
    suffix: str,
    phases_run: list[str],
    llm: str,
    embedder: str,
    elapsed_s: float,
    success: bool,
    extra: dict | None = None,
) -> None:
    meta = {
        "project":    slug,
        "suffix":     suffix,
        "run_at":     datetime.now(timezone.utc).isoformat(),
        "phases_run": phases_run,
        "llm":        llm,
        "embedder":   embedder,
        "elapsed_s":  round(elapsed_s, 2),
        "success":    success,
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Log directory resolution ──────────────────────────────────────────────────

def run_log_dir(workspace_root: Path, suffix: str, run_ts: str) -> Path:
    """Return logs/<suffix>/<run_ts>/ and create it."""
    d = workspace_root / "logs" / suffix / run_ts
    d.mkdir(parents=True, exist_ok=True)
    return d
