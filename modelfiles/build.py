#!/usr/bin/env python3
"""Cross-platform builder for codescope Ollama instructor models.

Works on Linux, macOS, and Windows. Requires `python3` and `ollama` on PATH.

Usage (from any directory):
    python codescope/modelfiles/build.py
    python codescope/modelfiles/build.py --modelfile as-android-instructor.Modelfile

Build the codescope chat model (default):
    python codescope/modelfiles/build.py
    # → ollama create codescope -f codescope.Modelfile

Build ProScope (product/architect mode):
    python codescope/modelfiles/build.py --modelfile proscope.Modelfile --name proscope
    # → ollama create proscope -f proscope.Modelfile

Build the Android Studio instructor:
    python codescope/modelfiles/build.py --modelfile as-android-instructor.Modelfile

Environment overrides:
    CODESCOPE_PROFILE      laptop-6gb → 7b base, desktop-12gb → 14b base (when --base unset)
    CODESCOPE_MODEL_NAME   default: Modelfile stem (codescope or as-android-instructor)
    CODESCOPE_BASE_MODEL   explicit override for the FROM line

The script reads the Modelfile, optionally rewrites the FROM line based on
CODESCOPE_BASE_MODEL, writes a temporary build file in the same directory, runs
`ollama create`, and removes the temp file when done.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Base model swapped in at build time when CODESCOPE_BASE_MODEL is unset.
_BASE_BY_PROFILE: dict[str, str] = {
    "laptop-6gb":       "qwen2.5-coder:7b-instruct-q4_K_M",
    "desktop-12gb":     "qwen2.5-coder:14b",
    "workstation-24gb": "qwen2.5-coder:14b",
}


def _default_base_model() -> str | None:
    """Pick qwen base from CODESCOPE_PROFILE unless CODESCOPE_BASE_MODEL is set."""
    if os.environ.get("CODESCOPE_BASE_MODEL"):
        return None
    profile = os.environ.get("CODESCOPE_PROFILE", "laptop-6gb").lower()
    return _BASE_BY_PROFILE.get(profile)


def _detect_ollama_host() -> str | None:
    return os.environ.get("OLLAMA_HOST")


def _swap_from_line(text: str, base_model: str) -> str:
    """Replace the first non-comment FROM directive with `FROM <base_model>`."""
    out_lines: list[str] = []
    replaced = False
    for line in text.splitlines():
        stripped = line.strip()
        if not replaced and stripped.startswith("FROM "):
            out_lines.append(f"FROM {base_model}")
            replaced = True
        else:
            out_lines.append(line)
    if not text.endswith("\n"):
        out_lines.append("")
    return "\n".join(out_lines)


def build(modelfile: Path, model_name: str, base_model: str | None) -> int:
    if not shutil.which("ollama"):
        print("Error: 'ollama' not found in PATH.", file=sys.stderr)
        print("Install from https://ollama.com/download or add it to PATH.", file=sys.stderr)
        return 127
    if not modelfile.is_file():
        print(f"Error: Modelfile not found: {modelfile}", file=sys.stderr)
        return 2

    text = modelfile.read_text(encoding="utf-8")

    if base_model:
        text = _swap_from_line(text, base_model)

    tmp = modelfile.with_name(f".{model_name}.build.Modelfile")
    tmp.write_text(text, encoding="utf-8")

    host = _detect_ollama_host()
    host_msg = f"  OLLAMA_HOST={host}" if host else "  OLLAMA_HOST=(default)"
    print(f"[build] os={platform.system().lower()} arch={platform.machine()}")
    print(f"[build] model={model_name}  base={base_model or '(unchanged from Modelfile)'}")
    print(f"[build]{host_msg}")
    print(f"[build] running: ollama create {model_name} -f {tmp.name}")

    try:
        proc = subprocess.run(
            ["ollama", "create", model_name, "-f", str(tmp)],
            cwd=str(modelfile.parent),
        )
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass

    if proc.returncode != 0:
        print(f"\n[build] ollama create failed (exit {proc.returncode}).", file=sys.stderr)
        return proc.returncode

    print(f"\n[build] OK. Test it:")
    print(f"    ollama run {model_name} \"Reply with one sentence.\"")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a codescope Ollama instructor model.")
    parser.add_argument(
        "--modelfile",
        default=str(HERE / "codescope.Modelfile"),
        help="Path to the Modelfile (default: codescope.Modelfile next to this script).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Name to register with Ollama "
            "(default: $CODESCOPE_MODEL_NAME, then the Modelfile stem)."
        ),
    )
    parser.add_argument(
        "--base",
        default=None,
        help=(
            "Override the FROM line in the Modelfile "
            "(default: $CODESCOPE_BASE_MODEL, then leave FROM unchanged)."
        ),
    )
    args = parser.parse_args()

    mf = Path(args.modelfile).resolve()
    name = args.name or os.environ.get("CODESCOPE_MODEL_NAME") or mf.stem
    base = args.base or os.environ.get("CODESCOPE_BASE_MODEL") or _default_base_model()

    return build(mf, name, base)


if __name__ == "__main__":
    sys.exit(main())
