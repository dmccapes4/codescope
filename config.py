from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# The workspace root is the directory that *contains* the codescope package.
# Override with CODESCOPE_ROOT env var if you install the package elsewhere.
WORKSPACE_ROOT = Path(
    os.environ.get("CODESCOPE_ROOT", Path(__file__).parent.parent)
).resolve()

CACHE_ROOT    = WORKSPACE_ROOT / "cache"
SESSIONS_ROOT = WORKSPACE_ROOT / "sessions"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

DEFAULT_LLM      = "qwen2.5-coder:7b-instruct-q4_K_M"
PLANNER_LLM      = os.environ.get("CODESCOPE_PLANNER_LLM", "llama3.2:latest")
# Separate planner model loads a second weights set each turn (slow on 6 GB VRAM).
USE_PLANNER_LLM  = os.environ.get("CODESCOPE_USE_PLANNER_LLM", "0").lower() in ("1", "true", "yes")
# When planner LLM is on, reuse the answer model instead of PLANNER_LLM (avoids model swap).
PLANNER_USE_ANSWER_MODEL = os.environ.get("CODESCOPE_PLANNER_SAME_MODEL", "1").lower() not in (
    "0", "false", "no",
)
# GPU layers for the planner LLM (only relevant when USE_PLANNER_LLM=1).
#   -1 → full GPU (default Ollama behaviour, fastest inference)
#    0 → CPU-only inference (keeps qwen2.5 weights + KV fully in VRAM but adds ~10-80s latency)
# On 6 GB: llama3.2:3b (2.0 GB) + qwen 7B (4.7 GB) + qwen KV @16 K (1.9 GB) ≈ 8.6 GB → swap
# With PLANNER_NUM_GPU=0 llama runs on CPU; qwen never gets evicted.
PLANNER_NUM_GPU  = int(os.environ.get("CODESCOPE_PLANNER_NUM_GPU", "-1"))
WARM_MODEL_AT_REPL = os.environ.get("CODESCOPE_WARM_MODEL", "1").lower() not in ("0", "false", "no")
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_BASE_URL  = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Ollama inference options for the RTX 4050 6 GB
# num_gpu=-1  → offload all layers that fit
# num_ctx     → context window per call
# keep_alive  → how long to keep the model in VRAM after the last request
#               -1 (int) = indefinitely (until ollama serve is stopped)
#                0 (int) = unload immediately after each call
#               "10m" etc. = duration string for idle timeout
OLLAMA_NUM_GPU   = int(os.environ.get("CODESCOPE_NUM_GPU", "-1"))
# Context window (tokens).
#   RTX 4050 6 GB VRAM budget (approx):
#     qwen2.5-coder 7B Q4_K_M weights : ~4.7 GB
#     KV cache @ 16384 ctx            : ~1.9 GB  → total ~6.6 GB (fits with minor spill)
#     KV cache @ 24576 ctx            : ~2.8 GB  → total ~7.5 GB (overflows → RAM spill → timeout)
#   Default 16384 is the safe ceiling on 6 GB.
#   Set CODESCOPE_NUM_CTX=24576 only if you have ≥ 8 GB VRAM.
OLLAMA_NUM_CTX   = int(os.environ.get("CODESCOPE_NUM_CTX", "16384"))

# Pre-load project docs when the user names a full path in the query (e.g. docs/GAME_PLAN.md)
PREFLIGHT_PROJECT_DOCS = os.environ.get("CODESCOPE_PREFETCH_PROJECT_DOCS", "1").lower() not in (
    "0", "false", "no",
)
PREFLIGHT_ANDROID_DOCS = os.environ.get("CODESCOPE_PREFETCH_ANDROID_DOCS", "0").lower() in (
    "1", "true", "yes",
)
# Run android_docs_validate immediately after android_docs (saves one LLM hop)
AUTO_ANDROID_VALIDATE = os.environ.get("CODESCOPE_AUTO_ANDROID_VALIDATE", "1").lower() not in (
    "0", "false", "no",
)


def _parse_keep_alive(raw: str) -> int | str:
    """Parse CODESCOPE_KEEP_ALIVE for Ollama API (int -1/0 or duration string)."""
    raw = raw.strip()
    if raw in ("-1", "inf", "infinite"):
        return -1
    if raw == "0":
        return 0
    if raw.lstrip("-").isdigit():
        return int(raw)
    return raw  # e.g. "5m", "1h"


OLLAMA_KEEP_ALIVE = _parse_keep_alive(
    os.environ.get("CODESCOPE_KEEP_ALIVE", "-1")
)

# ---------------------------------------------------------------------------
# Context budget (in tokens, using cl100k_base as approximation)
# ---------------------------------------------------------------------------

MAX_CONTEXT_TOKENS      = OLLAMA_NUM_CTX
MAX_TOOL_HOPS           = 6
MAX_TOOL_RESULT_TOKENS  = 800
MAX_DOCS_TOOL_RESULT_TOKENS = 2_500   # docs_lookup / android_docs (full pages)
HISTORY_TURNS           = 2      # user+assistant pairs kept in rolling window

# Prompt field is capped at 60 % of the context window.
# The remaining 40 % is reserved for the system prompt, model reasoning, and response.
#   At 16384 ctx: prompt ≤ 9830 tokens, response headroom ≥ 6554 tokens
#   At 24576 ctx: prompt ≤ 14745 tokens (only use if VRAM ≥ 8 GB)
_PROMPT_RATIO           = 0.60
PROMPT_FIELD_MAX_TOKENS = int(OLLAMA_NUM_CTX * _PROMPT_RATIO)
REPLY_RESERVE_TOKENS    = int(OLLAMA_NUM_CTX * 0.30)   # model reasoning + JSON response

SYSTEM_PROMPT_TOKENS    = 600   # larger now (tactics + OGrE rules)
GRAPH_SUMMARY_TOKENS    = 600
HISTORY_TOKENS          = min(1_600, OLLAMA_NUM_CTX // 10)
TOOL_OUTPUT_TOKENS      = min(2_500, OLLAMA_NUM_CTX // 6)
PREFLIGHT_DOC_MAX_TOKENS = 2_000
ANDROID_BRIEF_MAX_TOKENS = 1_000
QUERY_TOKENS            = 200

# Priority budget caps for build_answer_prompt sections (tokens).
# Sum of all caps must not exceed PROMPT_FIELD_MAX_TOKENS (they are maxima, not guarantees).
# At 16K (prompt budget ~9830):
BUDGET_GRAPH        = 600    # 1. codebase graph
BUDGET_RESEARCH     = 2_500  # 2. grep / semantic / graph_lookup / read_file
BUDGET_GIT          = 700    # 3. git log + diff summary
BUDGET_ANDROID      = 1_000  # 4. android_docs excerpt
BUDGET_SESSION      = 400    # 5. query_session.json notes
BUDGET_SESSION_LOG  = 1_200  # 6. relevant prior Q&A turns from session.jsonl
BUDGET_DOCS         = 1_800  # 7. preloaded project docs (compressed when tight)
BUDGET_HISTORY      = 500    # 8. last N turns (condensed)
# Total max cap: 600+2500+700+1000+400+1200+1800+500 = 8700 → fits under 9830 ✓

# ---------------------------------------------------------------------------
# Hybrid-fusion context curation floors (tokens).
# These guarantee a minimum token contribution from each source, even if the
# overall budget is tight. Only sources with a non-zero floor are pinned.
# ---------------------------------------------------------------------------
FLOOR_GRAPH         = 250   # always show some graph topology
FLOOR_RESEARCH      = 300   # always include top code-evidence chunks if research ran
FLOOR_GIT           = 0     # git context is only shown when explicitly triggered
FLOOR_ANDROID       = 0     # android docs shown only when fetched
FLOOR_SESSION       = 150   # show at least a snippet of this-turn session notes
FLOOR_SESSION_LOG   = 0     # prior turns only when session_search ran
FLOOR_DOCS          = 0     # project docs shown only when preloaded

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

MAX_FILE_BYTES      = 1_000_000   # skip files larger than this
MAX_LLM_FILE_BYTES  = 16_384      # truncate file content sent to LLM
MIN_LLM_FILE_BYTES  = 200         # skip tiny files in enrichment
CHUNK_SIZE          = 800         # characters per embedding chunk
CHUNK_OVERLAP       = 120
EMBED_BATCH_SIZE    = 64
PROMPT_VERSION      = 1           # bump to force re-enrichment of all files

# ---------------------------------------------------------------------------
# Exclusions (dirs and files always skipped during tree walk)
# ---------------------------------------------------------------------------

EXCLUDED_DIRS: frozenset[str] = frozenset({
    "build", ".gradle", ".idea", ".kotlin", ".git", ".cxx",
    "intermediates", "generated", "outputs", ".run", "release",
    "debug", "tmp", "__pycache__", ".eggs", "dist", "node_modules",
})

EXCLUDED_SUFFIXES: frozenset[str] = frozenset({
    ".iml", ".lock", ".jar", ".aar", ".apk", ".aab",
    ".class", ".dex", ".so", ".o", ".a",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".ico",
    ".ttf", ".otf", ".woff", ".woff2",
    ".mp3", ".mp4", ".wav", ".ogg",
    ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".pb", ".tflite", ".onnx", ".gguf", ".bin",
    ".keystore", ".jks",
})

EXCLUDED_FILENAMES: frozenset[str] = frozenset({
    "local.properties", "gradle-wrapper.jar", ".DS_Store", "Thumbs.db",
})

# ---------------------------------------------------------------------------
# Which file categories are passed to the embedder
# ---------------------------------------------------------------------------

EMBEDDABLE_CATEGORIES: frozenset[str] = frozenset({
    "source.kotlin",
    "source.java",
    "manifest.xml",
    "gradle.kts",
    "versions.toml",
    "markdown",
    "proto",
})
