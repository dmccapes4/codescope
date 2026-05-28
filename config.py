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

# ---------------------------------------------------------------------------
# Hardware profiles — set CODESCOPE_PROFILE or override individual vars.
#
#   laptop-6gb   (default) RTX 4050 6 GB
#     LLM    : qwen2.5-coder:7b-instruct-q4_K_M
#     ctx    : 12 288  (all 29 layers on GPU; leaves ~350 MB headroom)
#     planner: deterministic (USE_PLANNER_LLM=0)
#     embed  : cpu
#
#   workstation-24gb        i9 / RTX 4090 24 GB
#     LLM    : qwen2.5-coder:14b-instruct-q4_K_M  (9 GB weights)
#     ctx    : 32 768  (14B KV ≈ 3.3 GB; llama3.2:3b planner ≈ 2 GB → ~16 GB total)
#     planner: llama3.2:3b  (USE_PLANNER_LLM=1, PLANNER_SAME_MODEL=0)
#     embed  : cuda
#
#   Override any value individually with its CODESCOPE_* env var.
# ---------------------------------------------------------------------------
_PROFILE = os.environ.get("CODESCOPE_PROFILE", "laptop-6gb").lower()

# Answer model
DEFAULT_LLM = os.environ.get(
    "CODESCOPE_LLM",
    "qwen2.5-coder:14b-instruct-q4_K_M"
    if _PROFILE == "workstation-24gb"
    else "qwen2.5-coder:7b-instruct-q4_K_M",
)

# llama3.2:3b is the fast planner; :latest may resolve to a larger tag on some hosts.
_DEFAULT_PLANNER = "llama3.2:3b" if _PROFILE == "workstation-24gb" else "llama3.2:latest"
PLANNER_LLM = os.environ.get("CODESCOPE_PLANNER_LLM", _DEFAULT_PLANNER)
# Separate planner model loads a second weights set each turn.
# On 6 GB this causes VRAM swap → default OFF.  On 24 GB both models stay warm → default ON.
USE_PLANNER_LLM = os.environ.get(
    "CODESCOPE_USE_PLANNER_LLM",
    "1" if _PROFILE == "workstation-24gb" else "0",
).lower() in ("1", "true", "yes")
# When planner LLM is on, reuse the answer model instead of PLANNER_LLM (avoids model swap on 6 GB).
PLANNER_USE_ANSWER_MODEL = os.environ.get(
    "CODESCOPE_PLANNER_SAME_MODEL",
    "0" if _PROFILE == "workstation-24gb" else "1",
).lower() not in ("0", "false", "no")
# GPU layers for the planner LLM (only relevant when USE_PLANNER_LLM=1).
#   -1 → full GPU (default; on 24 GB both models stay in VRAM simultaneously)
#    0 → CPU-only inference (preserves qwen KV on 6 GB but adds ~10-80s latency)
PLANNER_NUM_GPU = int(os.environ.get("CODESCOPE_PLANNER_NUM_GPU", "-1"))
WARM_MODEL_AT_REPL = os.environ.get("CODESCOPE_WARM_MODEL", "1").lower() not in ("0", "false", "no")
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
# On 6 GB: keep embedder on CPU so all qwen layers stay on GPU (~100 MB freed).
# On 24 GB: run on CUDA for faster batch encoding during curation.
EMBED_DEVICE = os.environ.get(
    "CODESCOPE_EMBED_DEVICE",
    "cuda" if _PROFILE == "workstation-24gb" else "cpu",
)
OLLAMA_BASE_URL  = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# HTTP timeouts for Ollama /api/generate (seconds).
# Remote tunneled 14B first load can exceed 120s; workstation profile defaults higher.
_DEFAULT_LLM_TIMEOUT = "600" if _PROFILE == "workstation-24gb" else "120"
LLM_TIMEOUT = float(os.environ.get("CODESCOPE_LLM_TIMEOUT", _DEFAULT_LLM_TIMEOUT))
PLANNER_LLM_TIMEOUT = float(os.environ.get("CODESCOPE_PLANNER_TIMEOUT", str(min(LLM_TIMEOUT, 180))))

# Ollama inference options for the RTX 4050 6 GB
# num_gpu=-1  → offload all layers that fit
# num_ctx     → context window per call
# keep_alive  → how long to keep the model in VRAM after the last request
#               -1 (int) = indefinitely (until ollama serve is stopped)
#                0 (int) = unload immediately after each call
#               "10m" etc. = duration string for idle timeout
OLLAMA_NUM_GPU   = int(os.environ.get("CODESCOPE_NUM_GPU", "-1"))
# Context window (tokens).
#
#   laptop-6gb   (RTX 4050 6 GB, qwen 7B Q4_K_M):
#     weights 4.46 GB + KV@12K 0.50 GB + compute 0.55 GB ≈ 5.5 GB  ✓  all layers on GPU
#     KV@16K would push 21/29 layers to CPU → 3× slower generation
#
#   workstation-24gb  (RTX 4090 24 GB, qwen 14B Q4_K_M + llama3.2:3b):
#     qwen14B 9.0 GB + KV@32K 3.3 GB + llama3b 2.0 GB + compute ≈ 16 GB  ✓
#     All layers on GPU, comfortable headroom for embedder on CUDA.
#
#   Set CODESCOPE_NUM_CTX to override.
_DEFAULT_CTX = "32768" if _PROFILE == "workstation-24gb" else "12288"
OLLAMA_NUM_CTX = int(os.environ.get("CODESCOPE_NUM_CTX", _DEFAULT_CTX))

# Answer generation limits (num_predict) — separate from context window.
# Truncated answers are usually num_predict, not num_ctx.
if _PROFILE == "workstation-24gb":
    # 32K context host: allow substantially larger completions.
    ANSWER_NUM_PREDICT       = int(os.environ.get("CODESCOPE_NUM_PREDICT", "4500"))
    ANSWER_NUM_PREDICT_EXPLAIN = int(os.environ.get("CODESCOPE_NUM_PREDICT_EXPLAIN", "7000"))
    ANSWER_NUM_PREDICT_WRITE   = int(os.environ.get("CODESCOPE_NUM_PREDICT_WRITE", "9000"))
else:
    ANSWER_NUM_PREDICT       = int(os.environ.get("CODESCOPE_NUM_PREDICT", "2000"))
    ANSWER_NUM_PREDICT_EXPLAIN = int(os.environ.get("CODESCOPE_NUM_PREDICT_EXPLAIN", "2500"))
    ANSWER_NUM_PREDICT_WRITE   = int(os.environ.get("CODESCOPE_NUM_PREDICT_WRITE", "3500"))

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
# Derived as fractions of PROMPT_FIELD_MAX_TOKENS so they auto-scale with any num_ctx.
# At 12K (budget ~7373): total cap ≈ 6640 ✓
# At 16K (budget ~9830): total cap ≈ 8700 ✓
_B = PROMPT_FIELD_MAX_TOKENS
BUDGET_GRAPH        = min(600,   _B // 16)   # ~6 %   codebase graph
BUDGET_RESEARCH     = min(2_500, _B //  4)   # ~25%   grep / semantic / graph_lookup / read_file
BUDGET_GIT          = min(700,   _B // 14)   # ~7 %   git log + diff summary
BUDGET_ANDROID      = min(1_000, _B // 10)   # ~10%   android_docs excerpt
BUDGET_SESSION      = min(400,   _B // 24)   # ~4 %   query_session.json notes
BUDGET_SESSION_LOG  = min(1_200, _B //  8)   # ~12%   relevant prior Q&A turns from session.jsonl
BUDGET_DOCS         = min(1_800, _B //  5)   # ~20%   preloaded project docs
BUDGET_HISTORY      = min(500,   _B // 20)   # ~5 %   last N turns (condensed)

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
