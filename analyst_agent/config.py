"""Central settings, read from environment variables (see .env.example)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# load .env from the repo root (KEY=value lines) without overriding real environment variables
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _v = _v.split(" #", 1)[0]          # allow trailing comments
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
DATA_DIR = Path(os.getenv("AGENT_DATA_DIR", ROOT / ".agent_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))
MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "2048"))

SQL_ROW_LIMIT = 200          # rows returned to the LLM per query
SQL_TIMEOUT_S = 10
PY_TIMEOUT_S = 30
PY_MEMORY_MB = 1024
TOOL_OUTPUT_CHARS = 4000     # truncate tool output sent back to the LLM
SMALL_SAMPLE_N = 30
SIGNIFICANCE_ALPHA = 0.05


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- production settings
# PUBLIC_MODE=true when the app is open to strangers (e.g. Streamlit Community Cloud).
PUBLIC_MODE = _flag("PUBLIC_MODE", False)
# LLM-written Python is the riskiest tool: off by default in public mode unless it runs in E2B (below).

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "200000"))
DUCKDB_MEMORY = os.getenv("DUCKDB_MEMORY", "512MB")
RUN_TIMEOUT_S = int(os.getenv("RUN_TIMEOUT_S", "120"))          # whole question, all steps
SESSION_TTL_MIN = int(os.getenv("SESSION_TTL_MIN", "120"))      # uploaded files are deleted after this

# Protect the server's free API key: per-session and whole-app daily caps (0 = no limit).
QUESTIONS_PER_HOUR = int(os.getenv("QUESTIONS_PER_HOUR", "15" if PUBLIC_MODE else "0"))
DAILY_QUESTION_CAP = int(os.getenv("DAILY_QUESTION_CAP", "300" if PUBLIC_MODE else "0"))

# Where AI-written Python runs: "e2b" (isolated cloud microVM, needs E2B_API_KEY) or "local" (subprocess).
PYTHON_SANDBOX = os.getenv("PYTHON_SANDBOX", "e2b" if os.getenv("E2B_API_KEY") else "local").strip().lower()
# With a real sandbox, Python is safe to offer publicly, so it defaults back on.
ALLOW_PYTHON = _flag("ALLOW_PYTHON", (not PUBLIC_MODE) or PYTHON_SANDBOX == "e2b")
