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
