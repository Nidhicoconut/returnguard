"""Central configuration. Everything is overridable via environment variables or a .env file."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = BASE_DIR / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

DATA_DIR = Path(os.getenv("RG_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "returnguard.db"
POLICY_DIR = BASE_DIR / "policy"
STATIC_DIR = BASE_DIR / "static"

# Business rules (these are enforced in code, not left to the LLM)
RETURN_WINDOW_DAYS = int(os.getenv("RETURN_WINDOW_DAYS", "30"))
HUMAN_APPROVAL_ABOVE = float(os.getenv("HUMAN_APPROVAL_ABOVE", "10000"))  # INR
NON_RETURNABLE_CATEGORIES = {"personal care", "software", "gift cards"}

# Upload limits
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Vision / LLM selection
VISION_BACKEND = os.getenv("VISION_BACKEND", "auto").lower()
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama3.1")
