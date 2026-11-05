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

# Business rules (enforced in code, not left to the LLM). Defaults follow the project plan
# (10-day window, ₹5,000 auto-approve cap); both still need mentor confirmation.
RETURN_WINDOW_DAYS = int(os.getenv("RETURN_WINDOW_DAYS", "10"))
AUTO_APPROVE_MAX_PRICE = float(os.getenv("AUTO_APPROVE_MAX_PRICE", "5000"))  # INR
NON_RETURNABLE_TYPES = {"innerwear", "swimwear", "socks", "customised"}
# Late returns are declined without a reviewer (the customer can still appeal). The project plan says
# "never auto-reject"; set to 0 to send late returns to a reviewer instead.
AUTO_REJECT_OUTSIDE_WINDOW = os.getenv("AUTO_REJECT_OUTSIDE_WINDOW", "1").lower() in ("1", "true", "yes")
MAX_EVIDENCE_ROUNDS = int(os.getenv("MAX_EVIDENCE_ROUNDS", "2"))  # "please send another photo" requests per case

# Agent guardrails (project plan: max 8 tool calls / 120 s per case)
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "8"))
CASE_TIME_LIMIT_S = int(os.getenv("CASE_TIME_LIMIT_S", "120"))
# Run the agent inside the request instead of a background thread (tests use this)
AGENT_INLINE = os.getenv("AGENT_INLINE", "0").lower() in ("1", "true", "yes")

# Uploads
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_PHOTOS = 4
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Image pre-analysis (quality gate and integrity checks)
# Limits were checked on 244 good garment photos (real customer and test photos): about 1% fail.
MIN_SIDE_PX = int(os.getenv("MIN_SIDE_PX", "200"))
MIN_SHARPNESS = float(os.getenv("MIN_SHARPNESS", "30"))  # variance of the Laplacian at 640 px
DARK_MEAN, BRIGHT_MEAN = 25, 240                          # mean grey level limits (0-255)
DUPLICATE_HASH_BITS = 10                                  # of 256 dHash bits; same photo re-saved: <= 8
# OpenCV YuNet face detector, used to blur faces before a photo is sent to the external vision model
FACE_MODEL = Path(os.getenv("FACE_MODEL", BASE_DIR / "weights" / "face_detection_yunet_2023mar.onnx"))

# Accounts and sessions
SESSION_DAYS = int(os.getenv("SESSION_DAYS", "7"))
SHOW_DEMO_ACCOUNTS = os.getenv("SHOW_DEMO_ACCOUNTS", "1").lower() in ("1", "true", "yes")

# Vision: everything is clothing for now. The trained models come from CV/ (see CV/README.md);
# without the weights the app falls back to VISION_BACKEND (CLIP or the OpenCV heuristic).
VISION_BACKEND = os.getenv("VISION_BACKEND", "auto").lower()
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
CV_DIR = BASE_DIR / "CV"
CLOTHES_MODEL = os.getenv("CLOTHES_MODEL", "auto").lower()
CLOTHING_CATEGORIES = {"clothing", "apparel", "clothes"}
CLOTHES_DETECTOR = Path(os.getenv("CLOTHES_DETECTOR", CV_DIR / "runs" / "deploy" / "clothes_damage_detector.pt"))
CLOTHES_LOCATOR = Path(os.getenv("CLOTHES_LOCATOR", CV_DIR / "runs" / "deploy" / "clothes_damage_locator.pt"))
CLOTHES_DEVICE = os.getenv("CLOTHES_DEVICE", "auto").lower()  # auto | cpu | 0 (first GPU)
# Grounding DINO second opinion: ~700 MB model, slow on CPU. Needs CV/.pydeps (transformers).
CLOTHES_SECOND_OPINION = os.getenv("CLOTHES_SECOND_OPINION", "0").lower() in ("1", "true", "yes")
# Boxes shown to reviewers and sent to the vision check (the damaged/clean verdict does not use these):
# real defects are small, so boxes over a fifth of the photo are dropped; scores below STRONG are drawn as weak hints.
REGION_MAX_AREA = float(os.getenv("REGION_MAX_AREA", "0.2"))
REGION_MIN_SCORE = float(os.getenv("REGION_MIN_SCORE", "0.25"))
REGION_STRONG_SCORE = float(os.getenv("REGION_STRONG_SCORE", "0.3"))

# Text LLM (claim parsing and report wording)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama3.1")

# Vision-language check of damage crops (tool T9). Gemini through its OpenAI-compatible endpoint.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
# gemini-2.5-flash allows only 20 free requests a day (ran out during testing on 2026-11-03); the lite model has
# its own, larger free quota. Google retires model names over time: a 404 names the replacement.
VLM_MODEL = os.getenv("VLM_MODEL", "gemini-3.5-flash-lite")
VLM_MAX_CALLS = int(os.getenv("VLM_MAX_CALLS", "3"))  # per case
# Empty = model default. With gemini-2.5-flash set "none": its thinking otherwise uses up max_tokens and cuts the
# JSON reply short. gemini-3.5-flash-lite rejects "none" (use "minimal" or leave empty).
VLM_REASONING_EFFORT = os.getenv("VLM_REASONING_EFFORT", "")
# When Gemini is overloaded (5xx) or too slow, each call is retried once with this model. Keep each call short so
# an outage costs seconds rather than the agent's whole 120 s budget.
VLM_FALLBACK_MODEL = os.getenv("VLM_FALLBACK_MODEL", "gemini-2.5-flash")
VLM_FALLBACK_REASONING_EFFORT = os.getenv("VLM_FALLBACK_REASONING_EFFORT", "none")  # 2.5 "thinks" unless told not to
VLM_TIMEOUT = float(os.getenv("VLM_TIMEOUT", "25"))
# Product check (tool T3, app/product.py): is the ordered item in the photos?
PRODUCT_CHECK = os.getenv("PRODUCT_CHECK", "auto").lower()       # auto (when a Gemini key is set) | off
PRODUCT_MISMATCH_CONF = float(os.getenv("PRODUCT_MISMATCH_CONF", "0.7"))  # Gemini "no" must be this sure
CATALOG_DIR = Path(os.getenv("CATALOG_DIR", BASE_DIR / "catalog"))  # reference photos: catalog/<sku>/*.jpg
# Searched too: listing photos of the Amazon demo products (research use only, gitignored)
CATALOG_DIRS = [CATALOG_DIR, BASE_DIR / "demo_amazon" / "catalog"]
# CLIP cosine, checked on 120 Roboflow garments: copies of the same photo >= 0.85 in 67%, different garments 0.8%;
# Real customer close-ups score far lower (0.55-0.70 even for the right product), so catalog similarity
# only ever confirms a match; it never flags a mismatch.
CATALOG_MATCH_SIM = float(os.getenv("CATALOG_MATCH_SIM", "0.85"))
# The plan only auto-approves a defect when the VLM confirms it. Set to 0 to allow detector-only approval.
REQUIRE_VLM_FOR_APPROVAL = os.getenv("REQUIRE_VLM_FOR_APPROVAL", "1").lower() in ("1", "true", "yes")
