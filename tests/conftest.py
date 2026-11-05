import os
import sys
import tempfile
from pathlib import Path

# Must be set before the app is imported: isolated DB, no network LLM or vision model, classical vision
# instead of the trained clothes models, and the agent run inside the request so results are immediate.
os.environ["RG_DATA_DIR"] = tempfile.mkdtemp(prefix="rg_test_")
os.environ["LLM_PROVIDER"] = "rules"
os.environ["VISION_BACKEND"] = "heuristic"
os.environ["CLOTHES_MODEL"] = "off"
os.environ["GEMINI_API_KEY"] = ""
os.environ["PRODUCT_CHECK"] = "off"
os.environ["AGENT_INLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
