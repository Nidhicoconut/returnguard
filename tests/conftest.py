import os
import sys
import tempfile
from pathlib import Path

# Must be set before the app is imported: isolated DB, no network LLM, classical vision.
os.environ["RG_DATA_DIR"] = tempfile.mkdtemp(prefix="rg_test_")
os.environ["LLM_PROVIDER"] = "rules"
os.environ["VISION_BACKEND"] = "heuristic"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
