"""One overview grid (cyan = pre-box, number = detector confidence) (4 photo, 4 anime, 4 art) of pre-annotated boxes for the README / status report."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib; qa = importlib.import_module("07_qa_grids")
random.seed(5)
pick = []
for st in ["photo", "anime", "art"]:
    pick += random.sample([r for r in qa.rows if r["style"] == st], 4)
qa.grid([qa.tile(r, 360, 480, hint_colors=False) for r in pick], 4, 360, 480, qa.ROOT / "qa/qa_overview.jpg")
print([r["id"] for r in pick])
