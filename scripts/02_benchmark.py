"""Stage 2a: benchmark CPU hand detectors on a small stratified sample (speed + raw boxes)."""
import json, random, sys, time
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from detectors import YoloHand, GDino, MPHands

ROOT = Path(__file__).resolve().parent.parent
SRC = Path("/workspace/rag_civitai")
rows = [json.loads(l) for l in open(ROOT / "data/candidates.jsonl")]
random.seed(7)
sample = []
for st in ["photo", "anime", "art"]:
    pool = [r for r in rows if r["style"] == st and r["hand_clip"] > 0.25]
    sample += random.sample(pool, 16)
json.dump(sample, open(ROOT / "data/bench_sample.json", "w"))
dets = [YoloHand("hand_yolov8n.pt"), YoloHand("hand_yolov8s.pt"), YoloHand("hand_yolov9c.pt"), MPHands(), GDino()]
res = {}
for d in dets:
    t0 = time.time(); out = {}
    for r in sample:
        im = Image.open(SRC / r["path"]).convert("RGB")
        out[r["id"]] = d(im)
    dt = (time.time() - t0) / len(sample)
    res[d.name] = {"sec_per_img": round(dt, 3), "boxes": out}
    print(d.name, f"{dt:.3f}s/img", sum(len(v) for v in out.values()), "boxes", flush=True)
json.dump(res, open(ROOT / "data/bench_raw.json", "w"))
