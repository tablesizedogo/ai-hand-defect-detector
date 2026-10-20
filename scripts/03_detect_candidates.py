"""Stage 1b: run a hand detector over candidate images (screening pass).
Usage: 03_detect_candidates.py <weights> <in.jsonl> <out.jsonl> [conf]"""
import json, sys, time
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from detectors import YoloHand
SRC = Path("/workspace/rag_civitai")
w, inp, outp = sys.argv[1:4]; conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
det = YoloHand(w)
done = set()
if Path(outp).exists():
    done = {json.loads(l)["id"] for l in open(outp)}
rows = [json.loads(l) for l in open(inp)]
t0 = time.time(); n = 0
with open(outp, "a") as f:
    for r in rows:
        if r["id"] in done: continue
        im = Image.open(SRC / r["path"]).convert("RGB")
        boxes = [[round(v, 1) for v in b[:4]] + [round(b[4], 3)] for b in det(im, conf)]
        f.write(json.dumps({"id": r["id"], "w": im.width, "h": im.height, "boxes": boxes}) + "\n"); n += 1
        if n % 500 == 0:
            f.flush(); print(n, f"{(time.time()-t0)/n:.3f}s/img", flush=True)
print("done", n, f"{(time.time()-t0)/max(n,1):.3f}s/img", flush=True)
