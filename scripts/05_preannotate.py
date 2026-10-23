"""Stage 2: pre-annotate hand boxes on the selected images.
Boxes = NMS-union of ADetailer hand_yolov8s (conf>=0.15) and hand_yolov9c (conf>=0.25), IoU 0.4.
Every box gets the default class `hand_ok`. A weak zero-shot CLIP "defect hint" (p_bad) is computed on padded
crops and stored as metadata only - it is NOT a label."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from detectors import YoloHand, nms

ROOT = Path(__file__).resolve().parent.parent; SRC = Path("/workspace/rag_civitai")
MIN_BOX = 20  # px at 512-px width: smaller hands are not annotated (see docs/LABELING_GUIDE.md)
BAD = ["a deformed hand with extra fingers", "a mangled, malformed hand", "a hand with fused, melted fingers",
       "a distorted hand with the wrong number of fingers", "a twisted, broken-looking hand"]
OK = ["a normal human hand with five fingers", "a well-drawn, anatomically correct hand", "a natural-looking hand",
      "a clean illustration of a hand"]

sel = [json.loads(l) for l in open(ROOT / "data/selected.jsonl")]
v8 = {d["id"]: d for d in map(json.loads, open(ROOT / "data/det_v8s.jsonl"))}
det9 = YoloHand("hand_yolov9c.pt")
import open_clip
model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); model.eval()
tok = open_clip.get_tokenizer("ViT-B-32")
with torch.inference_mode():
    T = model.encode_text(tok(BAD + OK)).float(); T /= T.norm(dim=-1, keepdim=True)
cache = {}
if (ROOT / "data/preannotations_v0.jsonl").exists():  # reuse results of an earlier run for unchanged images
    cache = {d["id"]: d for d in map(json.loads, open(ROOT / "data/preannotations_v0.jsonl"))}
t0 = time.time(); out = []; t_det = t_clip = 0.0
for n, r in enumerate(sel):
    if r["id"] in cache:
        out.append({**r, "W": cache[r["id"]]["W"], "H": cache[r["id"]]["H"], "boxes": cache[r["id"]]["boxes"]}); continue
    im = Image.open(SRC / r["path"]).convert("RGB"); W, H = im.size
    a = time.time()
    b8 = [tuple(b) for b in v8[r["id"]]["boxes"] if b[4] >= 0.15]
    b9 = [tuple(b) for b in det9(im, 0.25)]
    src = {id(b): "v8s" for b in b8} | {id(b): "v9c" for b in b9}
    boxes = []
    for b in nms(b8 + b9, 0.4):
        if max(b[2] - b[0], b[3] - b[1]) < MIN_BOX: continue
        boxes.append({"xyxy": [round(max(0, b[0]), 1), round(max(0, b[1]), 1), round(min(W, b[2]), 1), round(min(H, b[3]), 1)],
                      "score": round(b[4], 3), "src": src[id(b)]})
    t_det += time.time() - a; a = time.time()
    if boxes:
        crops = []
        for bx in boxes:
            x0, y0, x1, y1 = bx["xyxy"]; p = 0.2 * max(x1 - x0, y1 - y0)
            crops.append(pre(im.crop((max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)))))
        with torch.inference_mode():
            I = model.encode_image(torch.stack(crops)).float(); I /= I.norm(dim=-1, keepdim=True)
            pr = (100 * I @ T.T).softmax(-1)
        pbad = pr[:, :len(BAD)].sum(-1).tolist()
        for bx, p in zip(boxes, pbad): bx["defect_hint"] = round(p, 3)
    t_clip += time.time() - a
    out.append({**r, "W": W, "H": H, "boxes": boxes})
    if (n + 1) % 250 == 0: print(n + 1, f"det {t_det/(n+1):.3f}s clip {t_clip/(n+1):.3f}s per img", flush=True)
with open(ROOT / "data/preannotations.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
nb = sum(len(r["boxes"]) for r in out)
print("done", len(out), "images", nb, "boxes", f"{time.time()-t0:.0f}s total", flush=True)
