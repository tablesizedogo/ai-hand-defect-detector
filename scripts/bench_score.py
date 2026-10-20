"""Score benchmark detectors against a hand GT built by visually verifying the union of all detectors'
boxes on 48 images (data/bench_gt.json). GT is therefore 'hands found by at least one detector'."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from detectors import iou, nms
ROOT = Path(__file__).resolve().parent.parent
raw = json.load(open(ROOT / "data/bench_raw.json")); sample = json.load(open(ROOT / "data/bench_sample.json"))
union = json.load(open(ROOT / "data/bench_union.json"))
# per sample index: list of hand groups (indices into union boxes); None = ambiguous, excluded
G = {0: [[0], [3]], 1: [[0]], 2: [[0], [1]], 3: [[0], [1, 2]], 4: [[0], [1]], 5: [[0], [1]], 6: [[0, 1]], 7: [[0, 1]],
     8: [[0]], 9: [[0]], 10: [[0], [1]], 11: [[0]],
     12: [[0, 3], [1]], 13: [[0], [1]], 14: [[0]], 15: [], 16: [[0]], 17: [[6], [0]], 18: [], 19: [[0], [1]],
     20: [[1, 2]], 21: [[2], [0]], 22: None, 23: [[0], [1]],
     24: [[1], [0]], 25: None, 26: [[0, 1]], 27: [], 28: [[1]], 29: None, 30: [[0, 1]], 31: [[1, 2]],
     32: [[0], [4]], 33: None, 34: [[0], [1, 3, 4]], 35: [[2, 3], [4]],
     36: [], 37: None, 38: [[0], [1]], 39: None, 40: [], 41: [[0], [1]], 42: [[0], [1], [2]], 43: [], 44: [[0]],
     45: None, 46: [[0, 1]], 47: []}
json.dump({str(sample[k]["id"]): v for k, v in G.items()}, open(ROOT / "data/bench_gt.json", "w"))

def match(b, grp, ub):
    return any(iou(b, ub[i]) >= 0.3 or (max(b[0], ub[i][0]) < min(b[2], ub[i][2]) and max(b[1], ub[i][1]) < min(b[3], ub[i][3])
               and iou(b, ub[i]) >= 0.15) for i in grp)

def score(get_boxes, thr):
    res = {}
    for st in ["photo", "anime", "art", "all"]:
        tp_h = n_h = tp_b = n_b = 0
        for k, grps in G.items():
            r = sample[k]
            if grps is None or (st != "all" and r["style"] != st): continue
            ub = union[str(r["id"])]; boxes = [b for b in get_boxes(str(r["id"])) if b[4] >= thr]
            n_h += len(grps); tp_h += sum(any(match(b, g, ub) for b in boxes) for g in grps)
            n_b += len(boxes); tp_b += sum(any(match(b, g, ub) for g in grps) for b in boxes)
        res[st] = (round(tp_h / max(n_h, 1), 2), round(tp_b / max(n_b, 1), 2), n_h)
    return res

out = {}
for name, d in raw.items():
    for thr in ([0.25, 0.3, 0.35, 0.4] if name == "gdino_tiny" else [0.15, 0.25, 0.35, 0.5]):
        s = score(lambda k: d["boxes"][k], thr); out[f"{name}@{thr}"] = s
        print(f"{name:14s} conf>={thr:<4} " + "  ".join(f"{st}: R={v[0]:.2f} P={v[1]:.2f}" for st, v in s.items()), f"[{d['sec_per_img']}s/img]")
def combo(a, ta, b, tb):
    return lambda k: nms([x for x in raw[a]["boxes"][k] if x[4] >= ta] + [x for x in raw[b]["boxes"][k] if x[4] >= tb], 0.4)
for (a, ta, b, tb) in [("hand_yolov8s", 0.25, "hand_yolov9c", 0.25), ("hand_yolov8s", 0.25, "mediapipe", 0.3),
                       ("hand_yolov9c", 0.25, "gdino_tiny", 0.35), ("hand_yolov8s", 0.25, "gdino_tiny", 0.35), ("hand_yolov8n", 0.25, "hand_yolov8s", 0.25)]:
    s = score(combo(a, ta, b, tb), 0); out[f"{a}+{b}"] = s
    print(f"UNION {a}@{ta}+{b}@{tb}: " + "  ".join(f"{st}: R={v[0]:.2f} P={v[1]:.2f}" for st, v in s.items()))
print("GT hands:", {st: v[2] for st, v in s.items()})
json.dump(out, open(ROOT / "data/bench_scores.json", "w"), indent=1)
for (ta, tb) in [(0.15, 0.15), (0.15, 0.25), (0.2, 0.2)]:
    s = score(combo("hand_yolov8s", ta, "hand_yolov9c", tb), 0); out[f"v8s@{ta}+v9c@{tb}"] = s
    print(f"UNION v8s@{ta}+v9c@{tb}: " + "  ".join(f"{st}: R={v[0]:.2f} P={v[1]:.2f}" for st, v in s.items()))
json.dump(out, open(ROOT / "data/bench_scores.json", "w"), indent=1)
