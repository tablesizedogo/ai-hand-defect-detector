"""Step 9.4: per-crop HADM-L scores from the full-set Kaggle run -> autolabel_data/hadm_hand.jsonl.
hadm_hand_iod50 = max score of HADM `hand`-artifact detections with >= 50% of their area inside the padded crop box (0 if none);
this is the rule used by 04_fuse.py --teacher hadm_gemma_det. Also stored: IoU>=0.1 with the hand box, and the whole-image max.
Higher = more likely BAD. Boxes are in the coordinates of the image HADM saw (input_all/boxes.json)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import D
B = json.load(open(D / 'hadm/input_all/boxes.json')); det = json.load(open(D / 'hadm/out_all/hadm_detections.json'))['detections']
def inter(b, c): return max(0, min(b[2], c[2]) - max(b[0], c[0])) * max(0, min(b[3], c[3]) - max(b[1], c[1]))
def area(b): return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
def iou(b, c): i = inter(b, c); return i / max(1e-9, area(b) + area(c) - i)
n = 0
with open(D / 'hadm_hand.jsonl', 'w') as f:
    for k, c in B['crops'].items():
        d = det[c['image']]; H = [(b, s) for b, s, cl in zip(d['bbox'], d['score'], d['class']) if cl == 'hand']
        f.write(json.dumps(dict(key=k, image=c['image'],
            hadm_hand_iod50=max([s for b, s in H if inter(b, c['crop_xyxy']) / max(1e-9, area(b)) >= 0.5], default=0.0),
            hadm_hand_iou01=max([s for b, s in H if iou(b, c['hand_xyxy']) >= 0.1], default=0.0),
            hadm_hand_image=max([s for b, s in H], default=0.0))) + '\n'); n += 1
print('wrote', n, 'crops')
