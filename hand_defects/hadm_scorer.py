"""Score every hand in an image with HADM-L: hand boxes from the ADetailer YOLOv8 hand detector, then the calibrated
probability that the hand is malformed.

    python -m hand_defects.hadm_scorer img1.jpg img2.png --models-dir ~/.cache/hand_fixer [--download] [--device cuda]

Rule (same as the evaluation in autolabel/09_hadm_kaggle): for each hand box take the square region of 1.3x its longer
side; raw = max HADM `hand` artifact score over detections with >= 50% of their area inside that region (0 if none);
p_bad = sigmoid(a*raw + b) with a, b from models/hadm_calibration.json (Platt on 182 human-labelled gold hands, 17 BAD).

HADM (Wang et al., arXiv:2411.13842, github.com/wangkaihong/HADM) has no LICENSE file, so its code is cloned from GitHub
at a pinned commit and its weights are fetched from the authors' link; nothing of HADM is vendored here.
The files _hadm_backend.py, _geometry.py and _detector.py are byte-identical copies of the ComfyUI node pack's
hand_fixer/{hadm_backend,geometry,detector}.py (tests/test_sync.py checks this when the pack is present).

Models dir layout:  <models-dir>/hand_yolov8s.pt, <models-dir>/hadm/HADM/ (git clone), <models-dir>/hadm/HADM-L_0249999.pth
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ._detector import YoloHandDetector, ensure_detector
from ._geometry import region_artifact_score, square_region
from ._hadm_backend import HADMDetector, ensure_repo, ensure_weights

CALIBRATION = Path(__file__).resolve().parent.parent / "models" / "hadm_calibration.json"
DEFAULT_MODELS = Path(os.environ.get("HAND_FIXER_MODELS", Path.home() / ".cache" / "hand_fixer"))


@dataclass
class Hand:
    box: list            # hand box [x0, y0, x1, y1] from the hand detector
    det_conf: float
    region: list         # square scoring region
    hadm_raw: float      # max HADM hand-artifact score inside the region (0 = nothing detected)
    p_bad: float         # calibrated P(malformed)
    bad: bool            # p_bad >= threshold


class HADMHandScorer:
    def __init__(self, models_dir: Path = DEFAULT_MODELS, device: str = "cuda", allow_download: bool = False,
                 calibration: Path = CALIBRATION, min_inside: float = 0.5, pad: float = 1.3):
        models_dir = Path(models_dir)
        if allow_download:
            ensure_repo(models_dir / "hadm")
            ensure_weights(models_dir / "hadm", allow_download=True)
        self.hands = YoloHandDetector(ensure_detector(models_dir / "hand_yolov8s.pt", allow_download), device=device)
        self.hadm = HADMDetector(models_dir / "hadm", device=device, allow_download=allow_download)
        c = json.load(open(calibration))
        self.a, self.b = c["a"], c["b"]
        self.min_inside, self.pad = min_inside, pad

    def p_bad(self, raw: float) -> float:
        return 1.0 / (1.0 + math.exp(-(self.a * raw + self.b)))

    def score_regions(self, rgb: np.ndarray, regions: list) -> list:
        """raw HADM hand scores for given square regions (one HADM pass per image)."""
        if not regions:
            return []
        dets = self.hadm(rgb)
        return [region_artifact_score(dets, r, "hand", self.min_inside) for r in regions]

    def __call__(self, rgb: np.ndarray, det_conf: float = 0.3, threshold: float = 0.2) -> list:
        """rgb: uint8 HxWx3. Returns one Hand per detected hand."""
        h, w = rgb.shape[:2]
        found = self.hands(rgb, conf=det_conf)
        regions = [square_region(b, w, h, self.pad) for b, _ in found]
        raws = self.score_regions(rgb, regions)
        out = []
        for (b, c), r, x in zip(found, regions, raws):
            p = self.p_bad(x)
            out.append(Hand([round(v, 1) for v in b], round(float(c), 4), [round(v, 1) for v in r], round(float(x), 4),
                            round(p, 4), p >= threshold))
        return out


def main(argv=None):
    from PIL import Image
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--models-dir", default=str(DEFAULT_MODELS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--download", action="store_true", help="fetch missing models (HADM-L weights are 5.7 GB)")
    ap.add_argument("--det-conf", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=0.2, help="P(bad) threshold; corpus base rate is ~0.10")
    a = ap.parse_args(argv)
    scorer = HADMHandScorer(Path(a.models_dir), a.device, a.download)
    for p in a.images:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        hands = scorer(rgb, a.det_conf, a.threshold)
        print(json.dumps(dict(image=p, hands=[asdict(h) for h in hands])), flush=True)


if __name__ == "__main__":
    sys.exit(main())
