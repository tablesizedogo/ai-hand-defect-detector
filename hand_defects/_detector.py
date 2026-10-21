"""Hand boxes with an ADetailer YOLOv8 hand detector (Bingsu/adetailer `hand_yolov8s.pt`, Apache-2.0) via ultralytics."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

HAND_DETECTOR_URL = "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt"


def ensure_detector(path: Path, allow_download: bool = False) -> Path:
    path = Path(path)
    if path.exists():
        return path
    if not allow_download:
        raise FileNotFoundError(f"hand detector not found at {path}; download {HAND_DETECTOR_URL} there "
                                "or enable allow_download in the loader node")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(HAND_DETECTOR_URL, path)
    return path


class YoloHandDetector:
    """Callable: RGB uint8 HxWx3 -> list of ([x0,y0,x1,y1], conf), sorted by conf."""

    def __init__(self, weights, device="cpu", imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(str(weights), task="detect")
        self.device, self.imgsz = device, imgsz

    def __call__(self, rgb: np.ndarray, conf: float = 0.3):
        r = self.model.predict(source=rgb[:, :, ::-1].copy(), conf=conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if not len(r.boxes):
            return []
        out = [([float(v) for v in b], float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        return sorted(out, key=lambda t: -t[1])
