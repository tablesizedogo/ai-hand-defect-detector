"""CPU hand detectors used for pre-annotation (benchmarked in 02_benchmark.py)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
W = ROOT / "weights"


class YoloHand:
    """ADetailer hand detectors (Bingsu/adetailer on HF hub, trained on real + anime hands)."""

    def __init__(self, name="hand_yolov8s.pt", imgsz=640):
        from ultralytics import YOLO
        self.m = YOLO(str(W / name))
        self.imgsz = imgsz
        self.name = name.replace(".pt", "")

    def __call__(self, im: Image.Image, conf=0.15):
        r = self.m.predict(im, imgsz=self.imgsz, conf=conf, verbose=False, device="cpu")[0]
        b = r.boxes
        return [(*map(float, xyxy), float(c)) for xyxy, c in zip(b.xyxy.numpy(), b.conf.numpy())]


class GDino:
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", prompt="a hand."):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.m = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).eval()
        self.prompt = prompt
        self.name = "gdino_tiny"

    def __call__(self, im: Image.Image, conf=0.2):
        import torch
        inp = self.proc(images=im, text=self.prompt, return_tensors="pt")
        with torch.inference_mode():
            out = self.m(**inp)
        res = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=conf, text_threshold=conf, target_sizes=[im.size[::-1]])[0]
        return [(*map(float, b), float(s)) for b, s in zip(res["boxes"].numpy(), res["scores"].numpy())]


class MPHands:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(W / "hand_landmarker.task")),
                                            num_hands=6, min_hand_detection_confidence=0.2)
        self.det = vision.HandLandmarker.create_from_options(opts)
        self.mp = mp
        self.name = "mediapipe"

    def __call__(self, im: Image.Image, conf=0.2):
        arr = np.asarray(im.convert("RGB"))
        r = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=arr))
        w, h = im.size
        out = []
        for lms, hd in zip(r.hand_landmarks, r.handedness):
            xs = [p.x * w for p in lms]; ys = [p.y * h for p in lms]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            pad = 0.15 * max(x1 - x0, y1 - y0)
            out.append((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad), float(hd[0].score)))
        return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def nms(boxes, thr=0.5):
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep = []
    for b in boxes:
        if all(iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep
