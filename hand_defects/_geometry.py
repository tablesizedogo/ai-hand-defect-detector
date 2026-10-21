"""Box / region / mask helpers. Boxes are [x0, y0, x1, y1] in pixels of the image being scored."""
from __future__ import annotations

import numpy as np


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    i = intersection(a, b)
    return i / max(1e-9, area(a) + area(b) - i)


def square_region(box, width, height, pad=1.3):
    """Square around a hand box with side = pad * max(w, h), clipped to the image.
    Same construction as the training crops (1.3x, see autolabel/01_crop_all.py)."""
    x0, y0, x1, y1 = box
    s = max(x1 - x0, y1 - y0) * pad / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [float(max(0, int(cx - s))), float(max(0, int(cy - s))),
            float(min(width, int(cx + s))), float(min(height, int(cy + s)))]


def region_artifact_score(detections, region, cls="hand", min_inside=0.5):
    """Rule chosen on the gold check: max score of `cls` artifact detections with
    >= `min_inside` of their own area inside `region`; 0.0 if none."""
    best = 0.0
    for d in detections:
        if cls is not None and d["cls"] != cls:
            continue
        a = area(d["box"])
        if a > 0 and intersection(d["box"], region) / a >= min_inside:
            best = max(best, float(d["score"]))
    return best


def box_mask(height, width, boxes, grow=0.15):
    """Float32 HxW mask (1 inside) covering each box enlarged by `grow` * box size on every side."""
    m = np.zeros((height, width), np.float32)
    for b in boxes:
        w, h = b[2] - b[0], b[3] - b[1]
        x0, y0 = int(max(0, b[0] - grow * w)), int(max(0, b[1] - grow * h))
        x1, y1 = int(min(width, np.ceil(b[2] + grow * w))), int(min(height, np.ceil(b[3] + grow * h)))
        m[y0:y1, x0:x1] = 1.0
    return m


def mask_overlaps(mask, box, min_frac=0.05):
    """True if at least `min_frac` of the box area is covered by mask > 0.5."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    sub = mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    return sub.size > 0 and float((sub > 0.5).mean()) >= min_frac


def fit_to(candidate, height, width, background):
    """Place `candidate` (h, w, 3) on a copy of `background` (height, width, 3).
    ComfyUI's VAEEncodeForInpaint centre-crops images to a multiple of 8, so a slightly smaller
    candidate is pasted centred; a candidate of a different scale is resized."""
    h, w = candidate.shape[:2]
    if (h, w) == (height, width):
        return candidate
    if 0 <= height - h < 8 and 0 <= width - w < 8:
        out = background.copy()
        oy, ox = (height - h) // 2, (width - w) // 2
        out[oy:oy + h, ox:ox + w] = candidate
        return out
    from PIL import Image
    im = Image.fromarray((np.clip(candidate, 0, 1) * 255).astype(np.uint8)).resize((width, height), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0
