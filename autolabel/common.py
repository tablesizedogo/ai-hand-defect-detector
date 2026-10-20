"""Shared helpers for the auto-labeling pipeline."""
import json
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]; D = R / 'autolabel_data'

def manifest():
    return [json.loads(l) for l in open(D / 'manifest.jsonl')]

def read_jsonl(path, key='key'):
    out = {}
    if Path(path).exists():
        for l in open(path):
            try: r = json.loads(l)
            except Exception: continue
            if 'error' not in r: out[r[key]] = r
    return out

def rank01(x):
    """Rank-normalise to (0,1]; NaNs stay NaN."""
    x = np.asarray(x, float); out = np.full_like(x, np.nan); m = ~np.isnan(x)
    if m.any():
        from scipy.stats import rankdata
        out[m] = rankdata(x[m]) / m.sum()
    return out

def ece(y, p, bins=10, w=None):
    y, p = np.asarray(y, float), np.asarray(p, float); w = np.ones_like(p) if w is None else np.asarray(w, float)
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.any(): e += w[m].sum() / w.sum() * abs(np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m]))
    return e
