"""Build the PNG charts used in README.md and docs/experiments/.

All numbers are copied from result files (paths in the comments) so the charts can be
rebuilt without the private data. Run: python scripts/make_doc_charts.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
EXP = ROOT / "docs" / "experiments" / "img"

BLUE, ORANGE, GREY = "#3b6ea8", "#d9822b", "#9aa5b1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})


def r2(v):
    """Round half up to 2 decimals, so 0.585 prints as 0.59 like the result tables."""
    return f"{np.floor(v * 100 + 0.5 + 1e-9) / 100:.2f}"


def hbar_pairs(names, a, b, label_a, label_b, title, path, xlim=(0.0, 1.0), note=None):
    """Horizontal grouped bars; a and b are lists of (value, lo, hi)."""
    y = np.arange(len(names))[::-1]
    h = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(names) + 1.3))
    for off, data, lab, col in ((h / 2, a, label_a, BLUE), (-h / 2, b, label_b, ORANGE)):
        v = np.array([d[0] for d in data])
        err = np.array([[d[0] - d[1] for d in data], [d[2] - d[0] for d in data]])
        ax.barh(y + off, v, height=h, color=col, label=lab, xerr=err,
                error_kw=dict(ecolor="#444", lw=1, capsize=2.5))
        for yy, vv in zip(y + off, v):
            ax.text(0.015, yy, r2(vv), va="center", ha="left", color="white",
                    fontsize=9.5, fontweight="bold")
    ax.axvline(0.5, color=GREY, ls="--", lw=1)
    ax.text(0.505, y[-1] - 0.75, "chance", color="#666", fontsize=9, ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(*xlim)
    ax.set_xlabel("AUC (bad hand = positive); bars show 95% bootstrap CI")
    ax.set_title(title, loc="left")
    ax.set_ylim(y[-1] - 0.9, y[0] + 0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2, frameon=False, fontsize=9.5)
    if note:
        fig.text(0.01, -0.02, note, fontsize=8.5, color="#555", ha="left", va="top")
    fig.savefig(path)
    plt.close(fig)


def main_results():
    # autolabel_data/gold_check_v2/eval.json and gold_check_combined/eval.json (07_ingest_eval.py)
    names = ["HADM-L", "Rank-average teacher", "CLIP student", "Detector confidence",
             "Gemma alone", "DINOv2 student"]
    v2 = [(0.7635, 0.514, 0.956), (0.6478, 0.431, 0.851), (0.5849, 0.325, 0.845), (0.5380, 0.329, 0.766),
          (0.5162, 0.234, 0.822), (0.5049, 0.264, 0.731)]
    pooled = [(0.7168, 0.539, 0.871), (0.7144, 0.582, 0.838), (0.6046, 0.443, 0.767), (0.6032, 0.451, 0.765),
              (0.6037, 0.431, 0.778), (0.5344, 0.386, 0.673)]
    hbar_pairs(names, v2, pooled, "Gold v2: 75 hands, 8 bad (held out)",
               "Gold v1+v2 pooled: 182 hands, 17 bad",
               "Which score finds broken hands on Civitai images?", IMG / "results_auc.png")


def external_vs_civitai():
    # external_experiments/results/eval_ft3/eval.md (10_eval.py --split)
    names = ["AbHuman val", "Distorted-5K test", "HAD test", "Civitai gold, all", "Civitai gold, photos"]
    hadm = [(0.680, 0.65, 0.71), (0.733, 0.70, 0.76), (0.851, 0.83, 0.87),
            (0.717, 0.53, 0.87), (0.758, 0.48, 0.95)]
    ft = [(0.816, 0.80, 0.84), (0.804, 0.78, 0.83), (0.791, 0.77, 0.82),
          (0.551, 0.36, 0.74), (0.429, 0.22, 0.70)]
    hbar_pairs(names, hadm, ft, "HADM-L (off the shelf)", "CLIP fine-tune on external data",
               "Strong on public test sets, no transfer to Civitai", IMG / "external_vs_civitai.png")


def score_hist():
    # external_experiments/results/fa (histograms of gold scores, % of hands per bin)
    ft3_ok = [2.4, 4.8, 10.9, 8.5, 18.2, 8.5, 13.3, 16.4, 15.8, 1.2]
    ft3_bad = [5.9, 0.0, 11.8, 5.9, 11.8, 0.0, 11.8, 17.6, 29.4, 5.9]
    hadm_ok = [9.7, 7.3, 11.5, 14.5, 24.8, 15.8, 11.5, 3.6, 1.2, 0.0]
    hadm_bad = [11.8, 0.0, 11.8, 0.0, 5.9, 17.6, 29.4, 11.8, 11.8, 0.0]
    x = np.arange(10) / 10 + 0.05
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, ok, bad, t in ((axes[0], ft3_ok, ft3_bad, "CLIP fine-tune"), (axes[1], hadm_ok, hadm_bad, "HADM-L")):
        ax.bar(x - 0.02, ok, width=0.04, color=BLUE, label="normal (165)")
        ax.bar(x + 0.02, bad, width=0.04, color=ORANGE, label="bad (17)")
        ax.set_title(t, loc="left")
        ax.set_xlabel("score")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("% of hands in class")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9.5)
    fig.savefig(EXP / "score_hist.png")
    plt.close(fig)


def crop_size():
    # external_experiments/NOTES.md, H4 (median box side, px; Laplacian variance of the 256 px crop)
    sets = ["Civitai gold", "AbHuman", "Distorted-5K", "HAD"]
    box = [126, 94, 70, 69]
    sharp = [1052, 462, 515, 361]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.6), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for ax, vals, t in ((axes[0], box, "Median hand box side, px"), (axes[1], sharp, "Sharpness (Laplacian variance)")):
        cols = [ORANGE] + [BLUE] * 3
        ax.barh(sets[::-1], vals[::-1], color=cols[::-1])
        for i, v in enumerate(vals[::-1]):
            ax.text(v, i, f" {v}", va="center", fontsize=9.5)
        ax.set_title(t, loc="left")
        ax.set_xlim(0, max(vals) * 1.22)
    fig.savefig(EXP / "crop_size.png")
    plt.close(fig)


def recrop():
    # external_experiments/results/fa/h4_auc.json (weighted AUC on the 53 photo hands, 8 bad)
    cfg = [("Stored JPEG crop (official)", 0.429), ("1.5x, no JPEG (as trained)", 0.508),
           ("Tighter 1.2x", 0.483), ("Tighter 1.3x", 0.518), ("Wider 2.0x", 0.516),
           ("Wider 2.5x", 0.493), ("Downscaled via 64 px", 0.485), ("Downscaled via 112 px", 0.471),
           ("JPEG quality 30", 0.576), ("Grayscale", 0.549), ("Light blur", 0.546)]
    names = [c[0] for c in cfg][::-1]
    vals = [c[1] for c in cfg][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(names, vals, color=[ORANGE if n.startswith("Stored") else BLUE for n in names])
    for i, v in enumerate(vals):
        ax.text(v, i, " " + r2(v), va="center", fontsize=9.5)
    ax.axvline(0.5, color=GREY, ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC of the CLIP fine-tune on gold photos (53 hands, 8 bad)")
    ax.set_title("Re-cropping does not fix it", loc="left")
    fig.savefig(EXP / "recrop_auc.png")
    plt.close(fig)


def training_curve():
    # external_experiments/results/ft3/ft_history.json (epochs stored 0-based; shown 1-based)
    ep = [1, 2, 3]
    loss = [0.5569, 0.4897, 0.4634]
    dev = {"AbHuman": [0.8264, 0.8381, 0.8390], "HAD": [0.8059, 0.8179, 0.8216],
           "Distorted-5K": [0.7610, 0.7592, 0.7545]}
    mean_sets = [0.79777, 0.80505, 0.80505]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    ax.plot(ep, loss, "o-", color=BLUE)
    for x, v in zip(ep, loss):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9.5)
    ax.set_xticks(ep); ax.set_xlabel("epoch"); ax.set_ylim(0.40, 0.60)
    ax.set_title("Training loss (BCE, mean per step)", loc="left")
    ax = axes[1]
    for (name, v), c in zip(dev.items(), ["#7fa7d1", "#a7b8c8", "#c9a27a"]):
        ax.plot(ep, v, "o-", color=c, lw=1.3, ms=4, label=name)
    ax.plot(ep, mean_sets, "o-", color=ORANGE, lw=2.4, label="mean of sets")
    ax.axvline(2, color=GREY, ls="--", lw=1)
    ax.annotate("kept: epoch 2\nmean AUC 0.805", (2, mean_sets[1]), textcoords="offset points", xytext=(10, -42),
                fontsize=9.5, color="#333")
    ax.set_xticks(ep); ax.set_xlabel("epoch"); ax.set_ylim(0.74, 0.85)
    ax.set_title("Dev AUC by test set", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8.5, ncol=4)
    fig.text(0.01, -0.01, "Epoch 3 lowers the loss, but the mean dev AUC stays at 0.805 (change below 0.00001), "
             "so selection kept epoch 2. Dev = 4,000 held-out hands.", fontsize=8.5, color="#555", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(EXP / "training_curve.png")
    plt.close(fig)


def training_pipeline():
    # external_experiments/kaggle/k7_ft3/hdx_ft3.py, results/ft3/ft_history.json
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.set_xlim(0, 110); ax.set_ylim(0, 39); ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#555", fs=9.5, tc="#111", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2))

    data = [("AbHuman\n18,119 hands", 28), ("HAD\n31,401 hands", 17), ("Distorted-5K\n4,428 hands", 6)]
    for t, y in data:
        box(0, y, 15, 8, t, "#eef3f9")
        arrow(15.5, y + 4, 20.5, 20.5)
    box(21, 14, 15, 13, "Balanced sampler\nset share ∝ √n\n50/50 normal/bad\n32,000 per epoch", "#f4f4f4")
    arrow(36.5, 20.5, 39.5, 20.5)
    box(40, 14, 12, 13, "Augment\nrandom crop\nflip, colour\n224 × 224", "#f4f4f4")
    arrow(52.5, 20.5, 55.5, 20.5)
    # CLIP-L tower
    ax.add_patch(FancyBboxPatch((56, 6), 31, 29, boxstyle="round,pad=0.3,rounding_size=1.5", fc="white", ec="#555", lw=1))
    ax.text(71.5, 33, "CLIP ViT-L/14 image tower", ha="center", va="center", fontsize=10, fontweight="bold")
    box(58, 9, 12, 20, "patch embed\n+ blocks 1–20\n\nfrozen", "#d9dde2", fs=9)
    box(72, 9, 13, 20, "blocks 21–24\nfinal norm\nprojection\n\ntrained", "#f6d2ae", ec=ORANGE, fs=9)
    arrow(70.3, 19, 71.7, 19)
    arrow(87.5, 20.5, 90.5, 20.5)
    box(91, 15, 8, 11, "linear\nhead\n768 → 1", "#f6d2ae", ec=ORANGE, fs=9)
    arrow(99.5, 20.5, 101.5, 20.5)
    box(102, 15.5, 7.5, 10, "P(bad)", "#eef3f9", bold=True)
    ax.text(0, 1.5, "Loss: binary cross-entropy on P(bad).  AdamW: lr 1e-5 for the trained blocks, 1e-3 for the head.  "
            "Batch 64, 3 epochs, one T4 GPU.", fontsize=9, color="#555")
    fig.savefig(EXP / "training_pipeline.png")
    plt.close(fig)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    main_results()
    external_vs_civitai()
    score_hist()
    crop_size()
    recrop()
    training_curve()
    training_pipeline()
    print("charts written to", IMG, "and", EXP)
