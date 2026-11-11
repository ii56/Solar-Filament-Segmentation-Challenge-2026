"""Figures for the report: qualitative examples (held-out half B) and training curves."""
from __future__ import annotations

import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "src"))
import filament as F

STATS, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif"})

stats = json.load(open(STATS))
ctx = F.setup(verbose=False)
probs = F.load_val_probs(ctx, "8_0.4508")

# ---- pick examples on held-out B: good / typical / hard, one per photo -------
per = dict(stats["per_image_B"])
cand = []
for iid, (iou, tp, fp, fn, _) in per.items():
    den = tp + fp / 2 + fn / 2
    if tp + fn >= 4:                                  # images with a few real filaments
        cand.append((iou / den if den else 0, iid))
cand.sort()
picks = [cand[int(len(cand) * q)] for q in (0.9, 0.5, 0.12)]
labels = ["better than typical", "typical (median)", "harder than typical"]


def overlay(iid, thick):
    img = F.read_full_image(ctx.images_by_id, iid)
    pred, _ = F.segment_instances(probs[iid].astype(np.float32), **F.SUBMISSION_CFG)
    gt = F.gt_instances(ctx.images_by_id, ctx.anns_by_image_id, iid, F.GT_MIN_AREA)[0]
    tp_pred = F.matched_prediction_ids(pred, gt)
    tp_gt = F.matched_prediction_ids(gt, pred)
    rgb = np.dstack([img] * 3).astype(np.float32) / 255
    col = {"tp": (0.1, 0.8, 1.0), "fp": (1.0, 0.25, 0.25)}
    for i in range(1, pred.max() + 1):
        m = pred == i
        c = np.array(col["tp" if i in tp_pred else "fp"])
        rgb[m] = 0.2 * rgb[m] + 0.8 * c
    # ground truth outlines: matched = yellow, missed = green
    for g in range(1, gt.max() + 1):
        m = (gt == g).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = (1.0, 0.9, 0.0) if g in tp_gt else (0.2, 1.0, 0.2)
        cv2.drawContours(rgb, cs, -1, c, thick)
    return rgb, pred, gt


fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.75))
for k, ((pq, iid), lab) in enumerate(zip(picks, labels)):
    rgb, pred, gt = overlay(iid, 6)
    rgbz, _, _ = overlay(iid, 2)
    # zoom on the 600x600 window holding the most ground-truth + predicted pixels
    fg = ((gt > 0) | (pred > 0)).astype(np.float32)
    win = 600
    s = cv2.boxFilter(fg, -1, (win, win), normalize=False, anchor=(0, 0),
                      borderType=cv2.BORDER_CONSTANT)[: 2048 - win, : 2048 - win]
    y0, x0 = np.unravel_index(np.argmax(s), s.shape)
    ax = axes[0, k]
    ax.imshow(rgb[::2, ::2])
    ax.add_patch(Rectangle((x0 / 2, y0 / 2), win / 2, win / 2, fill=False, ec="w", lw=0.8))
    name = os.path.splitext(ctx.images_by_id[iid]["file_name"])[0]
    ax.set_title(f"{name}\n{lab}, image PQ {pq:.2f}", fontsize=7)
    axes[1, k].imshow(rgbz[y0:y0 + win, x0:x0 + win])
    for a in (ax, axes[1, k]):
        a.set_xticks([]), a.set_yticks([])
fig.legend(handles=[Patch(color=(0.1, 0.8, 1.0), label="prediction, matched (TP)"),
                    Patch(color=(1.0, 0.25, 0.25), label="prediction, unmatched (FP)"),
                    Patch(ec=(1.0, 0.9, 0.0), fc="none", label="ground truth, matched"),
                    Patch(ec=(0.2, 1.0, 0.2), fc="none", label="ground truth, missed (FN)")],
           loc="lower center", ncol=4, fontsize=7, frameon=False)
plt.tight_layout(rect=(0, 0.045, 1, 1), h_pad=0.4, w_pad=0.3)
fig.savefig(os.path.join(OUTDIR, "examples.pdf"), dpi=300)
fig.savefig(os.path.join(OUTDIR, "examples.png"), dpi=150)
print("examples:", [(round(p, 3), ctx.images_by_id[i]["file_name"]) for p, i in picks])

# ---- training curves: the whole lineage of the submitted model ----------------
h1 = json.load(open(os.path.join(F.CHECKPOINT_DIR, f"{F.TAG}_history.json")))
h2 = json.load(open(os.path.join(F.CHECKPOINT_DIR, f"{F.TAG}_cosine_history.json")))
e1 = [h["epoch"] for h in h1]
e2 = [12 + h["epoch"] for h in h2]                 # cosine run starts from epoch-12 weights
fig, ax = plt.subplots(figsize=(3.33, 2.05))
ax.plot(e1, [h["val_iou"] for h in h1], "o-", ms=2.5, lw=1, c="0.55",
        label=r"constant LR $10^{-4}$")
ax.plot(e2, [h["val_iou"] for h in h2], "o-", ms=2.5, lw=1, c="tab:blue",
        label="cosine LR (from ep. 12)")
ax.plot([20], [h2[7]["val_iou"]], "*", ms=9, c="tab:red", label="submitted")
ax.set_xlabel("cumulative training epoch"), ax.set_ylabel("val tile IoU")
ax.grid(alpha=0.3)
fig.legend(fontsize=6, loc="lower center", ncol=3, frameon=False, handlelength=1.5,
           columnspacing=0.8)
ax2 = ax.twinx()
ax2.plot(e2, [h["lr"] * 1e4 for h in h2], ":", c="tab:blue", lw=0.9)
ax2.set_ylabel(r"cosine LR ($\times10^{-4}$)", fontsize=7)
ax2.set_ylim(0, 1.6)
ax.set_ylim(0.2, 0.47)
plt.tight_layout(pad=0.3, rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(OUTDIR, "curves.pdf"))
fig.savefig(os.path.join(OUTDIR, "curves.png"), dpi=150)
print("done")
