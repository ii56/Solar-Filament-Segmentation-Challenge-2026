"""Numbers for the technical report, computed on CPU from cached val probability maps."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "src"))
import filament as F

OUT = sys.argv[1]
ctx = F.setup(verbose=False)
A, B = F.split_val_by_file(ctx)
fname = {i: ctx.images_by_id[i]["file_name"] for i in ctx.val_ids}
gt = {i: F.gt_instances(ctx.images_by_id, ctx.anns_by_image_id, i, F.GT_MIN_AREA)[0]
      for i in ctx.val_ids}

CFGS = {
    "none": dict(low=0.5, min_area=1),
    "default": dict(low=0.5, min_area=20),
    "minarea": dict(low=0.3, min_area=800),
    "hyst_noarea": dict(low=0.4, high=0.995, min_area=20),
    "hyst": dict(low=0.4, high=0.995, min_area=400),
}
MODELS = {"run2_ep12": "12_0.4515", "cos_ep8": "8_0.4508", "cos_ep10": "10_0.4508"}


def per_image(probs, cfg):
    rows = {}
    for iid in ctx.val_ids:
        pred, _ = F.segment_instances(np.asarray(probs[iid], np.float32), **cfg)
        m, tp, fp, fn = F.match_instances(pred, gt[iid])
        p, g = pred > 0, gt[iid] > 0
        d = int(p.sum()) + int(g.sum())
        dice = 1.0 if d == 0 else 2 * int((p & g).sum()) / d
        rows[iid] = (sum(m), tp, fp, fn, dice)
    return rows


def pooled(rows, ids):
    s = np.array([rows[i][:4] for i in ids], dtype=float).sum(0)
    iou, tp, fp, fn = s
    den = tp + fp / 2 + fn / 2
    return {"pq": iou / den if den else 1.0, "sq": iou / tp if tp else 0.0,
            "rq": tp / den if den else 1.0, "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "dice": float(np.mean([rows[i][4] for i in ids]))}


def boot(rows, ids, n=2000, seed=0):
    """Bootstrap over PHOTOS (all annotator versions of a photo move together)."""
    by_file = defaultdict(list)
    for i in ids:
        by_file[fname[i]].append(i)
    files = sorted(by_file)
    arr = {f: np.array([rows[i][:4] for i in by_file[f]], float).sum(0) for f in files}
    dic = {f: [rows[i][4] for i in by_file[f]] for f in files}
    rng = np.random.RandomState(seed)
    pqs, dices = [], []
    for _ in range(n):
        pick = rng.randint(0, len(files), len(files))
        s = sum(arr[files[k]] for k in pick)
        iou, tp, fp, fn = s
        pqs.append(iou / (tp + fp / 2 + fn / 2))
        dices.append(np.mean([d for k in pick for d in dic[files[k]]]))
    return {"pq_std": float(np.std(pqs)), "pq_ci": [float(np.percentile(pqs, 2.5)),
                                                     float(np.percentile(pqs, 97.5))],
            "dice_std": float(np.std(dices)), "dice_ci": [float(np.percentile(dices, 2.5)),
                                                           float(np.percentile(dices, 97.5))]}


res = {"n": {"A_ids": len(A), "B_ids": len(B), "A_files": len({fname[i] for i in A}),
             "B_files": len({fname[i] for i in B}), "val_ids": len(ctx.val_ids),
             "val_files": len({fname[i] for i in ctx.val_ids}),
             "train_ids": len(ctx.train_ids),
             "train_files": len({ctx.images_by_id[i]["file_name"] for i in ctx.train_ids}),
             "train_tiles": len(ctx.train_dataset), "val_tiles": len(ctx.val_dataset)}}

# --- post-processing ablation on the submitted model -------------------------
probs = F.load_val_probs(ctx, MODELS["cos_ep8"])
rows_by_cfg = {}
res["ablation"] = {}
for name, cfg in CFGS.items():
    rows = per_image(probs, cfg)
    rows_by_cfg[name] = rows
    res["ablation"][name] = {"A": pooled(rows, A), "B": pooled(rows, B),
                             "all": pooled(rows, ctx.val_ids), "B_boot": boot(rows, B),
                             "all_boot": boot(rows, ctx.val_ids)}
    print(name, {k: round(v["pq"], 4) for k, v in res["ablation"][name].items()
                 if "boot" not in k}, flush=True)

# --- checkpoints, same post-processing ---------------------------------------
res["models"] = {}
for key, name in MODELS.items():
    pr = probs if key == "cos_ep8" else F.load_val_probs(ctx, name)
    rows = rows_by_cfg["hyst"] if key == "cos_ep8" else per_image(pr, CFGS["hyst"])
    res["models"][key] = {"A": pooled(rows, A), "B": pooled(rows, B),
                          "all": pooled(rows, ctx.val_ids), "B_boot": boot(rows, B)}
    print(key, round(res["models"][key]["B"]["pq"], 4), flush=True)

# --- per GONG site (letter before the trailing 'h') ---------------------------
rows = rows_by_cfg["hyst"]
site = defaultdict(list)
for i in ctx.val_ids:
    site[os.path.splitext(fname[i])[0][-2]].append(i)
res["per_site"] = {s: {**pooled(rows, ids), "n_ids": len(ids)} for s, ids in sorted(site.items())}

# --- annotator agreement: PQ of one annotator's labels against another's -------
by_file = defaultdict(list)
for i in ctx.val_ids:
    by_file[fname[i]].append(i)
h_rows, m_rows = [], []
for f, ids in by_file.items():
    if len(ids) < 2:
        continue
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            m, tp, fp, fn = F.match_instances(gt[ids[a]], gt[ids[b]])
            h_rows.append((sum(m), tp, fp, fn))
    for i in ids:                       # model on the same photos, for a like-for-like
        r = rows[i]
        m_rows.append(r[:4])


def pq_of(rs):
    iou, tp, fp, fn = np.array(rs, float).sum(0)
    return {"pq": iou / (tp + fp / 2 + fn / 2), "sq": iou / tp, "rq": tp / (tp + fp / 2 + fn / 2),
            "pairs": len(rs)}


res["annotator_agreement"] = {"human_vs_human": pq_of(h_rows), "model_same_photos": pq_of(m_rows),
                              "n_multi_files": sum(len(v) > 1 for v in by_file.values())}

# per-image PQ on B for picking example images
res["per_image_B"] = {str(i): rows[i] for i in B}
with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k not in ("per_image_B",)}, indent=1,
                 default=float))
