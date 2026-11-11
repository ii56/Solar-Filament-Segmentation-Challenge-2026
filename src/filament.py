"""Solar Filament Segmentation - shared pipeline.

Import this instead of re-running notebook cells. The expensive step (building
the tile index) is cached to disk, so a fresh kernel is ready in seconds.

    import sys; sys.path.insert(0, "../src")
    import filament as F
    ctx = F.setup()                       # data + splits + datasets (cached)
    model, ck = F.load_model(ctx, "best")

This file is the single source of truth for preprocessing. The trained
checkpoints depend on it - changing the CLAHE settings or the mask
rasterisation silently invalidates them.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as maskUtils
from shapely.geometry import Polygon, box
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 'Data' is capitalised on disk; Windows ignores case but Linux does not.
DATA_ROOT = os.path.join(PROJECT_ROOT, "Data", "filament-segmentation-2026",
                         "MAGFiLO_1.0_Kaggle_2026")
TRAIN_IMAGES = os.path.join(DATA_ROOT, "train", "train_images")
ANNOTATIONS = os.path.join(DATA_ROOT, "train",
                           "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

TILE_SIZE = 512
STRIDE = 384                  # 512 - 128 overlap
VAL_FRACTION = 0.15
SPLIT_SEED = 42
# Chosen by eye against grid 32x32 / 64x64 and higher clip limits, which
# amplified the granulation texture into visible noise.
CLAHE_CLIP_LIMIT = 1.0
CLAHE_TILE_GRID = (16, 16)
# Smallest ground-truth instance that counts when scoring PQ. Fixed, so
# tuning the prediction min_area cannot quietly shrink the scoring set.
GT_MIN_AREA = 20


# ----------------------------------------------------------------- geometry
def generate_tile_coords(img_width, img_height, tile_size=TILE_SIZE, stride=STRIDE):
    """Top-left corners covering the image, including the right/bottom edges."""
    if img_width < tile_size or img_height < tile_size:
        return [(0, 0)]
    xs = list(range(0, img_width - tile_size + 1, stride))
    ys = list(range(0, img_height - tile_size + 1, stride))
    if xs[-1] + tile_size < img_width:
        xs.append(img_width - tile_size)
    if ys[-1] + tile_size < img_height:
        ys.append(img_height - tile_size)
    return [(x0, y0) for y0 in ys for x0 in xs]


def seg_to_flat_rings(segmentation):
    """COCO 'segmentation' -> list of flat [x0,y0,x1,y1,...] rings.

    Returns every ring, not just the first: a filament split by the image edge
    is stored as several polygon parts.
    """
    if not segmentation:
        return []
    if isinstance(segmentation[0], (list, tuple)):
        return [list(r) for r in segmentation if r is not None and len(r) >= 6]
    return [list(segmentation)] if len(segmentation) >= 6 else []


def safe_polygon(flat_ring):
    """Valid shapely Polygon from a flat ring, or None if degenerate."""
    coords = list(zip(flat_ring[0::2], flat_ring[1::2]))
    if len(coords) < 3:
        return None
    try:
        poly = Polygon(coords)
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)           # self-intersections -> valid geometry
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def _polygon_parts(geom):
    """Flatten any geometry to its Polygon members, recursing into collections."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        parts = []
        for g in geom.geoms:
            parts.extend(_polygon_parts(g))
        return parts
    return []                           # Point / LineString -> no area


def clip_annotation_to_tile(segmentation, x0, y0, tile_size, min_area=4.0):
    """Clip one annotation to a tile; flat rings in tile-local coords."""
    tile_box = box(x0, y0, x0 + tile_size, y0 + tile_size)
    out = []
    for ring in seg_to_flat_rings(segmentation):
        poly = safe_polygon(ring)
        if poly is None:
            continue
        try:
            clipped = poly.intersection(tile_box)
        except Exception:
            continue
        for part in _polygon_parts(clipped):
            if part.area < min_area:
                continue
            local = [(px - x0, py - y0) for px, py in part.exterior.coords]
            if len(local) < 3:
                continue
            out.append([c for point in local for c in point])
    return out


# ------------------------------------------------------------------ dataset
class FilamentTileDataset(Dataset):
    def __init__(self, image_ids, images_by_id, anns_by_image_id, image_dir,
                 tile_size=TILE_SIZE, stride=STRIDE, skip_empty_prob=0.5,
                 use_clahe=True, samples=None):
        """skip_empty_prob: chance of DROPPING a tile with zero filaments.

        Keeps some empty tiles (so the model learns background) without
        drowning in them. Val should always pass 0.0.
        """
        self.image_dir = image_dir
        self.tile_size = tile_size
        self.images_by_id = images_by_id
        self.anns_by_image_id = anns_by_image_id
        self.use_clahe = use_clahe
        self._clahe = None              # cv2 CLAHE is unpicklable; build lazily

        if samples is not None:
            self.samples = [tuple(s) for s in samples]
            return

        self.samples = []
        rng = np.random.RandomState(SPLIT_SEED)
        for image_id in image_ids:
            info = images_by_id[image_id]
            polys = []
            for ann in anns_by_image_id.get(image_id, []):
                for ring in seg_to_flat_rings(ann["segmentation"]):
                    p = safe_polygon(ring)
                    if p is not None:
                        polys.append(p)
            for (x0, y0) in generate_tile_coords(info["width"], info["height"],
                                                 tile_size, stride):
                tb = box(x0, y0, x0 + tile_size, y0 + tile_size)
                if not any(p.intersects(tb) for p in polys):
                    if rng.rand() < skip_empty_prob:
                        continue
                self.samples.append((image_id, x0, y0))

    @property
    def clahe(self):
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                          tileGridSize=CLAHE_TILE_GRID)
        return self._clahe

    def __len__(self):
        return len(self.samples)

    def read_tile(self, idx, apply_clahe=None):
        """Raw uint8 tile, before tensor conversion."""
        image_id, x0, y0 = self.samples[idx]
        info = self.images_by_id[image_id]
        full = np.array(Image.open(os.path.join(self.image_dir,
                                                info["file_name"])).convert("L"))
        tile = full[y0:y0 + self.tile_size, x0:x0 + self.tile_size]
        if tile.shape != (self.tile_size, self.tile_size):
            pad = np.zeros((self.tile_size, self.tile_size), dtype=tile.dtype)
            pad[:tile.shape[0], :tile.shape[1]] = tile
            tile = pad
        if apply_clahe is None:
            apply_clahe = self.use_clahe
        if apply_clahe:
            tile = self.clahe.apply(np.ascontiguousarray(tile, dtype=np.uint8))
        return tile

    def build_mask(self, idx):
        """Binary uint8 mask (0/255) for the tile at idx."""
        image_id, x0, y0 = self.samples[idx]
        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
        for ann in self.anns_by_image_id.get(image_id, []):
            for flat in clip_annotation_to_tile(ann["segmentation"], x0, y0,
                                                self.tile_size):
                rles = maskUtils.frPyObjects([flat], self.tile_size, self.tile_size)
                mask[maskUtils.decode(maskUtils.merge(rles)) > 0] = 255
        return mask

    def __getitem__(self, idx):
        img = torch.from_numpy(self.read_tile(idx)).float().unsqueeze(0) / 255.0
        msk = torch.from_numpy(self.build_mask(idx)).float().unsqueeze(0) / 255.0
        return img, msk


# --------------------------------------------------------------------- loss
class BCETverskyLoss(torch.nn.Module):
    """BCE + Tversky.

    Tversky alone collapses to all-background: on an empty tile, predicting
    nothing drops the loss ~1.0 -> 0, while on a filament tile it only costs
    ~0.02. With half the tiles empty, suppression wins, logits saturate near
    -10, and every region loss gradient carries a p(1-p) factor that vanishes
    there. The BCE gradient for a missed positive stays at (p-1) ~ -1, keeping
    the escape route open; Tversky supplies the asymmetry (beta > alpha
    penalises false negatives harder).
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, tversky_weight=1.0):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.smooth, self.tversky_weight = smooth, tversky_weight
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # float32 regardless of autocast: these sums reach ~262k and would
        # overflow fp16 (max 65504) on a densely-positive tile.
        logits, targets = logits.float(), targets.float()
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1 - targets)).sum(dim=dims)
        fn = ((1 - probs) * targets).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn
                                        + self.smooth)
        return bce_loss + self.tversky_weight * (1 - tversky).mean()


# -------------------------------------------------------------------- setup
class Context(dict):
    """Plain dict with attribute access, so ctx.train_dataset reads nicely."""

    __getattr__ = dict.__getitem__


def load_coco():
    with open(ANNOTATIONS) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image_id = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image_id[ann["image_id"]].append(ann)
    return coco, images_by_id, dict(anns_by_image_id)


def make_splits(coco):
    """Split by physical file_name, not image_id.

    Several annotators can produce multiple image_id entries for one photo;
    splitting on image_id would leak the same photo across train and val.
    """
    file_to_ids = defaultdict(list)
    for img in coco["images"]:
        file_to_ids[img["file_name"]].append(img["id"])

    # Must match the original notebook exactly: insertion order (the order of
    # coco['images']), then np.random.seed(42) + shuffle. Sorting first would
    # produce a DIFFERENT split, invalidating every trained checkpoint and
    # leaking train images into val.
    np.random.seed(SPLIT_SEED)
    files = list(file_to_ids.keys())
    np.random.shuffle(files)
    n_val = int(len(files) * VAL_FRACTION)
    val_files, train_files = set(files[:n_val]), set(files[n_val:])

    train_ids = [i for f in train_files for i in file_to_ids[f]]
    val_ids = [i for f in val_files for i in file_to_ids[f]]
    return sorted(train_ids), sorted(val_ids), file_to_ids


def _cache_path(name, image_ids, tile_size, stride, skip_empty_prob):
    key = json.dumps([name, sorted(image_ids), tile_size, stride,
                      skip_empty_prob, SPLIT_SEED], sort_keys=True)
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"tiles_{name}_{digest}.json")


def build_dataset(name, image_ids, images_by_id, anns_by_image_id,
                  skip_empty_prob, tile_size=TILE_SIZE, stride=STRIDE,
                  use_clahe=True, use_cache=True, verbose=True):
    """Build a dataset, reusing the cached tile index when one exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(name, image_ids, tile_size, stride, skip_empty_prob)

    samples = None
    if use_cache and os.path.exists(path):
        try:
            with open(path) as f:
                samples = json.load(f)
            if verbose:
                print(f"  {name}: {len(samples)} tiles (cached)")
        except (json.JSONDecodeError, OSError):
            samples = None               # corrupt cache -> rebuild

    ds = FilamentTileDataset(
        image_ids=image_ids, images_by_id=images_by_id,
        anns_by_image_id=anns_by_image_id, image_dir=TRAIN_IMAGES,
        tile_size=tile_size, stride=stride, skip_empty_prob=skip_empty_prob,
        use_clahe=use_clahe, samples=samples,
    )

    if samples is None:
        with open(path, "w") as f:
            json.dump([list(s) for s in ds.samples], f)
        if verbose:
            print(f"  {name}: {len(ds)} tiles (built and cached)")
    return ds


def setup(skip_empty_prob=0.5, use_cache=True, verbose=True):
    """Everything needed to train or evaluate. Seconds when the cache is warm."""
    coco, images_by_id, anns_by_image_id = load_coco()
    train_ids, val_ids, file_to_ids = make_splits(coco)
    if verbose:
        print(f"images {len(coco['images'])} ids / {len(file_to_ids)} files, "
              f"annotations {len(coco['annotations'])}")
        print(f"split: {len(train_ids)} train ids / {len(val_ids)} val ids")

    train_ds = build_dataset("train", train_ids, images_by_id, anns_by_image_id,
                             skip_empty_prob, use_cache=use_cache, verbose=verbose)
    val_ds = build_dataset("val", val_ids, images_by_id, anns_by_image_id,
                           0.0, use_cache=use_cache, verbose=verbose)

    return Context(coco=coco, images_by_id=images_by_id,
                   anns_by_image_id=anns_by_image_id, train_ids=train_ids,
                   val_ids=val_ids, file_to_ids=file_to_ids,
                   train_dataset=train_ds, val_dataset=val_ds)


def loaders(ctx, batch_size=4, num_workers=0):
    pin = torch.cuda.is_available()
    return (DataLoader(ctx.train_dataset, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin),
            DataLoader(ctx.val_dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin))


def get_device():
    assert torch.cuda.is_available(), (
        'CUDA not available - select the "ML Lab (shared, CUDA)" kernel')
    return torch.device("cuda")


def build_model(device=None, encoder="resnet34"):
    import segmentation_models_pytorch as smp
    device = device or get_device()
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                    in_channels=1, classes=1, activation=None).to(device)


TAG = "unet_resnet34_clahe_bce_tversky"


def load_model(which="best", device=None, encoder="resnet34", tag=TAG):
    """Load a trained checkpoint. Returns (model, checkpoint_dict)."""
    device = device or get_device()
    path = which if os.path.sep in str(which) else os.path.join(
        CHECKPOINT_DIR, f"{tag}_{which}.pt")
    ck = torch.load(path, map_location=device)
    model = build_model(device, encoder)
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck
                          else ck)
    model.eval()
    if isinstance(ck, dict) and "epoch" in ck:
        print(f"loaded {os.path.basename(path)}: epoch {ck['epoch']}, "
              f"best val IoU {ck.get('best_val_iou', float('nan')):.4f}")
    return model, ck


# --------------------------------------------------------------- evaluation
def tile_iou(prob, gt, thr=0.5):
    """Pixel IoU for one tile. None when prediction and GT are both empty."""
    pb = (prob > thr).float()
    inter = (pb * gt).sum()
    union = pb.sum() + gt.sum() - inter
    return None if union == 0 else (inter / union).item()


@torch.no_grad()
def evaluate(model, loader, device=None, thr=0.5, criterion=None, limit=None):
    """Mean IoU and complete-miss rate over tiles that contain filaments."""
    device = device or get_device()
    model.eval()
    ious, total_loss, seen = [], 0.0, 0
    for k, (imgs, masks) in enumerate(loader):
        if limit is not None and k >= limit:
            break
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(imgs)
        if criterion is not None:
            total_loss += criterion(logits, masks).item() * imgs.size(0)
            seen += imgs.size(0)
        probs = torch.sigmoid(logits.float())
        for b in range(imgs.size(0)):
            if masks[b].sum() == 0:
                continue                 # empty tile: IoU undefined
            v = tile_iou(probs[b], masks[b], thr)
            if v is not None:
                ious.append(v)
    ious = np.array(ious)
    return {
        "n_tiles": len(ious),
        "mean_iou": float(ious.mean()) if len(ious) else 0.0,
        "median_iou": float(np.median(ious)) if len(ious) else 0.0,
        "miss_rate": float((ious == 0).mean()) if len(ious) else 1.0,
        "iou_over_50": float((ious > 0.5).mean()) if len(ious) else 0.0,
        "loss": total_loss / seen if seen else None,
        "ious": ious,
    }


# ----------------------------------------------------------------- training
def cosine_schedule(epochs, eta_min=1e-6):
    """scheduler_fn for train(): cosine decay over `epochs`, stepped per epoch."""
    def make(optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min)
    return make


def train(ctx, model=None, epochs=5, batch_size=4, lr=1e-4, resume=None,
          tag=TAG, criterion=None, device=None, scheduler=None, scheduler_fn=None,
          new_run=False, max_batches=None):
    """Train, checkpointing on validation IoU.

    resume: None starts fresh, or "last" / "best" / a path to continue.
    new_run: with resume, load the weights and optimizer state but start the
        epoch counter, best-IoU tracker and history from zero. Use this with a
        NEW tag when changing the training recipe; otherwise the best-IoU
        carried in the old checkpoint blocks any new best from ever saving.
    scheduler_fn: callable taking the optimizer and returning an LR scheduler,
        e.g. cosine_schedule(10). It must be a callable, not a scheduler object:
        train() builds the optimizer internally, so a scheduler created
        outside would be attached to a different optimizer and silently do
        nothing (the LR just stays constant).
    max_batches: cap batches per epoch, for smoke tests.
    Saves <tag>_best.pt (highest val IoU), <tag>_last.pt (every epoch, so a
    crash does not lose the run) and <tag>_history.json.

    Selection is on IoU, not val_loss: with ~60 percent of val tiles empty, a
    model that predicts nothing scores a low loss while being useless.
    """
    import time

    from tqdm.auto import tqdm

    if scheduler is not None:
        raise ValueError(
            "Pass scheduler_fn=<callable(optimizer)>, not a scheduler object. "
            "train() creates its own optimizer, so a scheduler built outside is "
            "attached to a different optimizer and never changes the real LR. "
            "Example: scheduler_fn=cosine_schedule(epochs).")

    device = device or get_device()
    criterion = criterion or BCETverskyLoss()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_path = os.path.join(CHECKPOINT_DIR, f"{tag}_best.pt")
    last_path = os.path.join(CHECKPOINT_DIR, f"{tag}_last.pt")
    hist_path = os.path.join(CHECKPOINT_DIR, f"{tag}_history.json")

    if model is None:
        model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, best_iou, history = 0, -1.0, []

    if resume:
        path = resume if os.path.sep in str(resume) else os.path.join(
            CHECKPOINT_DIR, f"{tag}_{resume}.pt")
        ck = torch.load(path, map_location=device)
        if isinstance(ck, dict) and "model" in ck:
            model.load_state_dict(ck["model"])
            if ck.get("optimizer"):
                optimizer.load_state_dict(ck["optimizer"])
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            if not new_run:
                start_epoch = ck.get("epoch", 0)
                best_iou = ck.get("best_val_iou", -1.0)
                history = ck.get("history", [])
        else:
            model.load_state_dict(ck)
        for group in optimizer.param_groups:
            group["lr"] = lr             # honour the lr argument on resume
            group["initial_lr"] = lr     # the scheduler decays from this value
        print(f"resumed {os.path.basename(path)} at epoch {start_epoch}, "
              f"best val IoU {best_iou:.4f}, lr {lr:g}"
              + (" (new run: counters reset)" if new_run else ""))

    sched = scheduler_fn(optimizer) if scheduler_fn else None
    if sched is not None and resume and not new_run and ck.get("scheduler"):
        sched.load_state_dict(ck["scheduler"])

    train_loader, val_loader = loaders(ctx, batch_size=batch_size)
    total = start_epoch + epochs
    torch.cuda.empty_cache()

    for epoch in range(start_epoch, total):
        t0 = time.time()
        lr_used = optimizer.param_groups[0]["lr"]    # the rate this epoch trains at
        model.train()
        running, seen = 0.0, 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total} [train]")
        for k, (imgs, masks) in enumerate(bar):
            if max_batches is not None and k >= max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
            loss = criterion(logits, masks)          # loss stays fp32
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * imgs.size(0)
            seen += imgs.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running / max(seen, 1)

        res = evaluate(model, val_loader, device, criterion=criterion,
                       limit=max_batches)
        if sched is not None:
            sched.step()

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": res["loss"], "val_iou": res["mean_iou"],
            "miss_rate": res["miss_rate"], "lr": lr_used,
            "minutes": (time.time() - t0) / 60,
        })
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"Epoch {epoch+1}/{total} - train {train_loss:.4f} | "
              f"val {res['loss']:.4f} | val IoU {res['mean_iou']:.4f} | "
              f"misses {100*res['miss_rate']:.1f}% | lr {lr_used:.1e} | "
              f"{(time.time()-t0)/60:.1f} min")

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch + 1,
                 "scheduler": sched.state_dict() if sched is not None else None,
                 "best_val_iou": max(best_iou, res["mean_iou"]),
                 "history": history}
        torch.save(state, last_path)
        if res["mean_iou"] > best_iou:
            best_iou = res["mean_iou"]
            torch.save(state, best_path)
            print(f"  -> new best val IoU {best_iou:.4f}")

    print(f"Done. Best val IoU {best_iou:.4f} -> {best_path}")
    return model, history


# ------------------------------------------------- full-image tiled inference
def read_full_image(images_by_id, image_id, image_dir=None, use_clahe=True):
    """Full greyscale image for one image_id, CLAHE applied as in training."""
    image_dir = image_dir or TRAIN_IMAGES
    info = images_by_id[image_id]
    img = np.array(Image.open(os.path.join(image_dir, info["file_name"])).convert("L"))
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        img = clahe.apply(np.ascontiguousarray(img, dtype=np.uint8))
    return img


@torch.no_grad()
def predict_full_image(model, full_img, device=None, tile_size=TILE_SIZE,
                       stride=STRIDE, batch_size=8):
    """Probability map for a whole image, by running the model over tiles.

    Tiles overlap (stride < tile_size), so predictions are averaged where they
    do. Taking the max instead would bias towards whichever tile saw the
    filament nearest its centre.
    """
    device = device or get_device()
    model.eval()
    H, W = full_img.shape
    prob_sum = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    coords = generate_tile_coords(W, H, tile_size, stride)
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = np.stack([
            np.pad(full_img[y0:y0 + tile_size, x0:x0 + tile_size],
                   ((0, max(0, tile_size - min(tile_size, H - y0))),
                    (0, max(0, tile_size - min(tile_size, W - x0)))))
            for (x0, y0) in chunk
        ])
        t = torch.from_numpy(batch).float().unsqueeze(1).to(device) / 255.0
        with torch.amp.autocast("cuda"):
            logits = model(t)
        probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]

        for (x0, y0), p in zip(chunk, probs):
            h = min(tile_size, H - y0)
            w = min(tile_size, W - x0)
            prob_sum[y0:y0 + h, x0:x0 + w] += p[:h, :w]
            counts[y0:y0 + h, x0:x0 + w] += 1

    return prob_sum / np.maximum(counts, 1)


# ------------------------------------------------------ instances and PQ
def label_instances(binary_mask, min_area=20):
    """Connected components -> integer instance map (0 = background)."""
    import scipy.ndimage as ndi

    labelled, n = ndi.label(binary_mask)
    if n == 0:
        return labelled, 0
    keep, next_id = np.zeros_like(labelled), 0
    for i in range(1, n + 1):
        region = labelled == i
        if region.sum() >= min_area:
            next_id += 1
            keep[region] = next_id
    return keep, next_id


def gt_instances(images_by_id, anns_by_image_id, image_id, min_area=20):
    """One integer instance map from the COCO polygons of a single image_id."""
    info = images_by_id[image_id]
    H, W = info["height"], info["width"]
    inst = np.zeros((H, W), dtype=np.int32)
    next_id = 0
    for ann in anns_by_image_id.get(image_id, []):
        rings = seg_to_flat_rings(ann["segmentation"])
        if not rings:
            continue
        rles = maskUtils.frPyObjects(rings, H, W)
        m = maskUtils.decode(maskUtils.merge(rles))
        if m.sum() < min_area:
            continue
        next_id += 1
        inst[m > 0] = next_id
    return inst, next_id


def match_instances(pred_inst, gt_inst, iou_threshold=0.5):
    """Match predicted to ground-truth instances by IoU.

    Above 0.5 the match is unique (two predictions cannot both overlap one
    ground-truth instance by more than half), so greedy matching is exact.
    Returns (matched_ious, n_true_pos, n_false_pos, n_false_neg).
    """
    pred_ids = [i for i in np.unique(pred_inst) if i != 0]
    gt_ids = [i for i in np.unique(gt_inst) if i != 0]
    if not pred_ids and not gt_ids:
        return [], 0, 0, 0

    gt_areas = {g: int((gt_inst == g).sum()) for g in gt_ids}
    pred_areas = {p: int((pred_inst == p).sum()) for p in pred_ids}

    # intersections in one pass over the overlapping pixels
    overlap = {}
    both = (pred_inst > 0) & (gt_inst > 0)
    if both.any():
        pairs, counts = np.unique(
            np.stack([pred_inst[both], gt_inst[both]]), axis=1, return_counts=True)
        for (p, g), c in zip(pairs.T, counts):
            overlap[(int(p), int(g))] = int(c)

    matched_ious, used_gt, used_pred = [], set(), set()
    for (p, g), inter in sorted(overlap.items(), key=lambda kv: -kv[1]):
        if p in used_pred or g in used_gt:
            continue
        iou = inter / (pred_areas[p] + gt_areas[g] - inter)
        if iou > iou_threshold:
            matched_ious.append(iou)
            used_pred.add(p)
            used_gt.add(g)

    tp = len(matched_ious)
    return matched_ious, tp, len(pred_ids) - tp, len(gt_ids) - tp


def panoptic_quality(matched_ious, tp, fp, fn):
    """PQ = sum(IoU of matches) / (TP + FP/2 + FN/2), split into SQ and RQ."""
    denom = tp + 0.5 * fp + 0.5 * fn
    if denom == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0}
    sq = (sum(matched_ious) / tp) if tp else 0.0
    rq = tp / denom
    return {"pq": sq * rq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}


def evaluate_pq(model, ctx, image_ids=None, thr=0.5, min_area=20, device=None,
                limit=None, verbose=True, gt_min_area=GT_MIN_AREA):
    """Panoptic Quality over whole images - the competition metric.

    Pixel IoU is computed on tiles and ignores instance identity; PQ needs full
    images, because a filament split across a tile boundary is one instance.
    """
    device = device or get_device()
    image_ids = image_ids if image_ids is not None else ctx.val_ids
    if limit:
        image_ids = image_ids[:limit]

    all_ious, TP, FP, FN = [], 0, 0, 0
    per_image = []
    for k, image_id in enumerate(image_ids):
        full = read_full_image(ctx.images_by_id, image_id)
        prob = predict_full_image(model, full, device)
        pred_inst, _ = label_instances(prob > thr, min_area)
        # The ground-truth floor is FIXED. Filtering GT by the same min_area
        # would delete small true filaments from the scoring set and flatter
        # the score - the competition still scores them.
        gt_inst, _ = gt_instances(ctx.images_by_id, ctx.anns_by_image_id,
                                  image_id, gt_min_area)

        ious, tp, fp, fn = match_instances(pred_inst, gt_inst)
        all_ious += ious
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        per_image.append({"image_id": image_id, **panoptic_quality(ious, tp, fp, fn)})
        if verbose and k % 25 == 0:
            print(f"  {k}/{len(image_ids)} images", flush=True)

    result = panoptic_quality(all_ious, TP, FP, FN)
    result["per_image"] = per_image
    result["n_images"] = len(image_ids)
    return result


def evaluate_pq_grid(model, ctx, thresholds=(0.3, 0.5, 0.7),
                     min_areas=(20, 100, 400, 800, 1600),
                     image_ids=None, device=None, limit=None, verbose=True,
                     gt_min_area=GT_MIN_AREA):
    """PQ for every (threshold, min_area) pair in ONE pass over the images.

    Each probability map costs a forward pass over ~25 tiles, so computing it
    once and thresholding it many ways is far cheaper than re-running the model
    per setting.
    """
    device = device or get_device()
    image_ids = image_ids if image_ids is not None else ctx.val_ids
    if limit:
        image_ids = image_ids[:limit]

    acc = {(t, a): {"ious": [], "tp": 0, "fp": 0, "fn": 0}
           for t in thresholds for a in min_areas}

    for k, image_id in enumerate(image_ids):
        full = read_full_image(ctx.images_by_id, image_id)
        prob = predict_full_image(model, full, device)
        # GT is built once per image at the fixed floor - min_area tunes the
        # PREDICTIONS only. Filtering GT by it too would quietly drop small
        # true filaments from the scoring set.
        gt_inst, _ = gt_instances(ctx.images_by_id, ctx.anns_by_image_id,
                                  image_id, gt_min_area)
        for a in min_areas:
            for t in thresholds:
                pred_inst, _ = label_instances(prob > t, a)
                ious, tp, fp, fn = match_instances(pred_inst, gt_inst)
                s = acc[(t, a)]
                s["ious"] += ious
                s["tp"] += tp
                s["fp"] += fp
                s["fn"] += fn
        if verbose and k % 25 == 0:
            print(f"  {k}/{len(image_ids)} images", flush=True)

    rows = []
    for (t, a), s in acc.items():
        r = panoptic_quality(s["ious"], s["tp"], s["fp"], s["fn"])
        rows.append({"threshold": t, "min_area": a, **r})
    return sorted(rows, key=lambda r: -r["pq"])


def fragmentation_stats(model, ctx, thr=0.5, min_area=20, image_ids=None,
                        device=None, limit=None):
    """How often one true filament is predicted as several pieces.

    Pixel IoU is blind to this; PQ is not, because each extra piece becomes a
    false positive and can drop the match below the 0.5 IoU threshold.
    """
    device = device or get_device()
    image_ids = image_ids if image_ids is not None else ctx.val_ids
    if limit:
        image_ids = image_ids[:limit]

    n_pred, n_gt, pieces_per_gt = 0, 0, []
    for image_id in image_ids:
        full = read_full_image(ctx.images_by_id, image_id)
        prob = predict_full_image(model, full, device)
        pred_inst, np_ = label_instances(prob > thr, min_area)
        gt_inst, ng_ = gt_instances(ctx.images_by_id, ctx.anns_by_image_id,
                                    image_id, GT_MIN_AREA)
        n_pred += np_
        n_gt += ng_
        for g in range(1, ng_ + 1):
            region = gt_inst == g
            touching = np.unique(pred_inst[region])
            pieces_per_gt.append(int((touching != 0).sum()))

    pieces = np.array(pieces_per_gt)
    return {
        "n_pred_instances": n_pred,
        "n_gt_instances": n_gt,
        "pred_per_gt_ratio": n_pred / max(n_gt, 1),
        "gt_found_by_0_pieces": float((pieces == 0).mean()) if len(pieces) else 0.0,
        "gt_found_by_1_piece": float((pieces == 1).mean()) if len(pieces) else 0.0,
        "gt_split_into_2plus": float((pieces >= 2).mean()) if len(pieces) else 0.0,
        "mean_pieces_per_found_gt": float(pieces[pieces > 0].mean()) if (pieces > 0).any() else 0.0,
    }


# ------------------------------------------------------------ post-processing
def component_table(labelled, n, prob):
    """Per-component area, mean/max probability and elongation, vectorised.

    Index i holds component i (index 0 is background and meaningless).
    """
    H, W = labelled.shape
    flat = labelled.ravel()
    fg = np.nonzero(flat)[0]
    lab = flat[fg]
    ys, xs = np.divmod(fg, W)
    p = prob.ravel()[fg]

    area = np.bincount(lab, minlength=n + 1).astype(np.float64)
    safe = np.maximum(area, 1)
    mean_p = np.bincount(lab, weights=p, minlength=n + 1) / safe
    max_p = np.zeros(n + 1)
    np.maximum.at(max_p, lab, p)

    mx = np.bincount(lab, weights=xs, minlength=n + 1) / safe
    my = np.bincount(lab, weights=ys, minlength=n + 1) / safe
    cxx = np.bincount(lab, weights=xs.astype(np.float64) ** 2, minlength=n + 1) / safe - mx ** 2
    cyy = np.bincount(lab, weights=ys.astype(np.float64) ** 2, minlength=n + 1) / safe - my ** 2
    cxy = np.bincount(lab, weights=xs.astype(np.float64) * ys, minlength=n + 1) / safe - mx * my
    tr, det = cxx + cyy, cxx * cyy - cxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-6)
    return {"area": area, "mean_p": mean_p, "max_p": max_p,
            "elong": np.sqrt(l1 / l2)}


_DISKS = {}


def _disk(r):
    if r not in _DISKS:
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        _DISKS[r] = (x * x + y * y) <= r * r
    return _DISKS[r]


def segment_instances(prob, low=0.5, high=None, close=0, min_area=20,
                      min_mean=0.0, min_elong=0.0):
    """Probability map -> integer instance map, with false-positive filters.

    low:   pixels above this form candidate regions (sets each region's extent).
    high:  a region survives only if its peak probability reaches this. On the
           validation set, blobs that never get past ~0.99 matched a real
           filament only ~2% of the time yet made up 44% of all false positives.
    close: radius of a binary closing applied before labelling, to rejoin
           fragments of one filament.
    min_area / min_mean / min_elong: drop small, low-confidence or round blobs.
    Returns (instance_map, n_instances).
    """
    import scipy.ndimage as ndi

    mask = prob > low
    if close:
        mask = ndi.binary_closing(mask, structure=_disk(close))
    labelled, n = ndi.label(mask)
    if n == 0:
        return labelled.astype(np.int32), 0
    t = component_table(labelled, n, prob)
    keep = (t["area"] >= min_area) & (t["mean_p"] >= min_mean) & (t["elong"] >= min_elong)
    if high is not None:
        keep &= t["max_p"] >= high
    keep[0] = False
    lut = np.zeros(n + 1, dtype=np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1)
    return lut[labelled], int(keep.sum())


def pq_from_probs(ids, probs, ctx, gt_cache=None, **cfg):
    """PQ for a set of image_ids from precomputed probability maps (CPU only).

    probs: dict image_id -> probability map. gt_cache: optional dict
    image_id -> instance map, so repeated calls do not re-rasterise polygons.
    """
    ious, TP, FP, FN = [], 0, 0, 0
    for iid in ids:
        pred, _ = segment_instances(np.asarray(probs[iid], dtype=np.float32), **cfg)
        if gt_cache is not None and iid in gt_cache:
            gt = gt_cache[iid].astype(np.int32)
        else:
            gt, _ = gt_instances(ctx.images_by_id, ctx.anns_by_image_id, iid, GT_MIN_AREA)
        m, tp, fp, fn = match_instances(pred, gt)
        ious += m
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    return panoptic_quality(ious, TP, FP, FN)


def split_val_by_file(ctx, seed=0):
    """Two halves of validation, split by physical file so no photo is in both.

    Several annotators may have labelled the same photo; splitting by image_id
    would put one photo's pixels on both sides of the tune/test line.
    """
    files = sorted({ctx.images_by_id[i]["file_name"] for i in ctx.val_ids})
    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    half = set(files[: len(files) // 2])
    A = [i for i in ctx.val_ids if ctx.images_by_id[i]["file_name"] in half]
    B = [i for i in ctx.val_ids if ctx.images_by_id[i]["file_name"] not in half]
    return A, B


# -------------------------------------------- cached probability maps (CPU work)
def cache_val_probs(model, ctx, name, image_ids=None, device=None):
    """Save full-image probability maps (float16) so post-processing can be tuned
    on the CPU, without re-running the network. Existing files are skipped, so
    `model` may be None when everything is already cached."""
    image_ids = image_ids if image_ids is not None else ctx.val_ids
    out = os.path.join(CACHE_DIR, f"valprob_{name}")
    os.makedirs(out, exist_ok=True)
    todo = [i for i in image_ids if not os.path.exists(os.path.join(out, f"{i}.npy"))]
    if todo and model is None:
        raise ValueError(f"{len(todo)} maps missing from {out} and no model was given")
    for iid in todo:
        full = read_full_image(ctx.images_by_id, iid)
        prob = predict_full_image(model, full, device)
        np.save(os.path.join(out, f"{iid}.npy"), prob.astype(np.float16))
    print(f"valprob_{name}: {len(image_ids)} maps ({len(todo)} computed, "
          f"{len(image_ids) - len(todo)} cached)")
    return out


def load_val_probs(ctx, name, image_ids=None):
    """dict image_id -> float16 probability map, from cache_val_probs."""
    image_ids = image_ids if image_ids is not None else ctx.val_ids
    d = os.path.join(CACHE_DIR, f"valprob_{name}")
    return {i: np.load(os.path.join(d, f"{i}.npy")) for i in image_ids}


def matched_prediction_ids(pred_inst, gt_inst, iou_threshold=0.5):
    """Set of predicted-instance ids that match a ground-truth instance (IoU > t)."""
    both = (pred_inst > 0) & (gt_inst > 0)
    if not both.any():
        return set()
    pa = np.bincount(pred_inst.ravel())
    ga = np.bincount(gt_inst.ravel())
    pairs, counts = np.unique(np.stack([pred_inst[both], gt_inst[both]]),
                              axis=1, return_counts=True)
    used_p, used_g = set(), set()
    for (p, g), c in sorted(zip(pairs.T.tolist(), counts.tolist()), key=lambda t: -t[1]):
        if p in used_p or g in used_g:
            continue
        if c / (pa[p] + ga[g] - c) > iou_threshold:
            used_p.add(p)
            used_g.add(g)
    return used_p


def component_diagnostics(ctx, probs, image_ids=None, low=0.3, min_area=20, gt_cache=None):
    """One row per predicted blob: its properties and whether it is a true positive.

    Shows which properties separate real filaments from false positives, so
    filters are chosen from evidence rather than guessed.
    Returns a dict of equal-length arrays: area, mean_p, max_p, elong, tp.
    """
    import scipy.ndimage as ndi

    image_ids = image_ids if image_ids is not None else list(probs)
    cols = {k: [] for k in ("area", "mean_p", "max_p", "elong", "tp")}
    for iid in image_ids:
        pr = np.asarray(probs[iid], dtype=np.float32)
        lab, n = ndi.label(pr > low)
        if n == 0:
            continue
        t = component_table(lab, n, pr)
        keep = np.nonzero(t["area"][1:] >= min_area)[0] + 1
        lut = np.zeros(n + 1, np.int32)
        lut[keep] = np.arange(1, len(keep) + 1)
        if gt_cache is not None and iid in gt_cache:
            gt = gt_cache[iid].astype(np.int32)
        else:
            gt, _ = gt_instances(ctx.images_by_id, ctx.anns_by_image_id, iid, GT_MIN_AREA)
        used = matched_prediction_ids(lut[lab], gt)
        for new_id, old_id in enumerate(keep, start=1):
            cols["area"].append(t["area"][old_id])
            cols["mean_p"].append(t["mean_p"][old_id])
            cols["max_p"].append(t["max_p"][old_id])
            cols["elong"].append(t["elong"][old_id])
            cols["tp"].append(new_id in used)
    return {k: np.array(v) for k, v in cols.items()}


# ------------------------------------------------------------ test-set inference
TEST_IMAGES = os.path.join(DATA_ROOT, "test", "test_images")

# Post-processing used for submissions. Chosen on half A of validation (split by
# photo) with the epoch-12 model and checked, not re-tuned, on the cosine-run
# model; held-out half B scored 0.3606 PQ. Change it in ONE place, here.
SUBMISSION_CFG = dict(low=0.4, high=0.995, min_area=400)


def list_test_images():
    return sorted(os.listdir(TEST_IMAGES))


def read_image_path(path, use_clahe=True):
    """Greyscale image from a path, CLAHE applied exactly as in training."""
    img = np.array(Image.open(path).convert("L"))
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        img = clahe.apply(np.ascontiguousarray(img, dtype=np.uint8))
    return img


def checkpoint_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cfg_tag(cfg):
    return "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))


def test_results_dir(model_key, cfg=None):
    return os.path.join(CACHE_DIR, f"test_{model_key}_{_cfg_tag(cfg or SUBMISSION_CFG)}")


def predict_test_instances(model, model_key, cfg=None, names=None, device=None,
                           checkpoint_path=None, verbose=True):
    """Instance maps for the test images, stored in a format-neutral way.

    For each image <stem>: <stem>.npz holds a uint16 instance map (0 = background)
    and <stem>.json the per-instance area / mean / max probability / bbox, so any
    output format (RLE, polygons, masks, with or without scores) can be produced
    later without re-running the network. Existing results are skipped. A
    manifest.json records the checkpoint hash and settings, so a result can
    always be traced back to the exact model that made it.
    """
    cfg = dict(cfg or SUBMISSION_CFG)
    names = names if names is not None else list_test_images()
    out = test_results_dir(model_key, cfg)
    os.makedirs(out, exist_ok=True)
    device = device or get_device()

    todo = [n for n in names
            if not os.path.exists(os.path.join(out, os.path.splitext(n)[0] + ".npz"))]
    for k, name in enumerate(todo):
        stem = os.path.splitext(name)[0]
        full = read_image_path(os.path.join(TEST_IMAGES, name))
        prob = predict_full_image(model, full, device)
        inst, n = segment_instances(prob, **cfg)
        table = component_table(inst, n, prob)
        rows = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(inst == i)
            rows.append({"id": i, "area": int(table["area"][i]),
                         "mean_p": float(table["mean_p"][i]),
                         "max_p": float(table["max_p"][i]),
                         "bbox_xywh": [int(xs.min()), int(ys.min()),
                                       int(xs.max() - xs.min() + 1),
                                       int(ys.max() - ys.min() + 1)]})
        np.savez_compressed(os.path.join(out, stem + ".npz"), inst=inst.astype(np.uint16))
        with open(os.path.join(out, stem + ".json"), "w") as f:
            json.dump({"file_name": name, "height": int(inst.shape[0]),
                       "width": int(inst.shape[1]), "instances": rows}, f)
        if verbose and (k % 30 == 0 or k == len(todo) - 1):
            print(f"  {k + 1}/{len(todo)} images", flush=True)

    manifest = {"model_key": model_key, "cfg": cfg, "n_images": len(names),
                "clahe": [CLAHE_CLIP_LIMIT, list(CLAHE_TILE_GRID)],
                "tile": [TILE_SIZE, STRIDE]}
    if checkpoint_path:
        manifest["checkpoint"] = os.path.basename(checkpoint_path)
        manifest["checkpoint_sha256"] = checkpoint_sha256(checkpoint_path)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(f"{len(names)} test images in {out} ({len(todo)} computed, "
              f"{len(names) - len(todo)} cached)")
    return out


def load_test_instances(results_dir, name):
    """(instance map uint16, metadata dict) for one test image file name."""
    stem = os.path.splitext(name)[0]
    inst = np.load(os.path.join(results_dir, stem + ".npz"))["inst"]
    with open(os.path.join(results_dir, stem + ".json")) as f:
        meta = json.load(f)
    return inst, meta


# --------------------------------------------- converting instances to formats
def instance_rle(inst, i):
    """COCO run-length encoding of instance `i` (counts as a str, JSON-safe)."""
    m = np.asfortranarray((inst == i).astype(np.uint8))      # RLE is column-major
    rle = maskUtils.encode(m)
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def instance_polygons(inst, i, min_points=3):
    """Outer contour(s) of instance `i` as flat [x0, y0, x1, y1, ...] lists.

    Interior holes are dropped (RETR_EXTERNAL); filaments are thin, so this
    changes the area negligibly. Pixel-exact boundaries are kept (no
    simplification), so rasterising the polygon recovers the mask.
    """
    m = (inst == i).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        pts = c.reshape(-1, 2)
        if len(pts) >= min_points:
            out.append([float(v) for v in pts.ravel()])
    return out


# ------------------------------------------------------- Kaggle submission CSV
SUBMISSION_HEADER = "filament_id,segmentation_rle"


def write_submission_csv(results_dir, out_path, names=None):
    """Write the Kaggle submission: one CSV row per predicted filament.

    filament_id       = <image name without extension>_<n>, e.g. 20150125172714Mh_2
    segmentation_rle  = COCO RLE *counts* only (the 2048 x 2048 size is fixed),
                        unquoted, as the competition asks.

    Compressed-RLE counts use characters '0'..'o' only, so they can never contain
    the delimiter, a quote or a newline; this is checked anyway, because a stray
    one would silently corrupt a row.
    Images with no detections contribute no rows (adding a dummy mask would only
    turn a missed filament into a missed filament plus a false positive).
    """
    names = names if names is not None else list_test_images()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n_rows, empty = 0, []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write(SUBMISSION_HEADER + "\n")
        for nm in names:
            stem = os.path.splitext(nm)[0]
            inst, meta = load_test_instances(results_dir, nm)
            if not meta["instances"]:
                empty.append(nm)
            for row in meta["instances"]:
                counts = instance_rle(inst, row["id"])["counts"]
                if any(c in counts for c in ',"\r\n'):
                    raise ValueError(f"unsafe character in RLE for {stem}_{row['id']}")
                f.write(f"{stem}_{row['id']},{counts}\n")
                n_rows += 1
    return {"path": out_path, "rows": n_rows, "images": len(names),
            "images_without_rows": empty}


def read_submission_csv(path):
    """[(filament_id, counts), ...] parsed with no CSV library (counts have no commas)."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    assert lines[0] == SUBMISSION_HEADER, f"bad header: {lines[0]!r}"
    rows = []
    for ln in lines[1:]:
        if ln:
            fid, counts = ln.split(",", 1)
            rows.append((fid, counts))
    return rows


def verify_submission_csv(path, results_dir, names=None, size=(2048, 2048)):
    """Check a written CSV against the instance maps it was made from.

    Decodes every RLE with pycocotools (an independent path from the encoder
    helper) and requires the pixels to match exactly. Also checks that ids are
    unique, belong to a test image, and that no quote characters are present.
    """
    names = names if names is not None else list_test_images()
    stems = {os.path.splitext(n)[0]: n for n in names}
    rows = read_submission_csv(path)
    ids = [r[0] for r in rows]
    problems = []
    if len(set(ids)) != len(ids):
        problems.append("duplicate filament_id values")
    with open(path, encoding="utf-8") as f:
        text = f.read()                      # read once; a second read() would be empty
    if '"' in text or "'" in text:
        problems.append("quote characters present")

    by_image = {}
    for fid, counts in rows:
        stem, _, tail = fid.rpartition("_")
        if stem not in stems:
            problems.append(f"{fid}: image id not in the test set")
            continue
        if not tail.isdigit():
            problems.append(f"{fid}: tail is not a number")
        by_image.setdefault(stem, []).append((int(tail), counts))

    mismatched = 0
    for stem, items in by_image.items():
        inst, meta = load_test_instances(results_dir, stems[stem])
        if len(items) != len(meta["instances"]):
            problems.append(f"{stem}: {len(items)} rows but {len(meta['instances'])} instances")
            continue
        for tail, counts in items:
            decoded = maskUtils.decode({"size": list(size), "counts": counts.encode("ascii")})
            if not np.array_equal(decoded.astype(bool), inst == tail):
                mismatched += 1
                problems.append(f"{stem}_{tail}: decoded mask differs from the instance map")
    return {"rows": len(rows), "images_with_rows": len(by_image),
            "images_without_rows": sorted(set(stems) - set(by_image)),
            "mismatched_masks": mismatched, "problems": problems}


# --------------------------------------------------------------------- Dice
def dice_from_probs(ids, probs, ctx, gt_cache=None, **cfg):
    """Mean per-image Dice of the foreground (all instances merged), CPU only.

    Dice = 2|P and G| / (|P| + |G|). An image whose prediction and ground truth
    are both empty scores 1. The competition names torchmetrics' DiceScore; its
    exact averaging is not documented on the page, so treat this as close, not
    identical.
    """
    scores = []
    for iid in ids:
        pred, _ = segment_instances(np.asarray(probs[iid], dtype=np.float32), **cfg)
        p = pred > 0
        if gt_cache is not None and iid in gt_cache:
            g = gt_cache[iid] > 0
        else:
            g = gt_instances(ctx.images_by_id, ctx.anns_by_image_id, iid, GT_MIN_AREA)[0] > 0
        denom = int(p.sum()) + int(g.sum())
        scores.append(1.0 if denom == 0 else 2.0 * int((p & g).sum()) / denom)
    return float(np.mean(scores)), scores
