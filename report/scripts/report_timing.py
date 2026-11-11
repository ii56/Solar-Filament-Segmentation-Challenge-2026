"""Per-image inference time on the test set, stage by stage (RTX 2050)."""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "src"))
import filament as F

if __name__ == "__main__":
    model, _ = F.load_model("best", tag=F.TAG + "_cosine")
    names = F.list_test_images()[:30]
    nparams = sum(p.numel() for p in model.parameters())
    t = {"read+clahe": [], "network": [], "post": [], "rle": []}
    torch.cuda.reset_peak_memory_stats()
    for k, nm in enumerate(names):
        s = time.perf_counter()
        img = F.read_image_path(os.path.join(F.TEST_IMAGES, nm))
        s1 = time.perf_counter()
        prob = F.predict_full_image(model, img)
        torch.cuda.synchronize()
        s2 = time.perf_counter()
        inst, n = F.segment_instances(prob, **F.SUBMISSION_CFG)
        s3 = time.perf_counter()
        for i in range(1, n + 1):
            F.instance_rle(inst, i)
        s4 = time.perf_counter()
        if k >= 3:                                   # skip warm-up
            for key, dt in zip(t, (s1 - s, s2 - s1, s3 - s2, s4 - s3)):
                t[key].append(dt)
    for key, v in t.items():
        print(f"{key:11} {np.mean(v):.3f} s  (sd {np.std(v):.3f})")
    print(f"total      {sum(np.mean(v) for v in t.values()):.3f} s/image over {len(t['post'])} images")
    print(f"params {nparams/1e6:.2f} M, peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
