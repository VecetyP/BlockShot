"""Depth Anything 3 on a COLMAP model's images, conditioned on its cameras. Writes depth + K per image.

    tools/bakeoff/.venv/Scripts/python.exe tools/bakeoff/dense_da3.py <sfm dir> <hf model id> [window]
"""
import sys
from pathlib import Path

import numpy as np
import torch

import colmap_text
from depth_anything_3.api import DepthAnything3

d, mid = Path(sys.argv[1]), sys.argv[2]
win = int(sys.argv[3]) if len(sys.argv) > 3 else 24
names, K, c2w, (W, H), _ = colmap_text.load(d / "text")
model = DepthAnything3.from_pretrained(mid).to("cuda")
n = len(names)
depth, Ko = [None] * n, [None] * n
for a in range(0, max(1, n - 4), win - 4):
    idx = list(range(a, min(n, a + win)))
    w2c = np.stack([np.linalg.inv(c2w[i]) for i in idx]).astype(np.float32)
    p = model.inference([str(d / "images" / names[i]) for i in idx], extrinsics=w2c,
                        intrinsics=np.repeat(K[None], len(idx), 0).astype(np.float32))
    for j, i in enumerate(idx):
        if depth[i] is None:
            depth[i], Ko[i] = p.depth[j], p.intrinsics[j]
    torch.cuda.empty_cache()
    print("window", a, flush=True)
tag = mid.split("/")[-1]
np.savez_compressed(d / f"dense_da3_{tag}.npz", depth=np.stack(depth), K=np.stack(Ko))
print("saved", np.stack(depth).shape)
