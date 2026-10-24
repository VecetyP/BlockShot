"""MapAnything on a COLMAP model's images, given its cameras (and/or intrinsics). Writes depth + K per image.

    tools/bakeoff/.venv/Scripts/python.exe tools/bakeoff/dense_mapanything.py <sfm dir> <poses|intrinsics> [window]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import colmap_text
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs

d, mode = Path(sys.argv[1]), sys.argv[2]
win = int(sys.argv[3]) if len(sys.argv) > 3 else 24
names, K, c2w, (W, H), _ = colmap_text.load(d / "text")
model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
n = len(names)
depth, Ko = [None] * n, [None] * n
starts = list(range(0, max(1, n - 4), win - 4))
for a in starts:
    idx = list(range(a, min(n, a + win)))
    views = []
    for i in idx:
        img = cv2.cvtColor(cv2.imread(str(d / "images" / names[i])), cv2.COLOR_BGR2RGB)
        v = {"img": torch.from_numpy(img), "intrinsics": torch.from_numpy(K).float()}
        if mode == "poses":
            v["camera_poses"] = torch.from_numpy(c2w[i]).float()
            v["is_metric_scale"] = torch.tensor([False])
        views.append(v)
    preds = model.infer(preprocess_inputs(views), memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                        apply_mask=False, mask_edges=False)
    for i, p in zip(idx, preds):
        if depth[i] is None:
            depth[i] = p["depth_z"][0, ..., 0].float().cpu().numpy()
            Ko[i] = p["intrinsics"][0].float().cpu().numpy()
    torch.cuda.empty_cache()
    print("window", a, flush=True)
np.savez_compressed(d / f"dense_mapanything_{mode}.npz", depth=np.stack(depth), K=np.stack(Ko))
print("saved", np.stack(depth).shape)
