"""MoGe depth for every image of a COLMAP model, told the camera's real FOV. Writes <sfm dir>/moge_full.npz.

    uv run python tools/bakeoff/dense_moge.py <sfm dir>
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import colmap_text  # noqa: E402
from backend.models import registry  # noqa: E402

d = Path(sys.argv[1])
names, K, _, (W, H), _ = colmap_text.load(d / "text")
fov_x = math.degrees(2 * math.atan(W / 2 / K[0, 0]))
moge = registry.get("moge")
depth, mask = [], []
for i, n in enumerate(names):
    rgb = cv2.cvtColor(cv2.imread(str(d / "images" / n)), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).to(moge.device).permute(2, 0, 1).float().div(255)
    with torch.inference_mode():
        out = moge.model.infer(x, fov_x=fov_x)
    dd = np.nan_to_num(out["depth"].float().cpu().numpy(), nan=0, posinf=0)
    depth.append(dd.astype(np.float32))
    mask.append(out["mask"].cpu().numpy().astype(bool) & (dd > 0))
    if i % 20 == 0:
        print(f"{i}/{len(names)}", flush=True)
np.savez_compressed(d / "moge_full.npz", depth=np.stack(depth), mask=np.stack(mask))
print("saved", d / "moge_full.npz")
