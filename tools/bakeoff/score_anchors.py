"""Dense depth vs COLMAP's well-triangulated points, on held-out points, per alignment.

    uv run python tools/bakeoff/score_anchors.py <sfm dir> <dense .npz>...
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import align_dense  # noqa: E402
import colmap_text  # noqa: E402
from fuse_scene import HUD, track_angles  # noqa: E402

d = Path(sys.argv[1])
names, K, c2w, (W, H), anchors = colmap_text.load(d / "text")
angles = track_angles(d / "text")
for npz in map(Path, sys.argv[2:]):
    D = np.load(npz)["depth"].astype(float)
    h, w = D.shape[1:]
    cells = []
    for amin in (5, 10):
        for kind in ("scale", "curve"):
            rng = np.random.default_rng(0)
            errs = []
            for i, a in enumerate(anchors):
                u = np.clip((a[:, 0] * w / W).astype(int), 0, w - 1)
                v = np.clip((a[:, 1] * h / H).astype(int), 0, h - 1)
                dd = D[i][v, u]
                ok = (dd > 0) & (a[:, 2] > 0) & (a[:, 1] > HUD) & (angles[i] >= amin)
                if ok.sum() < 40:
                    continue
                dd, zz = dd[ok], a[ok, 2]
                test = rng.random(len(dd)) < 0.3
                f = align_dense.fit(kind, dd[~test], zz[~test])
                errs.append(np.abs(f(dd[test]) / zz[test] - 1))
            e = np.concatenate(errs)
            cells.append(f"angle>={amin} {kind}: {100 * np.median(e):4.1f}% ({100 * np.mean(e < .05):.0f}% within 5%)")
    print(f"{npz.stem:32s} " + " | ".join(cells), flush=True)
