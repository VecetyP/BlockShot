"""Score dense depth against a COLMAP model: error at COLMAP's points, and agreement when fused at COLMAP's cameras.

    uv run python tools/bakeoff/score_dense.py <sfm dir> <dense .npz with depth (N,h,w) [and K (N,3,3)]>...
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import colmap_text  # noqa: E402
import common  # noqa: E402
from backend import video  # noqa: E402

HUD = 26  # px of the Roblox top bar at 540 px tall


def score(d: Path, npz: Path) -> dict:
    names, K, c2w, (W, H), anchors = colmap_text.load(d / "text")
    z = np.load(npz)
    D = z["depth"].astype(np.float64)
    h, w = D.shape[1:]
    Ks = z["K"] if "K" in z.files else np.repeat(K[None] * [[w / W], [h / H], [1]], len(D), 0)
    own, within, ratios = [], [], []
    for i, a in enumerate(anchors):
        u = np.clip((a[:, 0] * w / W).astype(int), 0, w - 1)
        v = np.clip((a[:, 1] * h / H).astype(int), 0, h - 1)
        dd = D[i][v, u]
        ok = (dd > 0) & (a[:, 2] > 0) & (a[:, 1] > HUD)
        if ok.sum() < 30:
            continue
        r = a[ok, 2] / dd[ok]
        ratios.append(np.median(r))
        rel = np.abs(r / np.median(r) - 1)
        own.append(np.median(rel))
        within.append(np.mean(rel < 0.05))
    ratios = np.array(ratios)
    # Fuse at COLMAP's cameras: each frame scaled by its own ratio, then the free-space check.
    g = np.median(ratios)
    depth = np.stack([cv2.resize(D[i], (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) for i in range(len(D))])
    sc = np.array([np.median(a[:, 2] / np.maximum(D[i][np.clip((a[:, 1] * h / H).astype(int), 0, h - 1), np.clip((a[:, 0] * w / W).astype(int), 0, w - 1)], 1e-9)) for i, a in enumerate(anchors)])
    depth = (depth * (sc / g)[:, None, None]).astype(np.float32)
    mask = depth > 0
    mask[:, : HUD // 2] = False
    Kh = np.repeat((K * [[0.5], [0.5], [1]])[None], len(D), 0).astype(np.float32)
    cw = c2w.copy()
    cw[:, :3, 3] /= g
    before = mask.sum()
    kept = video.free_space_filter(depth, mask, Kh, cw)
    rows = common.pair_stats(depth, kept, Kh.astype(float), cw)
    wt = rows[:, 2]
    nb = np.abs(rows[:, 0] - rows[:, 1]) <= 1
    return {"median error at points (own scale)": f"{100 * np.median(own):.1f}%",
            "points within 5%": f"{100 * np.median(within):.0f}%",
            "frame scale spread p90/p10": round(float(np.percentile(ratios, 90) / np.percentile(ratios, 10)), 2),
            "dropped by ghost filter": f"{100 * (1 - kept.sum() / before):.0f}%",
            "neighbour agreement": f"{100 * np.average(rows[nb, 3], weights=wt[nb]):.0f}%"}


if __name__ == "__main__":
    d = Path(sys.argv[1])
    for f in sys.argv[2:]:
        print(Path(f).stem.ljust(30), score(d, Path(f)), flush=True)
