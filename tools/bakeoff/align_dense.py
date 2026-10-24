"""Fit each frame's dense depth to COLMAP's points, scored on held-out points.

Alignments: scale (one number), disparity (1/z = a/d + b, the usual for monocular depth), curve
(piecewise-linear in log depth, 6 knots, monotone). Writes <npz stem>_<alignment>.npz with depth in
COLMAP units, for score_dense.py.

    uv run python tools/bakeoff/align_dense.py <sfm dir> <dense .npz>...
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import colmap_text  # noqa: E402

HUD = 26


def fit(kind, d, z):
    """-> function mapping model depth to COLMAP depth, fitted on pairs (d, z) robustly."""
    keep = np.ones(len(d), bool)
    for _ in range(3):
        if kind == "scale":
            s = np.median(z[keep] / d[keep]); f = lambda x, s=s: s * x
        elif kind == "disparity":
            A = np.c_[1 / d[keep], np.ones(keep.sum())]
            a, b = np.linalg.lstsq(A, 1 / z[keep], rcond=None)[0]
            f = lambda x, a=a, b=b: 1 / np.maximum(a / np.maximum(x, 1e-6) + b, 1e-6)
        else:  # monotone piecewise-linear in log-log
            ld, lz = np.log(d[keep]), np.log(z[keep])
            knots = np.percentile(ld, np.linspace(0, 100, 6))
            knots = np.unique(knots)
            yk = []
            for j, k in enumerate(knots):
                lo = knots[max(j - 1, 0)]; hi = knots[min(j + 1, len(knots) - 1)]
                m = (ld >= lo) & (ld <= hi)
                yk.append(np.median(lz[m] - ld[m]) + k if m.any() else np.nan)
            yk = np.maximum.accumulate(np.nan_to_num(np.array(yk), nan=-np.inf))
            f = lambda x, knots=knots, yk=yk: np.exp(np.interp(np.log(np.maximum(x, 1e-6)), knots, yk,
                                                                left=None, right=None) + np.where(np.log(np.maximum(x, 1e-6)) > knots[-1], np.log(np.maximum(x, 1e-6)) - knots[-1], 0) + np.where(np.log(np.maximum(x, 1e-6)) < knots[0], np.log(np.maximum(x, 1e-6)) - knots[0], 0))
        r = np.abs(np.log(f(d) / z))
        keep = r < 3 * np.median(r[keep]) + 1e-3
    return f


def main():
    d = Path(sys.argv[1])
    names, K, c2w, (W, H), anchors = colmap_text.load(d / "text")
    rng = np.random.default_rng(0)
    for npz in map(Path, sys.argv[2:]):
        z = np.load(npz); D = z["depth"].astype(np.float64); h, w = D.shape[1:]
        for kind in ("scale", "disparity", "curve"):
            errs, out = [], np.zeros_like(D, np.float32)
            for i, a in enumerate(anchors):
                u = np.clip((a[:, 0] * w / W).astype(int), 0, w - 1); v = np.clip((a[:, 1] * h / H).astype(int), 0, h - 1)
                dd = D[i][v, u]
                ok = (dd > 0) & (a[:, 2] > 0) & (a[:, 1] > HUD)
                if ok.sum() < 40:
                    continue
                dd, zz = dd[ok], a[ok, 2]
                test = rng.random(len(dd)) < 0.3
                f = fit(kind, dd[~test], zz[~test])
                errs.append(np.abs(f(dd[test]) / zz[test] - 1))
                f = fit(kind, dd, zz)
                out[i] = np.where(D[i] > 0, f(D[i]), 0)
            e = np.concatenate(errs)
            print(f"{npz.stem:28s} {kind:9s} held-out points: median error {100 * np.median(e):.1f}%, within 5% {100 * np.mean(e < .05):.0f}%, within 10% {100 * np.mean(e < .1):.0f}%", flush=True)
            np.savez_compressed(npz.with_name(f"{npz.stem}_{kind}.npz"), depth=out)


if __name__ == "__main__":
    main()
