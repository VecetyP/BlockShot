"""Fuse fitted dense depth at COLMAP's cameras by averaging (TSDF) instead of cutting, and score it.

Every frame's depth is integrated into a truncated signed-distance grid on the GPU: each voxel keeps
a weighted average of "how far in front of / behind the surface each frame saw it". The surface is
where that average crosses zero, so a 5% error in one frame is averaged out by the others instead of
deleting points (the free-space filter's cut).

Scored against COLMAP's well-triangulated points (seen from >= 10 degrees apart):
  accuracy     - distance from each such point to the nearest fused surface point, relative to its
                 distance from the camera (median);
  completeness - share of those points with fused surface within 2% of their distance;
  floaters     - share of fused points with fewer than 5 others within 2 voxels (isolated specks).

    uv run python tools/bakeoff/tsdf_fuse.py <sfm dir> <dense .npz> <out .npz>
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
from fuse_scene import fitted_depth, track_angles  # noqa: E402

import cv2  # noqa: E402

VOXEL = 1 / 30  # of the median anchor distance
TRUNC = float(__import__("os").environ.get("TSDF_TRUNC", 4))  # voxels: how far behind a surface a frame's evidence reaches
MIN_WEIGHT = float(__import__("os").environ.get("TSDF_MIN_WEIGHT", 2.0))  # a voxel needs about two frames' worth of evidence
CHUNK = 4_000_000
MAX_VOXELS = 60_000_000  # ~1.2 GB of grid on the GPU
REGION = 1.5  # fuse within this many camera-path widths of a camera


def tsdf(depth, rgb, K, c2w, lo, hi, vox):
    dev = "cuda"
    dims = np.ceil((hi - lo) / vox).astype(int)
    n = int(np.prod(dims))
    T = torch.zeros(n, device=dev)
    Wt = torch.zeros(n, device=dev)
    C = torch.zeros((n, 3), device=dev)
    tau = TRUNC * vox
    zmed = float(np.median(depth[depth > 0]))
    Kt = torch.tensor(K, device=dev, dtype=torch.float32)
    for i in range(len(depth)):
        if not (depth[i] > 0).any():
            continue
        w2c = torch.tensor(np.linalg.inv(c2w[i]), device=dev, dtype=torch.float32)
        D = torch.tensor(depth[i], device=dev)
        I = torch.tensor(rgb[i], device=dev, dtype=torch.float32)
        H, W = D.shape
        for a in range(0, n, CHUNK):
            idx = torch.arange(a, min(n, a + CHUNK), device=dev)
            x = idx % dims[0]
            y = (idx // dims[0]) % dims[1]
            z = idx // (dims[0] * dims[1])
            P = torch.stack([x, y, z], 1).float() * vox + torch.tensor(lo + vox / 2, device=dev, dtype=torch.float32)
            Q = P @ w2c[:3, :3].T + w2c[:3, 3]
            zc = Q[:, 2]
            u = (Kt[0, 0] * Q[:, 0] / zc + Kt[0, 2]).long()
            v = (Kt[1, 1] * Q[:, 1] / zc + Kt[1, 2]).long()
            ok = (zc > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            idx, u, v, zc = idx[ok], u[ok], v[ok], zc[ok]
            d = D[v, u]
            sdf = d - zc
            ok = (d > 0) & (sdf > -tau)
            idx, sdf, d, u, v = idx[ok], sdf[ok], d[ok], u[ok], v[ok]
            obs = torch.clamp(sdf / tau, max=1.0)
            w = torch.clamp(zmed / d, 0.25, 2.0)  # nearer views know more
            T[idx] = (T[idx] * Wt[idx] + obs * w) / (Wt[idx] + w)
            near = sdf.abs() < tau
            ci = idx[near]
            C[ci] = (C[ci] * Wt[ci, None] + I[v[near], u[near]] * w[near, None]) / (Wt[ci, None] + w[near, None])
            Wt[idx] += w
    return T.view(dims[2], dims[1], dims[0]).cpu().numpy(), Wt.view(dims[2], dims[1], dims[0]).cpu().numpy(), \
        C.view(dims[2], dims[1], dims[0], 3).cpu().numpy(), dims


def surface(T, Wt, C, lo, vox):
    """Zero crossings of the averaged distance between neighbouring voxels both seen enough."""
    pts, cols = [], []
    seen = Wt >= MIN_WEIGHT
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax], b[ax] = slice(0, -1), slice(1, None)
        ta, tb = T[tuple(a)], T[tuple(b)]
        m = seen[tuple(a)] & seen[tuple(b)] & (np.sign(ta) != np.sign(tb)) & (np.abs(ta - tb) < 1.5)
        zz, yy, xx = np.nonzero(m)
        f = ta[m] / (ta[m] - tb[m])
        g = np.stack([xx, yy, zz], 1).astype(np.float64)
        g[:, 2 - ax] += f  # axis order of the grid is (z, y, x)
        pts.append(lo + (g + 0.5) * vox)
        cols.append(np.where((np.abs(ta[m]) < np.abs(tb[m]))[:, None], C[tuple(a)][m], C[tuple(b)][m]))
    return np.concatenate(pts), np.concatenate(cols)


def score(points, anchors_w, anchors_dist, vox):
    tree = cKDTree(points)
    dist, _ = tree.query(anchors_w)
    rel = dist / anchors_dist
    cnt = tree.query_ball_point(points[np.random.default_rng(0).choice(len(points), min(len(points), 50_000), replace=False)],
                                2 * vox, return_length=True)
    return {"points": len(points), "accuracy (median)": f"{100 * np.median(rel):.2f}%",
            "completeness (<2%)": f"{100 * np.mean(rel < 0.02):.0f}%", "completeness (<5%)": f"{100 * np.mean(rel < 0.05):.0f}%", "floaters": f"{100 * np.mean(cnt < 6):.1f}%"}


def main():
    d, npz, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    names, K, c2w, (W, H), anchors, depth = fitted_depth(d, npz)
    rgb = np.stack([cv2.cvtColor(cv2.imread(str(d / "images" / n)), cv2.COLOR_BGR2RGB) for n in names])
    angles = track_angles(d / "text")
    aw, ad = [], []  # well-triangulated anchors in world coordinates, and their distance from the camera
    for i, a in enumerate(anchors):
        ok = (angles[i] >= 10) & (a[:, 2] > 0)
        x = (a[ok, 0] - K[0, 2]) / K[0, 0] * a[ok, 2]
        y = (a[ok, 1] - K[1, 2]) / K[1, 1] * a[ok, 2]
        P = np.stack([x, y, a[ok, 2]], 1)
        aw.append(P @ c2w[i, :3, :3].T + c2w[i, :3, 3])
        ad.append(a[ok, 2])
    aw, ad = np.concatenate(aw), np.concatenate(ad)
    zmed = float(np.median(ad))
    vox = VOXEL * zmed

    # Every frame's points, for bounds and for the comparisons.
    v, u = np.mgrid[0:H:2, 0:W:2]
    raw, raw_c = [], []
    for i in range(len(depth)):
        z = depth[i][v, u]
        m = z > 0
        P = np.stack([(u[m] - K[0, 2]) / K[0, 0] * z[m], (v[m] - K[1, 2]) / K[1, 1] * z[m], z[m]], 1)
        raw.append(P @ c2w[i, :3, :3].T + c2w[i, :3, 3])
        raw_c.append(rgb[i][v[m], u[m]])
    raw, raw_c = np.concatenate(raw), np.concatenate(raw_c)
    # Only the space around where the camera went: within REGION path-widths of any camera.
    cams = c2w[:, :3, 3]
    reach = REGION * np.linalg.norm(cams.max(0) - cams.min(0))
    near, _ = cKDTree(cams).query(raw, distance_upper_bound=reach)
    keep_raw = np.isfinite(near)
    lo, hi = np.percentile(raw[keep_raw], 1, 0), np.percentile(raw[keep_raw], 99, 0)
    vox = max(vox, float(np.prod(hi - lo) / MAX_VOXELS) ** (1 / 3))  # big scenes: coarser, to fit the GPU
    print("grid", np.ceil((hi - lo) / vox).astype(int), "voxel", round(vox, 4), flush=True)
    T, Wt, C, dims = tsdf(depth, rgb, K, c2w, lo, hi, vox)
    pts, cols = surface(T, Wt, C, lo, vox)

    # The same point budget for the others, on the same voxel grid (one point per voxel).
    def per_voxel(P, Cc):
        keys = np.floor((P - lo) / vox).astype(np.int64)
        inb = ((keys >= 0) & (keys < dims)).all(1)
        _, first = np.unique(keys[inb] @ np.array([1, dims[0], dims[0] * dims[1]]), return_index=True)
        return P[inb][first], Cc[inb][first]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend import video
    cut_mask = video.free_space_filter(depth, depth > 0, np.repeat(K[None], len(depth), 0).astype(np.float32), c2w)
    cut = []
    cut_c = []
    for i in range(len(depth)):
        m = cut_mask[i][v, u]
        z = depth[i][v, u][m]
        P = np.stack([(u[m] - K[0, 2]) / K[0, 0] * z, (v[m] - K[1, 2]) / K[1, 1] * z, z], 1)
        cut.append(P @ c2w[i, :3, :3].T + c2w[i, :3, 3])
        cut_c.append(rgb[i][v[m], u[m]])
    clouds = {"raw": per_voxel(raw, raw_c), "cut": per_voxel(np.concatenate(cut), np.concatenate(cut_c)), "tsdf": (pts, cols)}
    for name, (P, Cc) in clouds.items():
        print(f"{name:6s}", score(P, aw, ad, vox), flush=True)
    np.savez_compressed(out, cams=c2w.astype(np.float32), voxel=vox,
                        **{f"{k}_points": P.astype(np.float32) for k, (P, _) in clouds.items()},
                        **{f"{k}_rgb": np.clip(Cc, 0, 255).astype(np.uint8) for k, (_, Cc) in clouds.items()})
    print("saved", out)


if __name__ == "__main__":
    main()
