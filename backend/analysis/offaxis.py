import numpy as np

YAWS = (-30, -15, -5, 5, 15, 30)
BACKGROUND = 64


def render(points: np.ndarray, rgb: np.ndarray, mask: np.ndarray, K: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Re-render the cloud through the same intrinsics after rotating it by yaw_deg about the vertical
    axis through its centroid (equivalently, orbiting the camera). 2x2 splats with a z-buffer."""
    h, w = mask.shape
    m = mask & (points[..., 2] > 0)
    p, c = points[m].astype(np.float64), rgb[m]
    centroid = p.mean(0)
    t = np.deg2rad(yaw_deg)
    R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    q = (p - centroid) @ R.T + centroid
    front = q[:, 2] > 1e-9
    q, c = q[front], c[front]
    u = np.floor(q[:, 0] / q[:, 2] * K[0, 0] + K[0, 2]).astype(np.int64)
    v = np.floor(q[:, 1] / q[:, 2] * K[1, 1] + K[1, 2]).astype(np.int64)
    offs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    uu = np.concatenate([u + du for du, _ in offs])
    vv = np.concatenate([v + dv for _, dv in offs])
    zz, cc = np.tile(q[:, 2], 4), np.tile(c, (4, 1))
    rank = np.repeat(np.arange(len(offs)), len(q))  # 0 = the point's own pixel
    ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    # Far to near, and at equal depth a point's own pixel last, so it beats neighbors' extra splats.
    # With repeated indices numpy keeps the last write.
    order = np.lexsort((-rank[ok], -zz[ok]))
    out = np.full((h, w, 3), BACKGROUND, np.uint8)
    out[vv[ok][order], uu[ok][order]] = cc[ok][order]
    return out
