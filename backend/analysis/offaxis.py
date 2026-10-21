import numpy as np
from scipy.ndimage import binary_closing, uniform_filter

YAWS = (-30, -15, -5, 5, 15, 30)
BACKGROUND = 64


def render(points: np.ndarray, rgb: np.ndarray, mask: np.ndarray, K: np.ndarray, yaw_deg: float, with_hit: bool = False):
    """Re-render the cloud through the same intrinsics after rotating it by yaw_deg about the vertical
    axis through its centroid (equivalently, orbiting the camera). 2x2 splats with a z-buffer.
    With `with_hit`, also returns the mask of pixels that received a point."""
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
    if not with_hit:
        return out
    hit = np.zeros((h, w), bool)
    hit[vv[ok], uu[ok]] = True
    return out, hit


def patch_pinholes(img: np.ndarray, hit: np.ndarray) -> np.ndarray:
    """Fill holes up to a few px wide (gaps between splats on a stretched surface) with the mean of the
    rendered pixels around them. Larger empty areas, i.e. real disocclusions, keep the background."""
    holes = binary_closing(hit, iterations=2, border_value=1) & ~hit
    if not holes.any():
        return img
    w = uniform_filter(hit.astype(np.float32), size=5)
    out = img.copy()
    for c in range(3):
        avg = uniform_filter(np.where(hit, img[..., c], 0).astype(np.float32), size=5) / np.maximum(w, 1e-6)
        out[..., c][holes] = np.clip(avg[holes], 0, 255).astype(np.uint8)
    return out
