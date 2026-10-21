import numpy as np
from scipy.ndimage import binary_closing

from .fill import EDGE_JUMP, FAR_SLACK, Occlusion

YAWS = (-30, -15, -5, 5, 15, 30)
BACKGROUND = 64
GAP_COLOR = (255, 0, 200)  # hidden surface behind an occluding edge, within fill reach
HIDDEN_COLOR = (110, 60, 170)  # other hidden space: object back sides, far behind an edge
RAY_SAMPLES = 64


def _rotation(yaw_deg: float) -> np.ndarray:
    t = np.deg2rad(yaw_deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def centroid(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return points[mask & (points[..., 2] > 0)].astype(np.float64).mean(0)


def render(points, rgb, mask, K, yaw_deg, extra=None):
    """Re-render the cloud through the same intrinsics after rotating it by yaw_deg about the vertical
    axis through its centroid (equivalently, orbiting the camera). 2x2 splats with a z-buffer.
    `extra` = (xyz (N,3), rgb (N,3)) adds more points, e.g. a gap fill.
    Returns (image, hit) where hit marks pixels that received a point."""
    h, w = mask.shape
    m = mask & (points[..., 2] > 0)
    p, c = points[m].astype(np.float64), rgb[m]
    if extra is not None and len(extra[0]):
        p, c = np.concatenate([p, extra[0]]), np.concatenate([c, extra[1]])
    cen = centroid(points, mask)
    q = (p - cen) @ _rotation(yaw_deg).T + cen
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
    hit = np.zeros((h, w), bool)
    hit[vv[ok], uu[ok]] = True
    return out, hit


def closed(hit: np.ndarray) -> np.ndarray:
    """Hit mask with pinholes up to 2 px wide closed. Those come from a surface seen at a grazing
    angle: its points spread out when the view turns. That's undersampling, not hidden space."""
    return binary_closing(hit, iterations=1, border_value=1)


def gaps(depth, mask, K, occs: list[Occlusion], cen: np.ndarray, yaw_deg: float, hit: np.ndarray) -> list[np.ndarray]:
    """For each Occlusion: empty pixels of the yaw render whose line of sight passes through hidden
    space, i.e. behind one of its occluders as seen from the input camera but in front of the
    surface that occluder hides. Empty pixels whose ray only crosses space the input camera saw as
    empty are plain background, not gaps."""
    h, w = mask.shape
    vv, uu = np.nonzero(~closed(hit))
    found = [np.zeros(len(uu), bool) for _ in occs]
    x = (uu + 0.5 - K[0, 2]) / K[0, 0]
    y = (vv + 0.5 - K[1, 2]) / K[1, 1]
    z = depth[mask & (depth > 0)]
    R = _rotation(yaw_deg)
    for t in np.linspace(z.min() * 0.5, z.max() * 1.5, RAY_SAMPLES):  # depths along each ray
        q = np.stack([x * t, y * t, np.full_like(x, t)], -1)
        p = (q - cen) @ R + cen  # back into the input camera's frame
        pz = p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            us = np.floor(p[:, 0] / pz * K[0, 0] + K[0, 2])
            vs = np.floor(p[:, 1] / pz * K[1, 1] + K[1, 2])
        ok = (pz > 0) & (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
        i = np.flatnonzero(ok)
        r, s = vs[i].astype(np.int64), us[i].astype(np.int64)
        behind = pz[i] > depth[r, s] * (1 + EDGE_JUMP / 2)
        for occ, gap in zip(occs, found):
            gap[i[behind & occ.occluder[r, s] & (pz[i] < occ.far_depth[r, s] * (1 + FAR_SLACK))]] = True
    out = []
    for gap in found:
        m = np.zeros((h, w), bool)
        m[vv[gap], uu[gap]] = True
        out.append(m)
    return out
