import numpy as np


def reprojection_error(points: np.ndarray, K: np.ndarray, mask: np.ndarray) -> float:
    """Mean distance (px) between each point projected through K and the pixel center it came from.
    Near zero means points and intrinsics agree, i.e. the snapped view is exact."""
    h, w = mask.shape
    m = mask & (points[..., 2] > 0)
    if not m.any():
        return float("nan")
    p = points[m]
    u = p[:, 0] / p[:, 2] * K[0, 0] + K[0, 2]
    v = p[:, 1] / p[:, 2] * K[1, 1] + K[1, 2]
    gv, gu = np.nonzero(m)
    return float(np.hypot(u - (gu + 0.5), v - (gv + 0.5)).mean())
