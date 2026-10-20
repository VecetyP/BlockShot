import numpy as np


def discontinuity(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per pixel: largest 3D distance to a valid 4-neighbor, divided by the pixel's depth.

    Scale-invariant, so it compares across models. Pixels with no valid neighbor get 0.
    """
    h, w = mask.shape
    out = np.zeros((h, w), np.float32)
    for dy, dx in ((0, 1), (1, 0)):
        a = (slice(0, h - dy), slice(0, w - dx))
        b = (slice(dy, h), slice(dx, w))
        d = np.linalg.norm(points[a] - points[b], axis=-1)
        ok = mask[a] & mask[b]
        d = np.where(ok, d, 0)
        out[a] = np.maximum(out[a], d)
        out[b] = np.maximum(out[b], d)
    z = np.abs(points[..., 2])
    return np.where(mask & (z > 0), out / np.maximum(z, 1e-12), 0).astype(np.float32)


if __name__ == "__main__":
    # A plane at z=1 with a step to z=2 on the right half: the tear sits exactly on the step.
    h, w = 8, 8
    z = np.where(np.arange(w) < 4, 1.0, 2.0)[None].repeat(h, 0)
    pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], -1).astype(np.float32)
    d = discontinuity(pts, np.ones((h, w), bool))
    assert np.allclose(d[:, 3], 1.0) and np.allclose(d[:, 4], 0.5)
    assert np.allclose(d[:, :3], 0) and np.allclose(d[:, 5:], 0)
    print("ok")
