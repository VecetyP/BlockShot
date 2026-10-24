"""Read a COLMAP text model (cameras.txt, images.txt, points3D.txt) into numpy, sorted by image name."""
from pathlib import Path

import numpy as np


def _rot(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load(d):
    """-> names, K (3, 3), c2w (N, 4, 4) OpenCV, size (W, H), anchors: per image (u, v, z_camera) of its 3D points."""
    d = Path(d)
    cam = [l for l in (d / "cameras.txt").read_text().splitlines() if not l.startswith("#")][0].split()
    W, H, fx, fy, cx, cy = int(cam[2]), int(cam[3]), *map(float, cam[4:8])
    pts = {int(l.split()[0]): np.array(list(map(float, l.split()[1:4])))
           for l in (d / "points3D.txt").read_text().splitlines() if l and not l.startswith("#")}
    lines = [l for l in (d / "images.txt").read_text().splitlines() if not l.startswith("#")]
    rows = []
    for a, b in zip(lines[0::2], lines[1::2]):
        p = a.split()
        R, t = _rot(*map(float, p[1:5])), np.array(list(map(float, p[5:8])))
        obs = np.array(b.split(), float).reshape(-1, 3) if b.strip() else np.zeros((0, 3))
        obs = obs[obs[:, 2] >= 0]
        P = np.stack([pts[int(i)] for i in obs[:, 2]]) if len(obs) else np.zeros((0, 3))
        z = (P @ R.T + t)[:, 2]
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ t
        rows.append((p[9], c2w, np.c_[obs[:, :2], z]))
    rows.sort(key=lambda r: r[0])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return [r[0] for r in rows], K, np.stack([r[1] for r in rows]), (W, H), [r[2] for r in rows]
