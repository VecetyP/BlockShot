"""Shared bake-off pieces: the fixed frame windows, the frame-agreement score, arm masking and a
fused-cloud render. numpy / OpenCV / scipy only, so it runs in both the app's venv (VGGT baseline)
and the bake-off venv (contenders)."""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FRAME_SIZE = (640, 360)  # every model gets the same frames, decoded at this size

# (name, start s, end s, frames). 8 fps windows: neighbouring frames 1/8 s apart; 2 fps: 1/2 s apart.
WINDOWS = [
    ("calm 8fps 1-3s", 1, 3, 16),
    ("mid 8fps 10-12s", 10, 12, 16),
    ("frantic 8fps 29-31s", 29, 31, 16),
    ("running 2fps 0-8s", 0, 8, 16),
    ("respawn 2fps 17-25s", 17, 25, 16),
    ("end 2fps 28-35s", 28, 35, 16),
    ("across respawn 4fps 12-28s", 12, 28, 64),
]
CALM = ["calm 8fps 1-3s", "mid 8fps 10-12s"]
FAST = ["running 2fps 0-8s", "respawn 2fps 17-25s", "end 2fps 28-35s"]


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def load_window(clip: str, name: str, t0: float, t1: float, n: int) -> tuple[list[np.ndarray], list[Path]]:
    """RGB frames (and cached PNG paths, for models that take files) evenly spaced in [t0, t1]."""
    d = OUT / "frames" / slug(name)
    paths = [d / f"{i:03d}.png" for i in range(n)]
    if not all(p.exists() for p in paths):
        d.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(clip)
        fps = cap.get(cv2.CAP_PROP_FPS)
        for p, t in zip(paths, np.linspace(t0, t1, n)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, f = cap.read()
            if not ok:
                raise RuntimeError(f"could not read {clip} at {t:.2f} s")
            cv2.imwrite(str(p), cv2.resize(f, FRAME_SIZE, interpolation=cv2.INTER_AREA))
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths], paths


def world_points(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w] + 0.5
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    P = np.stack([x, y, depth], -1).astype(np.float64)
    return P @ c2w[:3, :3].T + c2w[:3, 3]


def pair_stats(depth, mask, K, c2w, step=6, min_overlap=300):
    """For every ordered pair of frames: frame i's points projected into frame j, compared with j's
    depth. rel = (z_proj - depth_j) / depth_j. Rows: i, j, n, agree (|rel| < 5%), front (rel < -5%:
    floats in space j saw through, a ghost), behind (rel > 5%: occluded, fine)."""
    n, h, w = depth.shape
    rows = []
    for i in range(n):
        P = world_points(depth[i], K[i], c2w[i])[::step, ::step][mask[i][::step, ::step]]
        for j in range(n):
            if i == j:
                continue
            w2c = np.linalg.inv(c2w[j])
            q = P @ w2c[:3, :3].T + w2c[:3, 3]
            z = q[:, 2]
            ok = z > 1e-6
            u = (K[j][0, 0] * q[ok, 0] / z[ok] + K[j][0, 2]).astype(int)
            v = (K[j][1, 1] * q[ok, 1] / z[ok] + K[j][1, 2]).astype(int)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u, v, zz = u[inside], v[inside], z[ok][inside]
            m = mask[j][v, u]
            if m.sum() < min_overlap:
                continue
            rel = (zz[m] - depth[j][v[m], u[m]]) / depth[j][v[m], u[m]]
            rows.append((i, j, len(rel), np.mean(np.abs(rel) < 0.05), np.mean(rel < -0.05), np.mean(rel > 0.05)))
    return np.array(rows)


def score_mask(pred: dict, extra: np.ndarray | None = None) -> np.ndarray:
    """Pixels scored: valid depth, above each frame's 20th confidence percentile (as in the app's
    original evaluation), minus `extra` (e.g. blanked arms)."""
    conf, depth = pred["conf"], pred["depth"]
    mask = np.stack([(depth[i] > 0) & np.isfinite(depth[i]) & (conf[i] > np.percentile(conf[i], 20)) for i in range(len(depth))])
    return mask if extra is None else mask & ~extra


def score(pred: dict, extra_mask: np.ndarray | None = None) -> dict:
    rows = pair_stats(pred["depth"], score_mask(pred, extra_mask), pred["K"], pred["c2w"])
    if not len(rows):
        return {"neighbours": 0.0, "all": 0.0, "ghosts": 0.0, "pairs": 0}
    w = rows[:, 2]
    nb = np.abs(rows[:, 0] - rows[:, 1]) <= 1
    fw = pred["c2w"][:, :3, 2]
    turn = np.degrees(np.arccos(np.clip((fw[:-1] * fw[1:]).sum(1), -1, 1)))
    return {
        "neighbours": float(np.average(rows[nb, 3], weights=w[nb])) if nb.any() else 0.0,
        "all": float(np.average(rows[:, 3], weights=w)),
        "ghosts": float(np.average(rows[:, 4], weights=w)),
        "pairs": int(len(rows)),
        "turn_median_deg": float(np.median(turn)),
        "turn_max_deg": float(turn.max()),
    }


def near_mask(depth: np.ndarray) -> np.ndarray:
    """The player's own arms / body: pixels far closer than the rest of the frame, grown a little."""
    return np.stack([binary_dilation(d < 0.25 * np.median(d[d > 0]), iterations=6) for d in depth])


def blank(frames: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    """Frames with `mask` painted grey (mask at model resolution is resized to the frames)."""
    out = []
    for f, m in zip(frames, mask):
        m = cv2.resize(m.astype(np.uint8), (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append(np.where(m[..., None], 128, f).astype(np.uint8))
    return out


def render(pred: dict, frame0: np.ndarray, W=480, H=270) -> np.ndarray:
    """Input frame 0 next to all frames' points fused and seen from frame 0's camera: a consistent
    reconstruction looks like the input; ghosting and doubling show up as smears."""
    mask = score_mask(pred)
    n = len(pred["depth"])
    P = np.concatenate([world_points(pred["depth"][i], pred["K"][i], pred["c2w"][i])[mask[i]] for i in range(n)])
    C = np.concatenate([pred["rgb"][i][mask[i]] for i in range(n)])
    w2c = np.linalg.inv(pred["c2w"][0])
    q = P @ w2c[:3, :3].T + w2c[:3, 3]
    ok = q[:, 2] > 1e-3
    q, C = q[ok], C[ok]
    K = pred["K"][0]
    sx, sy = W / pred["depth"].shape[2], H / pred["depth"].shape[1]
    u = (K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2]) * sx
    v = (K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2]) * sy
    img = np.zeros((H, W, 3), np.uint8)
    order = np.argsort(-q[:, 2])
    u, v = u[order].astype(int), v[order].astype(int)
    keep = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    img[v[keep], u[keep]] = C[order][keep]
    left = cv2.resize(frame0, (W, H), interpolation=cv2.INTER_AREA)
    return np.concatenate([left, img], 1)


def save_result(model: str, results: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{model}.json"
    p.write_text(json.dumps(results, indent=1))
    return p
