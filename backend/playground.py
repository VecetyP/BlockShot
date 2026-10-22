"""Ground truth for playground recordings, and scoring runs against it.

The web app's playground draws the scene itself, so every recording or snapshot can carry its exact
truth (truth.json, written by the browser):

    {"version": 1, "scene": "Courtyard", "width": 1280, "height": 720, "fov_deg": 60,
     "frames": [{"t": 0.0, "p": [x, y, z], "q": [x, y, z, w]}, ...],   # camera, three.js world, y up
     "objects": [{"shape": "box" | "pyramid" | "sphere" | "cylinder", "center": [x, y, z],
                  "size": [w, h, d], "rot": radians about y}, ...],
     "ground": {"y": 0, "half": 60}, "nuisances": {"hud": false, "arms": false, ...}}

Units are metres. Shapes: a box spans size around its centre; a pyramid has a w x d base at the bottom
and its apex at the top; a sphere's radius is w/2; a cylinder is vertical, radius w/2, height h.

Cameras are converted to OpenCV camera-to-world (x right, y down, z forward), the convention every
other part of BlockShot uses, while world points stay in the three.js frame (y up).
"""
import math

import numpy as np

from . import video

FLIP = np.diag([1.0, -1.0, -1.0])  # three.js camera axes (x right, y up, looks down -z) <-> OpenCV
SKY = np.inf


# --- poses -----------------------------------------------------------------------------------------

def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def c2w_cv(p, q) -> np.ndarray:
    """OpenCV camera-to-world (4x4) of a three.js camera at position p with quaternion q."""
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(np.asarray(q, float) / np.linalg.norm(q)) @ FLIP
    m[:3, 3] = p
    return m


def _slerp(q0, q1, a):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


def pose_at(frames: list[dict], t: float) -> np.ndarray:
    """OpenCV camera-to-world at time t (s), interpolated between logged frames (clamped at the ends)."""
    ts = np.array([f["t"] for f in frames])
    if t <= ts[0]:
        return c2w_cv(frames[0]["p"], frames[0]["q"])
    if t >= ts[-1]:
        return c2w_cv(frames[-1]["p"], frames[-1]["q"])
    k = int(np.searchsorted(ts, t)) - 1
    a = (t - ts[k]) / max(ts[k + 1] - ts[k], 1e-9)
    p = (1 - a) * np.asarray(frames[k]["p"]) + a * np.asarray(frames[k + 1]["p"])
    return c2w_cv(p, _slerp(frames[k]["q"], frames[k + 1]["q"], a))


def intrinsics(truth: dict, w: int, h: int) -> np.ndarray:
    """K of the playground camera (vertical FOV, square pixels) at w x h pixels."""
    f = (h / 2) / math.tan(math.radians(truth["fov_deg"]) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


# --- ray casting ---------------------------------------------------------------------------------

def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _convex(o, d, planes):
    """Ray (o + t d) against a convex solid {x: n.x <= c for all planes}; nearest t >= 0 or inf."""
    t0 = np.zeros(len(d))
    t1 = np.full(len(d), np.inf)
    for n, c in planes:
        dn, on = d @ n, o @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (c - on) / dn
        enter = dn < 0
        leave = dn > 0
        t0 = np.where(enter, np.maximum(t0, t), t0)
        t1 = np.where(leave, np.minimum(t1, t), t1)
        t1 = np.where((dn == 0) & (on > c), -np.inf, t1)  # parallel and outside
    hit = t0 <= t1
    return np.where(hit & (t1 >= 0), t0, np.inf)


def _box_planes(hw, hh, hd):
    return [(np.array([1.0, 0, 0]), hw), (np.array([-1.0, 0, 0]), hw), (np.array([0, 1.0, 0]), hh),
            (np.array([0, -1.0, 0]), hh), (np.array([0, 0, 1.0]), hd), (np.array([0, 0, -1.0]), hd)]


def _pyramid_planes(hw, hh, hd):
    """Square pyramid centred on the origin: base at y = -hh, apex at y = +hh."""
    h = 2 * hh
    planes = [(np.array([0, -1.0, 0]), hh)]
    for sx, sz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        half = hw if sx else hd
        # through the apex (0, hh, 0) and the base edge at x = half (or z): n = (h, half, 0) scaled
        n = np.array([sx * h, half, sz * h], float)
        planes.append((n, n @ np.array([0, hh, 0])))
    return planes


def _sphere(o, d, c, r):
    oc = o - c
    a = (d * d).sum(1)
    b = 2 * (oc * d).sum(1)
    cc = (oc * oc).sum(1) - r * r
    disc = b * b - 4 * a * cc
    sq = np.sqrt(np.maximum(disc, 0))
    t = (-b - sq) / (2 * a)
    t = np.where(t < 0, (-b + sq) / (2 * a), t)
    return np.where((disc >= 0) & (t >= 0), t, np.inf)


def _cylinder(o, d, c, r, hh):
    """Vertical cylinder centred at c, radius r, half height hh (side and caps)."""
    oc = o - c
    a = d[:, 0] ** 2 + d[:, 2] ** 2
    b = 2 * (oc[:, 0] * d[:, 0] + oc[:, 2] * d[:, 2])
    cc = oc[:, 0] ** 2 + oc[:, 2] ** 2 - r * r
    disc = b * b - 4 * a * cc
    sq = np.sqrt(np.maximum(disc, 0))
    best = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)):
            y = oc[:, 1] + t * d[:, 1]
            ok = (disc >= 0) & (a > 1e-12) & (t >= 0) & (np.abs(y) <= hh)
            best = np.where(ok & (t < best), t, best)
        for cap in (hh, -hh):
            t = (cap - oc[:, 1]) / d[:, 1]
            x, z = oc[:, 0] + t * d[:, 0], oc[:, 2] + t * d[:, 2]
            ok = (t >= 0) & (x * x + z * z <= r * r)
            best = np.where(ok & (t < best), t, best)
    return best


def raycast(o: np.ndarray, d: np.ndarray, truth: dict) -> np.ndarray:
    """Nearest hit parameter t along rays o + t d (world, y up) against the scene; inf for sky."""
    o = np.broadcast_to(o, d.shape)
    best = np.full(len(d), np.inf)
    g = truth.get("ground", {"y": 0.0, "half": 60.0})
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (g["y"] - o[:, 1]) / d[:, 1]
        x, z = o[:, 0] + t * d[:, 0], o[:, 2] + t * d[:, 2]
    best = np.where((t > 0) & (np.abs(x) <= g["half"]) & (np.abs(z) <= g["half"]), t, best)
    for ob in truth["objects"]:
        c = np.asarray(ob["center"], float)
        w, h, dd = (float(v) for v in ob["size"])
        if ob["shape"] == "sphere":
            t = _sphere(o, d, c, w / 2)
        elif ob["shape"] == "cylinder":
            t = _cylinder(o, d, c, w / 2, h / 2)
        else:
            R = _rot_y(float(ob.get("rot", 0.0)))
            lo, ld = (o - c) @ R, d @ R  # into the object's frame (R maps local -> world)
            planes = _box_planes(w / 2, h / 2, dd / 2) if ob["shape"] == "box" else _pyramid_planes(w / 2, h / 2, dd / 2)
            t = _convex(lo, ld, planes)
        best = np.minimum(best, t)
    return best


def true_depth(truth: dict, c2w: np.ndarray, K: np.ndarray, w: int, h: int, step: int = 1) -> np.ndarray:
    """z-depth the playground camera saw (inf = sky), at every `step`-th pixel of a w x h image
    (through those pixels' own centres, so it lines up with image[::step, ::step])."""
    v, u = np.mgrid[0:h:step, 0:w:step] + 0.5
    hs, ws = u.shape
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    # OpenCV camera -> world, then OpenCV world == three.js world (only camera axes were flipped)
    d = rays @ c2w[:3, :3].T
    t = raycast(c2w[:3, 3][None], d, truth)  # rays have z = 1, so t is the z-depth
    return t.reshape(hs, ws)


def surface_distance(P: np.ndarray, truth: dict) -> np.ndarray:
    """Approximate distance from world points to the nearest true surface (signed distances of the
    primitives; exact for ground, spheres, cylinders and boxes from outside, close for pyramids)."""
    g = truth.get("ground", {"y": 0.0})
    best = np.abs(P[:, 1] - g["y"])
    for ob in truth["objects"]:
        c = np.asarray(ob["center"], float)
        w, h, dd = (float(v) for v in ob["size"])
        q = P - c
        if ob["shape"] == "sphere":
            dist = np.abs(np.linalg.norm(q, axis=1) - w / 2)
        elif ob["shape"] == "cylinder":
            rad = np.hypot(q[:, 0], q[:, 2]) - w / 2
            hy = np.abs(q[:, 1]) - h / 2
            dist = np.abs(np.where((rad > 0) | (hy > 0), np.hypot(np.maximum(rad, 0), np.maximum(hy, 0)), np.maximum(rad, hy)))
        else:
            q = q @ _rot_y(float(ob.get("rot", 0.0)))
            if ob["shape"] == "box":
                e = np.abs(q) - np.array([w, h, dd]) / 2
                dist = np.abs(np.linalg.norm(np.maximum(e, 0), axis=1) + np.minimum(e.max(1), 0))
            else:
                dist = np.abs(np.max([q @ (n / np.linalg.norm(n)) - c0 / np.linalg.norm(n)
                                      for n, c0 in _pyramid_planes(w / 2, h / 2, dd / 2)], axis=0))
        best = np.minimum(best, dist)
    return best


# --- alignment and scores -------------------------------------------------------------------------

def _project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def align(pred_c2w: np.ndarray, true_c2w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) taking the predicted world onto the true one. Rotation from the camera
    orientations (works when the camera barely moves, e.g. turning in place), then scale and
    translation from the camera centres by least squares."""
    R = _project_so3(sum(tc[:3, :3] @ pc[:3, :3].T for pc, tc in zip(pred_c2w, true_c2w)))
    p = pred_c2w[:, :3, 3] @ R.T
    g = true_c2w[:, :3, 3]
    pc, gc = p - p.mean(0), g - g.mean(0)
    denom = (pc * pc).sum()
    s = float((pc * gc).sum() / denom) if denom > 1e-12 else 1.0
    if s <= 0:
        s = 1.0
    return s, R, g.mean(0) - s * p.mean(0)


def _rot_err_deg(pred_c2w, true_c2w, R):
    errs = []
    for pc, tc in zip(pred_c2w, true_c2w):
        M = tc[:3, :3].T @ (R @ pc[:3, :3])
        errs.append(math.degrees(math.acos(np.clip((np.trace(M) - 1) / 2, -1, 1))))
    return np.array(errs)


def _metric(key, label, value, unit, good, ok, note=""):
    """One score line; good/ok are upper bounds for the verdict colour (None: informational)."""
    verdict = None if good is None else ("good" if value <= good else "ok" if value <= ok else "poor")
    return {"key": key, "label": label, "value": round(float(value), 4), "unit": unit, "verdict": verdict, "note": note}


def score_scene(scene: dict, meta: dict, truth: dict) -> dict:
    """Scores for a video scene run (scene arrays from store.load_scene, meta with frame_times_s)."""
    times = meta.get("frame_times_s")
    if not times:
        raise ValueError("this scene has no frame times; recompute it so its frames can be matched to the recording")
    pred = scene["c2w"].astype(np.float64)
    n, h, w = scene["depth"].shape
    # The recording and the pose log can be a frame or two apart: find the offset that fits best.
    # Steady motion (a constant-speed circle or turn) looks the same at any offset, so only an offset
    # that fits clearly better than none (by 0.1°) is used and reported.
    fits = []
    for off in sorted(np.arange(-0.2, 0.2001, 0.02), key=abs):  # offset 0 first
        true = np.stack([pose_at(truth["frames"], t + off) for t in times])
        s, R, t = align(pred, true)
        fits.append((float(np.median(_rot_err_deg(pred, true, R))), off, true, s, R, t))
    best = min(fits, key=lambda f: f[0])
    measurable = fits[0][0] - best[0] >= 0.1
    rot_err, offset, true, s, R, t = best if measurable else fits[0]
    # Scale from the depths themselves (true / predicted, over every scored pixel), not from how far
    # the camera moved: turning on the spot gives no usable camera path, but plenty of depth.
    K = intrinsics(truth, w, h)
    step = 2
    pairs = []
    for i in range(n):
        td = true_depth(truth, true[i], K, w, h, step)
        pd = scene["depth"][i][::step, ::step].astype(np.float64)
        m = scene["mask"][i][::step, ::step] & np.isfinite(td) & (pd > 0)
        pairs.append((pd[m], td[m]))
    ratios = np.concatenate([t_ / p_ for p_, t_ in pairs])
    if len(ratios) < 100:
        raise ValueError("too few pixels to score")
    s = float(np.median(ratios))
    t = true[:, :3, 3].mean(0) - s * (pred[:, :3, 3] @ R.T).mean(0)
    cam = pred[:, :3, 3] @ R.T * s + t
    ate = float(np.sqrt(((cam - true[:, :3, 3]) ** 2).sum(1).mean()))
    path = float(np.linalg.norm(np.diff(true[:, :3, 3], axis=0), axis=1).sum())
    rel = np.concatenate([np.abs(p_ * s - t_) / t_ for p_, t_ in pairs])
    pts = []
    for i in range(n):
        P = video.world_points(scene["depth"][i], scene["K"][i].astype(np.float64), pred[i])[::8, ::8][scene["mask"][i][::8, ::8]]
        pts.append(P @ R.T * s + t)
    P = np.concatenate(pts)
    if len(P) > 200_000:
        P = P[np.random.default_rng(0).choice(len(P), 200_000, replace=False)]
    surf = surface_distance(P, truth)
    return {
        "kind": "scene",
        "frames": n,
        "metrics": [
            _metric("rotation", "Camera angle error", rot_err, "°", 2, 5, "median over frames"),
            _metric("ate", "Camera position error", ate, "m", max(0.1, 0.01 * path), max(0.3, 0.03 * path),
                    f"RMS over a {path:.1f} m path"),
            _metric("depth", "Depth error", float(np.median(rel)), "rel", 0.05, 0.15, "median |depth - true| / true"),
            _metric("surface", "Surface error", float(np.median(surf)), "m", 0.05, 0.2, "median distance of points to the true surfaces"),
            _metric("scale", "Scale vs metres", s, "×", None, None, "how much the scene had to be scaled to match"),
            _metric("offset", "Recording offset", offset * 1000, "ms", None, None,
                    "best match between video and pose log" if measurable else "not measurable: the motion was too steady, 0 used"),
        ],
    }


def score_image(pred_depth: np.ndarray, pred_mask: np.ndarray, metric: bool, truth: dict) -> dict:
    """Scores for an image run (a snapshot): depth after median scaling; metric models also get the
    scale they were off by."""
    h, w = pred_depth.shape
    c2w = c2w_cv(truth["frames"][0]["p"], truth["frames"][0]["q"])
    td = true_depth(truth, c2w, intrinsics(truth, w, h), w, h)
    m = pred_mask & np.isfinite(td) & (pred_depth > 0)
    if m.sum() < 100:
        raise ValueError("too few pixels to score")
    k = float(np.median(td[m] / pred_depth[m]))
    rel = np.abs(pred_depth[m] * k - td[m]) / td[m]
    metrics = [_metric("depth", "Depth error (shape)", float(np.median(rel)), "rel", 0.05, 0.15,
                       "median |depth - true| / true, after one overall scale")]
    if metric:
        metrics.append(_metric("metric_scale", "Metric scale off by", max(k, 1 / k), "×", 1.1, 1.5,
                               "1× means the model's metres were right"))
    return {"kind": "image", "metrics": metrics}
