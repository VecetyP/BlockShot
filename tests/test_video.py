import math

import cv2
import numpy as np

from backend import video as V

rng = np.random.default_rng(0)


def _rot(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    i, j = [k for k in range(3) if k != axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def test_umeyama_recovers_a_similarity():
    src = rng.normal(size=(200, 3))
    R = _rot(1, 30) @ _rot(0, -10)
    dst = 2.5 * src @ R.T + [1, -2, 3]
    s, R2, t = V.umeyama(src, dst)
    assert math.isclose(s, 2.5, rel_tol=1e-9) and np.allclose(R2, R) and np.allclose(t, [1, -2, 3])


def test_c2w_inverts_the_extrinsics():
    R, t = _rot(2, 40), np.array([0.3, -1.0, 2.0])
    E = np.concatenate([R, t[:, None]], 1)[None]
    c2w = V.c2w_from_extrinsics(E)[0]
    x = rng.normal(size=3)
    assert np.allclose(c2w[:3, :3] @ (R @ x + t) + c2w[:3, 3], x)


def _wall_frames(centres, K, h, w, wall_z=6.0):
    """Depth maps of a wall at world z = wall_z seen from cameras at `centres` looking along +z."""
    v, u = np.mgrid[0:h, 0:w] + 0.5
    E, depth = [], []
    for c in centres:
        E.append(np.concatenate([np.eye(3), -np.asarray(c)[:, None]], 1))  # camera-from-world, no rotation
        depth.append(np.full((h, w), wall_z - c[2], np.float32))
    return np.array(E), np.array(depth)


def test_merge_chunks_puts_two_differently_scaled_chunks_into_one_frame():
    h, w = 40, 60
    K = np.array([[50.0, 0, w / 2], [0, 50.0, h / 2], [0, 0, 1]])
    centres = [np.array([0.3 * i, 0.0, 0.1 * i]) for i in range(9)]
    E, depth = _wall_frames(centres, K, h, w)
    conf = np.ones((9, h, w), np.float32) + rng.random((9, h, w)).astype(np.float32)
    a = {"extrinsics": E[:6], "intrinsics": np.repeat(K[None], 6, 0), "depth": depth[:6], "depth_conf": conf[:6]}
    # Chunk B (frames 3..8) lives in its own frame: x_B = (R x + t) / s, depths / s.
    s, R, t = 2.0, _rot(1, 25), np.array([1.0, 0.5, -2.0])
    EB = []
    for e in E[3:]:
        # x_cam = Rc x + tc with x = R^T (s x_B - t); in B's units: x_cam / s = Rc R^T x_B + (tc - Rc R^T t) / s
        Rc, tc = e[:, :3], e[:, 3]
        EB.append(np.concatenate([Rc @ R.T, (tc - Rc @ R.T @ t)[:, None] / s], 1))
    b = {"extrinsics": np.array(EB), "intrinsics": np.repeat(K[None], 6, 0), "depth": depth[3:] / s, "depth_conf": conf[3:]}
    out = V.merge_chunks([a, b], [0, 3], 9)
    true = V.c2w_from_extrinsics(E)
    assert np.allclose(out["c2w"], true, atol=1e-6)
    assert np.allclose(out["depth"], depth, atol=1e-5)


def test_hud_mask_finds_a_static_overlay_on_moving_footage():
    h, w = 120, 200
    base = (rng.random((h, w * 3)) * 255).astype(np.uint8)
    frames = []
    for i in range(12):
        f = np.repeat(base[:, i * 8:i * 8 + w, None], 3, 2).copy()  # the scene pans
        f[10:30, 10:70] = 255
        f[15:25, 15:65:4] = 0  # static "text"
        frames.append(f)
    hud = V.hud_mask(frames)
    assert hud is not None
    assert hud[10:30, 10:70].mean() > 0.5
    assert hud[60:, 100:].mean() < 0.02


def test_sample_frames_drops_duplicates_and_caps_count(tmp_path):
    path = str(tmp_path / "clip.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 90))
    tex = (rng.random((90, 1000, 3)) * 255).astype(np.uint8)
    for i in range(90):
        x = 0 if i < 30 else (i - 30) * 10  # 1 s of a still camera, then a pan
        out.write(np.ascontiguousarray(tex[:, x:x + 160]))
    out.release()
    frames, stats = V.sample_frames(path, V.Settings(fps=10, max_frames=100))
    assert stats["frames_read"] == 30 and stats["duplicates_dropped"] >= 8
    assert 18 <= len(frames) <= 22 and frames[0].shape == (90, 160, 3)
    frames, _ = V.sample_frames(path, V.Settings(fps=10, max_frames=5))
    assert len(frames) == 5
