import math

import numpy as np

from backend import playground as pg


def _scene(objects):
    return {"fov_deg": 60, "width": 64, "height": 36, "objects": objects, "ground": {"y": 0.0, "half": 60.0}}


def _ray(o, d, truth):
    return float(pg.raycast(np.array([o], float), np.array([d], float), truth)[0])


def test_raycast_hits_each_shape_at_the_right_distance():
    o, d = (0, 1, 5), (0, 0, -1)  # looking down -z from 5 m away, 1 m up
    assert math.isclose(_ray(o, d, _scene([{"shape": "box", "center": [0, 1, 0], "size": [2, 2, 2]}])), 4)
    assert math.isclose(_ray(o, d, _scene([{"shape": "sphere", "center": [0, 1, 0], "size": [2, 2, 2]}])), 4)
    assert math.isclose(_ray(o, d, _scene([{"shape": "cylinder", "center": [0, 1, 0], "size": [2, 2, 2]}])), 4)
    # A box turned 45 degrees: its corner points at the camera, sqrt(2) out from the centre.
    turned = _scene([{"shape": "box", "center": [0, 1, 0], "size": [2, 2, 2], "rot": math.pi / 4}])
    assert math.isclose(_ray(o, d, turned), 5 - math.sqrt(2), rel_tol=1e-9)
    # Straight down onto a 2 m pyramid (base 0..2 m): the apex is at y = 2.
    pyr = _scene([{"shape": "pyramid", "center": [0, 1, 0], "size": [2, 2, 2]}])
    assert math.isclose(_ray((0, 5, 0), (0, -1, 0), pyr), 3)
    # Halfway out from the apex the side is at height 1: x = 0.5 is above the slope at y = 1.
    assert math.isclose(_ray((0.5, 5, 0), (0, -1, 0), pyr), 4)
    # Ground and sky.
    assert math.isclose(_ray((0, 2, 0), (0, -1, 0), _scene([])), 2)
    assert _ray((0, 2, 0), (0, 1, 0), _scene([])) == np.inf


def test_three_camera_converts_to_opencv_axes_and_interpolates():
    c = pg.c2w_cv([1, 2, 3], [0, 0, 0, 1])  # identity three.js camera looks down -z, y up
    assert np.allclose(c[:3, 2], [0, 0, -1]) and np.allclose(c[:3, 1], [0, -1, 0]) and np.allclose(c[:3, 3], [1, 2, 3])
    yaw = lambda a: [0, math.sin(a / 2), 0, math.cos(a / 2)]
    frames = [{"t": 0.0, "p": [0, 0, 0], "q": yaw(0)}, {"t": 1.0, "p": [2, 0, 0], "q": yaw(math.pi / 2)}]
    mid = pg.pose_at(frames, 0.5)
    assert np.allclose(mid[:3, 3], [1, 0, 0])
    assert np.allclose(mid[:3, :3], pg.c2w_cv([0, 0, 0], yaw(math.pi / 4))[:3, :3])


def test_true_depth_of_a_wall_straight_ahead():
    truth = {**_scene([{"shape": "box", "center": [0, 1, -5], "size": [40, 40, 1]}]), "ground": {"y": -50.0, "half": 60.0}}
    c2w = pg.c2w_cv([0, 1, 0], [0, 0, 0, 1])
    K = pg.intrinsics(truth, 64, 36)
    d = pg.true_depth(truth, c2w, K, 64, 36)
    assert np.allclose(d, 4.5)  # the wall's near face is 4.5 m ahead, and z-depth is the same everywhere


def _perfect_run(truth, times, offset=0.0, scale=1.0):
    """A 'prediction' made from the truth itself (seen at times + offset), in another frame and scale."""
    w, h = 64, 36
    K = pg.intrinsics(truth, w, h)
    true = np.stack([pg.pose_at(truth["frames"], t + offset) for t in times])
    depth = np.stack([pg.true_depth(truth, c, K, w, h) for c in true])
    mask = np.isfinite(depth)
    depth = np.where(mask, depth, 0).astype(np.float32)
    # Put the prediction in another frame: rotate the world, shift it, and shrink it by `scale`.
    a = 0.7
    G = np.eye(4)
    G[:3, :3] = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    G[:3, 3] = [3, -1, 2]
    pred = np.stack([G @ c for c in true])
    pred[:, :3, 3] /= scale
    scene = {"c2w": pred, "depth": depth / scale, "mask": mask, "K": np.repeat(K[None], len(times), 0)}
    return scene, {"frame_times_s": list(times)}


def _walk(n=31, dt=0.1):
    """A wandering walk: the turning and sideways drift change speed, as a person's would (steady
    motion can't reveal a time offset, since a constant shift then looks the same at any offset)."""
    yaw = lambda a: [0, math.sin(a / 2), 0, math.cos(a / 2)]
    return [{"t": i * dt, "p": [0.8 * math.sin(i * dt * 2), 1.6, -0.1 * i], "q": yaw(0.6 * math.sin(i * dt * 2.5))} for i in range(n)]


def test_a_perfect_prediction_scores_zero_error():
    truth = {**_scene([{"shape": "box", "center": [0, 1, -6], "size": [6, 2, 1]},
                       {"shape": "sphere", "center": [2, 1, -4], "size": [1, 1, 1]}]), "frames": _walk()}
    scene, meta = _perfect_run(truth, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5], scale=2.0)
    m = {x["key"]: x["value"] for x in pg.score_scene(scene, meta, truth)["metrics"]}
    assert m["rotation"] < 0.01 and m["ate"] < 1e-3 and m["depth"] < 1e-3 and m["surface"] < 0.02
    assert math.isclose(m["scale"], 2.0, rel_tol=1e-3) and abs(m["offset"]) < 1e-6


def test_a_recording_offset_is_found():
    truth = {**_scene([{"shape": "box", "center": [0, 1, -6], "size": [6, 2, 1]}]), "frames": _walk()}
    scene, meta = _perfect_run(truth, [0.2, 0.7, 1.2, 1.7, 2.2], offset=0.1)
    m = {x["key"]: x["value"] for x in pg.score_scene(scene, meta, truth)["metrics"]}
    assert abs(m["offset"] - 100) < 1e-6 and m["rotation"] < 0.01


def test_image_scores_depth_shape_and_metric_scale():
    truth = {**_scene([{"shape": "box", "center": [0, 1, -6], "size": [6, 2, 1]}]),
             "frames": [{"t": 0, "p": [0, 1.6, 0], "q": [0, 0, 0, 1]}]}
    c2w = pg.c2w_cv(truth["frames"][0]["p"], truth["frames"][0]["q"])
    d = pg.true_depth(truth, c2w, pg.intrinsics(truth, 64, 36), 64, 36)
    mask = np.isfinite(d)
    pred = np.where(mask, d * 1.25, 0)  # right shape, metres 25% too long
    m = {x["key"]: x["value"] for x in pg.score_image(pred, mask, True, truth)["metrics"]}
    assert m["depth"] < 1e-6 and math.isclose(m["metric_scale"], 1.25, rel_tol=1e-6)
