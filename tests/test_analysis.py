import numpy as np

from backend.analysis import offaxis
from backend.analysis.discontinuity import discontinuity, top_tears
from backend.analysis.reproject import reprojection_error
from backend.models.base import default_intrinsics, unproject

H, W = 48, 64


def _scene():
    depth = np.where(np.arange(W) < 32, 2.0, 4.0)[None].repeat(H, 0).astype(np.float32)  # step at col 32
    K = default_intrinsics(H, W)
    return unproject(depth, K), K, np.ones((H, W), bool)


def test_reprojection_of_unprojected_depth_is_zero():
    pts, K, mask = _scene()
    assert reprojection_error(pts, K, mask) < 1e-4


def test_top_tears_sit_on_the_step():
    pts, _, mask = _scene()
    tears = top_tears(discontinuity(pts, mask), k=5, radius=4)
    assert tears and all(u in (31, 32) for u, _, _ in tears)


def test_offaxis_at_zero_yaw_reproduces_the_input():
    pts, K, mask = _scene()
    rgb = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)
    out, _ = offaxis.render(pts, rgb, mask, K, 0)
    same = (out == rgb).all(-1)
    same[:, 32] = True  # the near side's 2x2 splats correctly cover the far side of the step
    assert same.mean() > 0.99


def test_large_images_are_analyzed_on_a_consistent_subsampled_grid():
    from backend import store
    from backend.analysis import integrability as I
    from tests.fixtures import fields

    n, K, mask = fields.sphere(n=1100, f=1300.0)
    s, sl, K2 = store._analysis_grid(1100, 1100, K)
    assert s == 2  # 1100² is over the cap
    r = I.analyze(I.from_normals(n[sl], K2, mask[sl]))
    assert r.residual_fraction < 0.01
    assert store._upsample(r.residual, s, 1100, 1100).shape == (1100, 1100)
