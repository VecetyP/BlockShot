import numpy as np

from backend.analysis import integrability as I
from tests.fixtures import fields


def test_sphere_normals_are_integrable():
    n, K, mask = fields.sphere()
    r = I.analyze(I.from_normals(n, K, mask))
    assert r.n_edges > 10_000
    assert r.residual_fraction < 0.01


def test_depth_derived_gradients_are_integrable_by_construction():
    rng = np.random.default_rng(0)
    depth = (2 + rng.random((64, 64))).astype(np.float32)
    r = I.analyze(I.from_depth(depth, np.ones((64, 64), bool)))
    assert r.residual_fraction < 1e-8


def test_penrose_field_is_not_integrable_and_residual_runs_around_the_loop():
    n, K, mask, lab = fields.penrose()
    r = I.analyze(I.from_normals(n, K, mask))
    assert r.residual_fraction > 0.1
    # Concentrated along the one cycle: every bar of the loop carries a real share of the residual,
    # rather than it piling up at a single bad junction.
    energy = np.array([(r.residual[lab == k] ** 2).sum() for k in (1, 2, 3)])
    assert (energy / energy.sum()).min() > 0.15
