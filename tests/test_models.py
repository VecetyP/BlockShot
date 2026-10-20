"""One shape/dtype check per model wrapper, on CPU with a 64x64 synthetic image."""
import numpy as np
import pytest

from backend.models import registry

H = W = 64


@pytest.mark.parametrize("name", list(registry.MODELS))
def test_prediction_contract(name):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    p = registry.cls(name)("cpu").predict(image)

    assert p.points.shape == (H, W, 3) and p.points.dtype == np.float32
    assert p.depth.shape == (H, W) and p.depth.dtype == np.float32
    assert p.mask.shape == (H, W) and p.mask.dtype == bool
    assert p.intrinsics.shape == (3, 3) and p.intrinsics.dtype == np.float32
    if p.confidence is not None:
        assert p.confidence.shape == (H, W) and p.confidence.dtype == np.float32
        assert 0 <= p.confidence.min() and p.confidence.max() <= 1
    if p.normals is not None:
        assert p.normals.shape == (H, W, 3) and p.normals.dtype == np.float32
        n = np.linalg.norm(p.normals[p.mask], axis=-1)
        assert np.allclose(n, 1, atol=1e-3)
    assert p.scale_kind in ("metric", "relative", "affine-invariant")
    assert {"device", "inference_s", "input_resolution"} <= p.meta.keys()
    # points must agree with depth on the z axis
    assert np.allclose(p.points[..., 2][p.mask], p.depth[p.mask], rtol=1e-5)
