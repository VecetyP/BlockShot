import numpy as np
import pytest

from backend.analysis import fill as F
from backend.analysis import offaxis
from backend.models.base import default_intrinsics, unproject

H, W = 96, 128


def _wall(box: bool):
    """A tilted plane (linear in inverse depth), optionally with a box floating in front of it."""
    v, u = np.mgrid[0:H, 0:W] + 0.5
    wall = 1 / (0.30 + 0.0010 * u + 0.0006 * v)
    depth = wall.copy()
    if box:
        depth[36:60, 52:76] = 1.5
    K = default_intrinsics(H, W)
    rgb = np.full((H, W, 3), 200, np.uint8)
    rgb[36:60, 52:76] = (200, 40, 40)
    return depth.astype(np.float32), wall.astype(np.float32), rgb, K, np.ones((H, W), bool)


@pytest.mark.parametrize("order", [2, 3])
def test_fill_continues_the_hidden_wall_exactly(order):
    depth, wall, rgb, _, mask = _wall(box=True)
    f = F.fill(depth, rgb, mask, reach=40, order=order)
    assert len(f.depth) == 24 * 24  # the whole box hides wall
    err = np.abs(f.depth / wall[f.rows, f.cols] - 1)
    assert err.max() < 1e-3
    assert (f.rgb[:, 1] > 150).all()  # wall colour, not the box's red


def test_offaxis_gaps_are_found_and_covered_by_the_fill():
    depth, _, rgb, K, mask = _wall(box=True)
    pts = unproject(depth, K)
    occ = F.occlusion(depth, mask, reach=1e9)
    f = F.fill(depth, rgb, mask, reach=40)
    extra = (unproject_px(f, K), f.rgb)
    cen = offaxis.centroid(pts, mask)
    _, hit = offaxis.render(pts, rgb, mask, K, 15)
    (gap,) = offaxis.gaps(depth, mask, K, [occ], cen, 15, hit)
    _, hit_filled = offaxis.render(pts, rgb, mask, K, 15, extra)
    assert gap.sum() > 50
    assert (hit_filled & gap).sum() / gap.sum() > 0.95


def test_a_plain_wall_has_no_gaps():
    depth, _, rgb, K, mask = _wall(box=False)
    pts = unproject(depth, K)
    occ = F.occlusion(depth, mask, reach=1e9)
    _, hit = offaxis.render(pts, rgb, mask, K, 15)
    assert not offaxis.gaps(depth, mask, K, [occ], offaxis.centroid(pts, mask), 15, hit)[0].any()


def unproject_px(f, K):
    x = (f.cols + 0.5 - K[0, 2]) / K[0, 0] * f.depth
    y = (f.rows + 0.5 - K[1, 2]) / K[1, 1] * f.depth
    return np.stack([x, y, f.depth], -1)
