import numpy as np

from backend import world


def city_tops(rng, x0, x1, z0, z1):
    """Points on the roofs and streets of a random box city, inside [x0, x1) x [z0, z1) blocks."""
    H = np.zeros((200, 200))
    for _ in range(60):
        x, z = rng.integers(0, 190, 2)
        w, l = rng.integers(4, 20, 2)
        H[x:x + w, z:z + l] = rng.integers(5, 40)
    x, z = np.meshgrid(np.arange(x0, x1), np.arange(z0, z1), indexing="ij")
    x, z = x.ravel() + 0.5, z.ravel() + 0.5
    return np.stack([x, H[x.astype(int), z.astype(int)], z], 1)


def test_propose_recovers_turn_scale_and_offset():
    rng = np.random.default_rng(0)
    pts = city_tops(rng, 0, 200, 0, 200)
    old = pts[(pts[:, 0] < 120)]  # the world so far
    new = pts[(pts[:, 0] >= 70)]  # the next recording overlaps it by 50 blocks
    # As the new scene saw it: its own origin and turn, in units of half a block, 7 higher.
    truth = {"turns": 1, "scale": 2.0, "offset": [-90.0, -7.0, 40.0]}
    seen = (new - truth["offset"]) @ world.rot(truth["turns"]) / truth["scale"]
    assert np.allclose(world.place(seen, truth), new)
    got = world.propose(old, seen, base=1.9)  # eye height guessed 5% off
    assert got["turns"] == truth["turns"]
    assert abs(got["scale"] / 2.0 - 1) < 0.06
    placed = world.place(seen, got)
    assert np.abs(placed - new).max() < 3


def test_follow_places_a_piece_from_its_shared_frames(monkeypatch):
    # One camera path in world blocks; piece B saw its last 4 frames turned, at the same scale.
    rng = np.random.default_rng(1)
    path = np.cumsum(rng.normal(0, 1, (12, 3)), 0)
    heading = rng.normal(0, 1, (12, 3))
    prev = {"scene": "A", "turns": 0, "scale": 1.0, "offset": [0.0, 0.0, 0.0]}
    truth = {"turns": 3, "scale": 1.0, "offset": [5.0, 1.0, -3.0]}
    own = lambda p: (p - truth["offset"]) @ world.rot(truth["turns"]) / truth["scale"]  # world -> B's own units
    cams = {"A": (path[:8], heading[:8]), "B": (own(path[4:]), heading[4:] @ world.rot(truth["turns"]))}
    monkeypatch.setattr(world, "scene_cameras", lambda rid: cams[rid])
    got = world.follow(prev, "B", 4)
    assert got["turns"] == truth["turns"]
    assert got["scale"] == 1.0
    assert np.allclose(got["offset"], truth["offset"])
