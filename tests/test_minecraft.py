import gzip
import math
import struct

import numpy as np

from backend import minecraft as mc
from backend.models.base import default_intrinsics, unproject


def read_nbt(data: bytes):
    """Minimal NBT reader for the tag types the schematic writer uses."""
    pos = 0

    def take(n):
        nonlocal pos
        pos += n
        return data[pos - n:pos]

    def payload(tag):
        if tag == 2:
            return struct.unpack(">h", take(2))[0]
        if tag == 3:
            return struct.unpack(">i", take(4))[0]
        if tag == 7:
            return take(struct.unpack(">i", take(4))[0])
        if tag == 9:
            t, n = struct.unpack(">bi", take(5))
            return [payload(t) for _ in range(n)]
        if tag == 10:
            out = {}
            while (t := take(1)[0]) != 0:
                key = take(struct.unpack(">H", take(2))[0]).decode()  # name before payload
                out[key] = payload(t)
            return out
        if tag == 11:
            n = struct.unpack(">i", take(4))[0]
            return list(struct.unpack(f">{n}i", take(4 * n)))
        raise ValueError(tag)

    tag = take(1)[0]
    name = take(struct.unpack(">H", take(2))[0]).decode()
    return name, payload(tag)


def decode_varints(data: bytes) -> list[int]:
    out, v, shift = [], 0, 0
    for byte in data:
        v |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            out.append(v)
            v, shift = 0, 0
    return out


def placed(schem: bytes) -> dict[tuple[int, int, int], str]:
    """Where each block lands relative to the player's feet when pasted: idx + WEOffset."""
    _, s = read_nbt(gzip.decompress(schem))
    W, L = s["Width"], s["Length"]
    names = {i: n for n, i in s["Palette"].items()}
    off = [s["Metadata"][k] for k in ("WEOffsetX", "WEOffsetY", "WEOffsetZ")]
    out = {}
    for i, v in enumerate(decode_varints(s["BlockData"])):
        if v:
            x, z, y = i % W, (i // W) % L, i // (W * L)
            out[(x + off[0], y + off[1], z + off[2])] = names[v]
    return out


def _scene(h=48, w=64):
    """A grey wall 4 units ahead with a red square in its upper-right quarter."""
    depth = np.full((h, w), 4.0, np.float32)
    rgb = np.full((h, w, 3), 128, np.uint8)
    rgb[4:16, 44:60] = (200, 30, 30)
    K = default_intrinsics(h, w)
    return unproject(depth, K), rgb, np.ones((h, w), bool), K


def test_varints_round_trip_past_one_byte():
    v = np.array([0, 1, 127, 128, 129, 200, 300, 16383])
    assert decode_varints(mc.varints(v)) == v.tolist()


def test_schematic_is_valid_sponge_v2():
    b = mc.build_from_image(*_scene(), mode="fit", value=32)
    name, s = read_nbt(gzip.decompress(mc.whole(b)))
    assert name == "Schematic" and s["Version"] == 2 and s["DataVersion"] == mc.DATA_VERSION
    W, H, L = s["Width"], s["Height"], s["Length"]
    assert (W, H, L) == b.dims and len(decode_varints(s["BlockData"])) == W * H * L
    assert s["Palette"]["minecraft:air"] == 0 and s["PaletteMax"] == len(s["Palette"]) == len(mc.BLOCK_NAMES) + 1
    assert len(placed(mc.whole(b))) == len(b.ijk)


def test_paste_puts_the_camera_at_the_eyes_and_image_right_to_the_west():
    b = mc.build_from_image(*_scene(), mode="fit", value=32)
    world = b.ijk + np.array(b.offset)  # where blocks land relative to the player's feet block
    eye = mc.EYE
    assert np.abs(b.camera + b.offset - eye).max() <= 0.5  # rounding to whole blocks
    assert (world[:, 2] > eye[2]).all()  # ahead = south
    assert world[:, 0].min() < eye[0] < world[:, 0].max() and world[:, 1].min() < eye[1] < world[:, 1].max()
    red = np.array([mc.BLOCK_NAMES[i].startswith("minecraft:red") for i in b.block])
    assert red.any() and world[red, 0].mean() < eye[0] and world[red, 1].mean() > eye[1]  # up and west
    assert 0 < b.fov_deg < 90 and abs(b.pitch_deg) < 1e-9


def test_tiles_pasted_from_one_spot_rebuild_the_whole_build():
    b = mc.build_from_image(*_scene(96, 128), mode="fit", value=120)
    whole = placed(mc.whole(b))
    parts = mc.tiles(b, size=16)
    assert len(parts) > 4
    merged = {}
    for _, _, data in parts:
        merged.update(placed(data))
    assert merged == whole


def test_auto_hits_the_target_block_count_at_any_scene_size():
    for scale in (1.0, 50.0):  # a small room and a wide scene give the same block count
        pts, rgb, mask, K = _scene(96, 128)
        b = mc.build_from_image(pts * scale, rgb, mask, K, mode="auto", value=2)  # 2k blocks
        assert 1700 <= len(b.ijk) <= 2300


def test_surface_filling_makes_a_far_wall_solid_but_never_bridges_a_depth_step():
    h, w = 40, 60
    depth = np.where(np.arange(w) < 30, 50.0, 80.0)[None].repeat(h, 0).astype(np.float32)  # step at col 30
    K = default_intrinsics(h, w)
    pts, rgb, mask = unproject(depth, K), np.full((h, w, 3), 100, np.uint8), np.ones((h, w), bool)
    s = 0.25  # far smaller than the ~1.5-unit spacing between pixels at that distance
    b = mc.build_from_grids([(pts.astype(np.float64), rgb, mask, np.zeros(3))], mc.IMAGE_POSE, 60, "meters", s)
    z = b.lo[2] + (b.ijk[:, 2] + 0.5) * s  # block centres' distance ahead (Minecraft z = forward)
    near, far = np.abs(z - 50) < 1, np.abs(z - 80) < 1
    assert (near | far).all()  # nothing in between: the step is not bridged
    # Solid: the near wall's blocks cover its whole face, not just one block per pixel.
    face = mc.to_minecraft_axes(pts[:, :30].reshape(-1, 3))
    area = np.ptp(face[:, 0]) * np.ptp(face[:, 1])
    assert near.sum() > 0.8 * area / s**2


def test_budget_caps_block_count():
    b = mc.build_from_image(*_scene(96, 128), mode="meters", value=0.01, budget=500)
    assert b.budget_limited and len(b.ijk) <= 500


def test_tilted_camera_still_gives_level_floors_and_reports_pitch():
    # A floor (y = 1.5 in a y-down frame) seen by a camera pitched 20° down; "up" is the world's.
    g = np.stack(np.meshgrid(np.linspace(-3, 3, 80), np.linspace(2, 8, 80)), -1).reshape(-1, 2)
    floor = np.c_[g[:, 0], np.full(len(g), 1.5), g[:, 1]]
    t = math.radians(20)
    pose = mc.Pose(np.zeros(3), np.array([0, math.sin(t), math.cos(t)]), np.array([0.0, -1.0, 0.0]))
    b = mc.build(floor, np.full((len(floor), 3), 90, np.uint8), pose, 60, "meters", 0.25)
    assert len(np.unique(b.ijk[:, 1])) == 1  # one level layer
    assert math.isclose(b.pitch_deg, 20, abs_tol=1e-6)


def test_view_from_another_camera_matches_building_from_it():
    # A box room seen from two cameras: one facing +z, one turned 90° to face +x (both level).
    rng = np.random.default_rng(1)
    pts = rng.uniform(-4, 4, (40_000, 3))
    pts[np.arange(len(pts)), rng.integers(0, 3, len(pts))] = rng.choice([-4.0, 4.0], len(pts))  # on the walls
    rgb = np.full((len(pts), 3), 120, np.uint8)
    up = np.array([0.0, -1.0, 0.0])
    a = mc.Pose(np.zeros(3), np.array([0.0, 0, 1]), up)
    turned = mc.Pose(np.array([1.0, 0.5, -1.0]), np.array([1.0, 0, 0]), up)
    base = mc.build(pts, rgb, a, 60, "meters", 0.25)
    v = mc.view_from(base, turned, 50)
    # Facing +x in camera coordinates (y down) is facing west in Minecraft (east = up x south): yaw 90.
    assert math.isclose(v.yaw_deg, 90, abs_tol=1e-6) and math.isclose(v.pitch_deg, 0, abs_tol=1e-6) and v.fov_deg == 50
    # Pasting from the eye: block centre + offset - EYE is the block's position relative to the eye;
    # turning that into cloud coordinates must give back the block's own position.
    B, _ = mc._basis(base.pose)
    rel = (v.ijk + 0.5 + np.array(v.offset) - mc.EYE) * base.block_size  # Minecraft axes, from the eye
    world = rel @ B.T + turned.position
    centre = (base.lo + (v.ijk + 0.5) * base.block_size) @ B.T + base.pose.position
    assert np.abs(world - centre).max() < base.block_size  # within the offset's rounding
    # The space around the new eye is cleared.
    assert (np.linalg.norm(v.ijk + 0.5 - v.camera, axis=1) > mc.CLEAR_BLOCKS).all()


def test_find_up_finds_a_tilted_floor_among_walls():
    # Normals of a floor tilted 12 degrees (and a ceiling, facing down), plus many more walls and some
    # slanted junk; the prior is the untilted y axis.
    rng = np.random.default_rng(3)
    t = math.radians(12)
    true_up = np.array([0.0, math.cos(t), math.sin(t)])
    noisy = lambda v, k: v + rng.normal(0, 0.05, (k, 3))
    floor = noisy(true_up, 3000)
    ceiling = noisy(-true_up, 1000)
    side = np.cross(true_up, [1.0, 0, 0]); side /= np.linalg.norm(side)
    walls = np.vstack([noisy(side, 3000), noisy(np.cross(true_up, side), 3000)])  # two wall directions
    junk = rng.normal(size=(2000, 3))
    N = np.vstack([floor, ceiling, walls, junk])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    up, share = mc.find_up(N, np.array([0.0, 1.0, 0.0]))
    assert math.degrees(math.acos(min(1.0, up @ true_up))) < 1.0 and share > 0.3
    R = mc.rotation_between(up, np.array([0.0, -1.0, 0.0]))
    assert np.allclose(R @ up, [0, -1, 0]) and np.isclose(np.linalg.det(R), 1)


def test_grid_normals_of_a_plane():
    v, u = np.mgrid[0:40, 0:60].astype(float)
    P = np.stack([u, np.full_like(u, 2.0) + 0.1 * u, v], -1)  # plane y = 2 + 0.1 x
    N = mc.grid_normals(P, np.ones((40, 60), bool))
    n = np.array([-0.1, 1.0, 0.0]) / math.hypot(0.1, 1)
    assert len(N) > 50 and np.allclose(np.abs(N @ n), 1)


def test_seen_from_above_builds_a_top_down_picture_flat():
    # Ground 30 m below a camera looking straight down, with a few 3 m buildings on it.
    h, w = 96, 128
    K = np.array([[100.0, 0, w / 2], [0, 100.0, h / 2], [0, 0, 1]])
    depth = np.full((h, w), 30.0)
    depth[20:40, 30:50] = 27.0
    depth[60:80, 80:110] = 27.0
    pts = unproject(depth.astype(np.float32), K)
    rgb = np.full((h, w, 3), 100, np.uint8)
    mask = np.ones((h, w), bool)
    b = mc.build_from_image(pts, rgb, mask, K, "meters", 0.5, up=mc.image_up(pts, mask, above=True))
    assert b.dims[1] <= 12  # 3 m of buildings at 0.5 m blocks: a flat map, not a 40 m wall
    assert b.pitch_deg > 80  # looking down at it
    side = mc.build_from_image(pts, rgb, mask, K, "meters", 0.5)
    assert side.dims[1] > 30  # without the option it stands up like a wall
