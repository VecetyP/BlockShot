import gzip
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


def _scene():
    """A grey wall 4 units ahead with a red square in its upper-right quarter."""
    h, w = 48, 64
    depth = np.full((h, w), 4.0, np.float32)
    rgb = np.full((h, w, 3), 128, np.uint8)
    rgb[4:16, 44:60] = (200, 30, 30)
    K = default_intrinsics(h, w)
    return unproject(depth, K), rgb, np.ones((h, w), bool), K


def test_schematic_is_valid_sponge_v2():
    b = mc.build(*_scene(), size=32)
    name, s = read_nbt(gzip.decompress(mc.schematic(b)))
    assert name == "Schematic" and s["Version"] == 2 and s["DataVersion"] == mc.DATA_VERSION
    W, H, L = s["Width"], s["Height"], s["Length"]
    assert (W, H, L) == b.dims and len(s["BlockData"]) == W * H * L
    assert s["Palette"]["minecraft:air"] == 0 and s["PaletteMax"] == len(s["Palette"])
    assert [s["Metadata"][k] for k in ("WEOffsetX", "WEOffsetY", "WEOffsetZ")] == list(b.offset)
    by_index = {i: n for n, i in s["Palette"].items()}
    x, y, z = b.ijk[0]
    assert by_index[s["BlockData"][x + z * W + y * W * L]] == mc.BLOCK_NAMES[b.block[0]]
    assert sum(v != 0 for v in s["BlockData"]) == len(b.ijk)


def test_paste_puts_the_camera_at_the_eyes_and_image_right_to_the_west():
    pts, rgb, mask, K = _scene()
    b = mc.build(pts, rgb, mask, K, size=32)
    world = b.ijk + np.array(b.offset)  # where blocks land relative to the player's feet block
    eye = mc.EYE
    assert np.abs(b.camera + b.offset - eye).max() <= 0.5  # rounding to whole blocks
    # The wall is ahead (south, +z) of the eyes, and spans the view around the eye line.
    assert (world[:, 2] > eye[2]).all()
    assert world[:, 0].min() < eye[0] < world[:, 0].max() and world[:, 1].min() < eye[1] < world[:, 1].max()
    # The red square is in the image's upper right: up (+y) and, facing south, to the west (-x).
    red = np.array([mc.BLOCK_NAMES[i].startswith("minecraft:red") for i in b.block])
    assert red.any()
    assert world[red, 0].mean() < eye[0] and world[red, 1].mean() > eye[1]
    assert 0 < b.fov_deg < 90
