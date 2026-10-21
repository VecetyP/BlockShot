"""Point cloud -> Minecraft build that looks like the input image from one spot.

The cloud is snapped to a block grid in Minecraft's axes with the input camera facing south, each
block takes the palette block closest in colour (CIELAB), and the result is written as a Sponge v2
schematic (.schem) with WorldEdit's paste offset set so that `//paste` puts the input camera at
the player's eyes. Stand there, face south, set the FOV to `fov_deg`, and the build lines up with
the image.
"""
import gzip
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_VERSION = 2586  # Minecraft 1.16.5: the palette's blocks all exist there; newer versions upgrade it
EYE = np.array([0.5, 1.62, 0.5])  # player eye position inside the block the player stands in
OUTLIER_PCT = 0.5  # % of points ignored at each end of each axis when sizing the build

_palette = json.loads((Path(__file__).parent / "minecraft_palette.json").read_text())["blocks"]
BLOCK_NAMES = list(_palette)
BLOCK_RGB = np.array(list(_palette.values()), np.uint8)
assert len(BLOCK_NAMES) < 127, "BlockData is written as one byte per block"


def _linear(srgb: np.ndarray) -> np.ndarray:
    c = srgb / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(linear_rgb: np.ndarray) -> np.ndarray:
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = linear_rgb @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], -1)


_BLOCK_LAB = _lab(_linear(BLOCK_RGB.astype(np.float64)))


def nearest_blocks(linear_rgb: np.ndarray) -> np.ndarray:
    """Palette index of the closest block in CIELAB for each colour (linear light, 0..1)."""
    lab = _lab(linear_rgb)
    out = np.empty(len(lab), np.int64)
    for i in range(0, len(lab), 65536):  # chunks keep the distance matrix small
        d = ((lab[i:i + 65536, None, :] - _BLOCK_LAB[None]) ** 2).sum(-1)
        out[i:i + 65536] = d.argmin(1)
    return out


@dataclass
class Build:
    ijk: np.ndarray  # (N, 3) block coordinates (x east, y up, z south) from the build's min corner
    block: np.ndarray  # (N,) palette index
    dims: tuple[int, int, int]  # (width x, height y, length z) in blocks
    block_size: float  # size of one block in the cloud's units
    lo: np.ndarray  # (3,) min corner in Minecraft-axis cloud units
    camera: np.ndarray  # (3,) input camera in block coordinates (continuous)
    offset: tuple[int, int, int]  # block coordinate + offset = position relative to the player's feet block
    fov_deg: float  # vertical field of view of the input camera


def to_minecraft_axes(p: np.ndarray) -> np.ndarray:
    """OpenCV camera frame (x right, y down, z forward) -> Minecraft axes with the camera facing
    south (+z). Facing south, east (+x) is on your left, so image-right is -x."""
    return p * np.array([-1.0, -1.0, 1.0])


def build(points: np.ndarray, rgb: np.ndarray, mask: np.ndarray, K: np.ndarray, size: int) -> Build:
    """Snap the cloud to blocks so its longest side is `size` blocks."""
    h = mask.shape[0]
    m = mask & (points[..., 2] > 0)
    p = to_minecraft_axes(points[m].astype(np.float64))
    col = rgb[m]
    lo, hi = np.percentile(p, OUTLIER_PCT, 0), np.percentile(p, 100 - OUTLIER_PCT, 0)
    keep = ((p >= lo) & (p <= hi)).all(1)
    p, col = p[keep], col[keep]
    s = float((hi - lo).max()) / size
    dims = np.minimum(np.floor((hi - lo) / s).astype(int) + 1, size)
    idx = np.minimum(np.floor((p - lo) / s).astype(int), dims - 1)

    # One block per occupied cell, coloured by the mean of its points in linear light.
    lin = (idx[:, 1] * dims[2] + idx[:, 2]) * dims[0] + idx[:, 0]
    cells, inv = np.unique(lin, return_inverse=True)
    n = np.bincount(inv)
    mean = np.stack([np.bincount(inv, _linear(col[:, k].astype(np.float64))) for k in range(3)], -1) / n[:, None]
    ijk = np.stack([cells % dims[0], cells // (dims[0] * dims[2]), (cells // dims[0]) % dims[2]], -1)

    camera = (0.0 - lo) / s  # the input camera sits at the origin of its own frame
    offset = tuple(int(v) for v in np.round(EYE - camera))
    fov = math.degrees(2 * math.atan(h / 2 / float(K[1, 1])))
    return Build(ijk, nearest_blocks(mean), tuple(int(d) for d in dims), s, lo, camera, offset, fov)


# --- Sponge schematic v2 (gzipped NBT) ---------------------------------------------------------

def _name(tag: int, name: str) -> bytes:
    b = name.encode()
    return struct.pack(">bH", tag, len(b)) + b


def _int(name, v):
    return _name(3, name) + struct.pack(">i", v)


def _short(name, v):
    return _name(2, name) + struct.pack(">h", v)


def _int_array(name, vs):
    return _name(11, name) + struct.pack(">i", len(vs)) + struct.pack(f">{len(vs)}i", *vs)


def _byte_array(name, data: bytes):
    return _name(7, name) + struct.pack(">i", len(data)) + data


def _compound(name, *items: bytes):
    return _name(10, name) + b"".join(items) + b"\x00"


def schematic(b: Build) -> bytes:
    """Sponge schematic v2. BlockData is ordered x + z*Width + y*Width*Length; with fewer than 128
    palette entries every varint is a single byte. WorldEdit pastes block (x, y, z) at
    player + (x, y, z) + WEOffset, so WEOffset carries the camera alignment."""
    W, H, L = b.dims
    names = ["minecraft:air"] + BLOCK_NAMES
    grid = np.zeros(W * H * L, np.uint8)  # 0 = air
    grid[b.ijk[:, 0] + b.ijk[:, 2] * W + b.ijk[:, 1] * W * L] = b.block + 1
    ox, oy, oz = b.offset
    root = _compound(
        "Schematic",
        _int("Version", 2),
        _int("DataVersion", DATA_VERSION),
        _short("Width", W), _short("Height", H), _short("Length", L),
        _int_array("Offset", [0, 0, 0]),
        _compound("Metadata", _int("WEOffsetX", ox), _int("WEOffsetY", oy), _int("WEOffsetZ", oz)),
        _int("PaletteMax", len(names)),
        _compound("Palette", *(_int(n, i) for i, n in enumerate(names))),
        _byte_array("BlockData", grid.tobytes()),
        _name(9, "BlockEntities") + struct.pack(">bi", 10, 0),  # empty list of compounds
    )
    return gzip.compress(root)
