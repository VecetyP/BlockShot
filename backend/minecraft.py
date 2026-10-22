"""Point cloud -> Minecraft build that looks like the input from one spot.

The cloud is turned into Minecraft's axes around a standing camera (facing south, "up" = level),
snapped to a block grid, and each block takes the palette block closest in colour (CIELAB). Output
is Sponge v2 schematics (.schem) with WorldEdit's paste offset set so that `//paste` puts the camera
at the player's eyes; big builds are split into tiles that all paste from the same spot.
"""
import gzip
import io
import json
import math
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DATA_VERSION = 3465  # Minecraft 1.20.1: the palette was measured from 1.20 blocks
WORLD_HEIGHT = 384  # 1.18+: y = -64 .. 319
EYE = np.array([0.5, 1.62, 0.5])  # player eye position inside the block the player stands in
OUTLIER_PCT = 0.5  # % of points ignored at each end of each axis (stray points)
# mode -> allowed value range. auto: target thousands of blocks; meters: block size; fit: longest side.
MODES = {"auto": (20, 8000), "meters": (0.01, 10.0), "fit": (16, 4096)}
DEFAULT_BUDGET = 4_000_000  # blocks; past this the block size grows until the build fits
TEAR = 0.05  # neighbouring pixels further apart than 5% of their distance are separate surfaces
SIZING_SAMPLES = 4_000_000  # per sizing pass, so "auto" doesn't fill the surface several times over
MAX_SAMPLES = 30_000_000  # cap on points after filling surfaces between pixels
CLEAR_BLOCKS = 4  # keep this many blocks around the viewer's eyes empty (own character, railings)
MAX_SINGLE_CELLS = 64_000_000  # larger builds only export as tiles (WorldEdit loads dense volumes)
PREVIEW_MAX = 400_000  # cubes the browser preview draws; larger builds preview coarser

_palette = json.loads((Path(__file__).parent / "minecraft_palette.json").read_text())["blocks"]
BLOCK_NAMES = list(_palette)
BLOCK_RGB = np.array(list(_palette.values()), np.uint8)
assert len(BLOCK_NAMES) + 1 < 1 << 14, "BlockData varints are written with at most 2 bytes"


def _linear(srgb: np.ndarray) -> np.ndarray:
    c = srgb / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(linear_rgb: np.ndarray) -> np.ndarray:
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = linear_rgb @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], -1)


_BLOCK_LAB = _lab(_linear(BLOCK_RGB.astype(np.float64)))


def _quiet(*_, **__):
    pass


def nearest_blocks(linear_rgb: np.ndarray, progress=_quiet) -> np.ndarray:
    """Palette index of the closest block in CIELAB for each colour (linear light, 0..1)."""
    lab = _lab(linear_rgb)
    out = np.empty(len(lab), np.int64)
    step = 65536  # chunks keep the distance matrix small
    for i in range(0, len(lab), step):
        progress("matching block colours", i, len(lab))
        d = ((lab[i:i + step, None, :] - _BLOCK_LAB[None]) ** 2).sum(-1)
        out[i:i + step] = d.argmin(1)
    return out


# --- camera and axes ---------------------------------------------------------------------------

@dataclass
class Pose:
    """Where the viewer stands, in the cloud's own coordinates."""
    position: np.ndarray
    forward: np.ndarray  # viewing direction
    up: np.ndarray  # what should be "up" in the game (level floors)


# A single image: camera at the origin of its OpenCV frame (+x right, +y down, +z forward).
IMAGE_POSE = Pose(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]))


def _basis(pose: Pose) -> tuple[np.ndarray, float]:
    """Columns = Minecraft x (east), y (up), z (south) in cloud coordinates, with the viewer facing
    south; plus the pitch to look along `forward` (Minecraft pitch: positive looks down)."""
    up = pose.up / np.linalg.norm(pose.up)
    fwd = pose.forward / np.linalg.norm(pose.forward)
    south = fwd - (fwd @ up) * up
    south /= np.linalg.norm(south)
    east = np.cross(up, south)  # facing south, east is on your left
    pitch = -math.degrees(math.asin(float(np.clip(fwd @ up, -1, 1))))
    return np.stack([east, up, south], 1), pitch


def to_minecraft(points: np.ndarray, pose: Pose = IMAGE_POSE) -> np.ndarray:
    """Cloud coordinates -> Minecraft axes, with the viewer at the origin."""
    B, _ = _basis(pose)
    return (points - pose.position) @ B


def to_minecraft_axes(p: np.ndarray) -> np.ndarray:
    """Single-image camera frame <-> Minecraft axes (the map is its own inverse)."""
    return p * np.array([-1.0, -1.0, 1.0])


def point_spacing(p: np.ndarray, sample: int = 50_000) -> float:
    """Median distance from a point to its nearest neighbour (on a random sample)."""
    rng = np.random.default_rng(0)
    q = p[rng.choice(len(p), min(sample, len(p)), replace=False)] if len(p) > sample else p
    d, _ = cKDTree(p).query(q, k=2)
    return float(np.median(d[:, 1]))


def surface_samples(P: np.ndarray, rgb: np.ndarray, mask: np.ndarray, center: np.ndarray, s: float,
                    max_samples: int = MAX_SAMPLES):
    """Points of an image grid (P (H, W, 3), already in Minecraft axes) plus extra samples inside
    every 2x2 pixel cell whose surface patch is bigger than a block, so distant surfaces come out
    solid instead of dotted. Cells spanning a depth tear (a gap between separate surfaces) are not
    filled. `center` is the camera position, for judging tears relative to distance."""
    pts, cols = [P[mask]], [rgb[mask].astype(np.float32)]
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    ok = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    edge = np.maximum.reduce([np.linalg.norm(x - y, axis=-1) for x, y in ((b, a), (c, a), (d, b), (d, c))])
    dist = np.linalg.norm(a - center, axis=-1)
    ok &= edge <= TEAR * dist
    up = np.where(ok, np.ceil(edge / (0.75 * s)), 1).astype(np.int64)
    max_up = 16
    while max_up > 1 and (np.minimum(up, max_up) ** 2)[up > 1].sum() > max_samples:
        max_up //= 2
    up = np.minimum(up, max_up)
    ca, cb, cc, cd = (x.astype(np.float32) for x in (rgb[:-1, :-1], rgb[:-1, 1:], rgb[1:, :-1], rgb[1:, 1:]))
    for k in range(2, max_up + 1):
        cells = up == k
        if not cells.any():
            continue
        f = (np.arange(k) + 0.5) / k
        fy, fx = (g.ravel() for g in np.meshgrid(f, f, indexing="ij"))
        w = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], -1)  # (k², 4) bilinear
        corners = np.stack([a[cells], b[cells], c[cells], d[cells]], 1)  # (n, 4, 3)
        ccols = np.stack([ca[cells], cb[cells], cc[cells], cd[cells]], 1)
        pts.append(np.einsum("sc,ncx->nsx", w, corners).reshape(-1, 3))
        cols.append(np.einsum("sc,ncx->nsx", w, ccols).reshape(-1, 3))
    return np.concatenate(pts).astype(np.float32), np.concatenate(cols)


@dataclass
class Build:
    ijk: np.ndarray  # (N, 3) block coordinates (x east, y up, z south) from the build's min corner
    block: np.ndarray  # (N,) palette index
    dims: tuple[int, int, int]  # (width x, height y, length z) in blocks
    block_size: float  # one block, in the cloud's units (metres for metric models)
    lo: np.ndarray  # (3,) min corner in Minecraft axes (cloud units)
    camera: np.ndarray  # (3,) the viewer in block coordinates (continuous)
    offset: tuple[int, int, int]  # block coordinate + offset = position relative to the player's feet block
    fov_deg: float  # vertical field of view to set in game
    pitch_deg: float  # in-game pitch to look along the camera
    budget_limited: bool = False  # the block budget made blocks bigger than asked
    pose: Pose = field(default_factory=lambda: IMAGE_POSE)


def _keys(p, lo, s, dims):
    idx = np.clip(np.floor((p - lo) / s).astype(np.int64), 0, np.array(dims) - 1)
    return (idx[:, 1] * dims[2] + idx[:, 2]) * dims[0] + idx[:, 0]


def _dims(lo, hi, s):
    return tuple(int(d) for d in np.floor((hi - lo) / s).astype(int) + 1)


def _voxelize(p, lin, lo, s, dims):
    cells, inv = np.unique(_keys(p, lo, s, dims), return_inverse=True)
    n = np.bincount(inv)
    mean = np.stack([np.bincount(inv, lin[:, k]) for k in range(3)], -1) / n[:, None]
    ijk = np.stack([cells % dims[0], cells // (dims[0] * dims[2]), (cells // dims[0]) % dims[2]], -1)
    return ijk, mean


def _finish(p, rgb, lo, hi, s, pose, fov_deg, budget, progress, limited=False, target=None) -> Build:
    """`target`: "auto" builds aim at this many blocks; if the sizing passes undershot the size,
    blocks are enlarged here (the samples are dense enough for bigger blocks, not smaller)."""
    # Inside the bounds, and not in the space around the viewer's eyes: anything that close (a
    # third-person character, a railing) would become giant blocks covering the view.
    keep = ((p >= lo) & (p <= hi)).all(1) & (np.linalg.norm(p, axis=1) > CLEAR_BLOCKS * s)
    p, lin = p[keep], _linear(np.asarray(rgb)[keep].astype(np.float64))
    for _ in range(6):
        progress("placing blocks")
        dims = _dims(lo, hi, s)
        ijk, mean = _voxelize(p, lin, lo, s, dims)
        if target and budget >= len(ijk) > 1.15 * target:
            s *= math.sqrt(len(ijk) / target)
            continue
        if len(ijk) <= budget:
            break
        s *= math.sqrt(len(ijk) / budget) * 1.05  # surfaces: block count ~ 1/size²
        limited = True
    block = nearest_blocks(mean, progress)
    camera = (0.0 - lo) / s  # the viewer sits at the origin of the Minecraft frame
    offset = tuple(int(v) for v in np.round(EYE - camera))
    _, pitch = _basis(pose)
    return Build(ijk.astype(np.int32), block.astype(np.int16), dims, s, lo, camera, offset, fov_deg,
                 pitch, limited, pose)


def _bounds(p):
    return np.percentile(p, OUTLIER_PCT, 0), np.percentile(p, 100 - OUTLIER_PCT, 0)


def build(points: np.ndarray, rgb: np.ndarray, pose: Pose, fov_deg: float, mode: str = "meters",
          value: float = 0.25, budget: int = DEFAULT_BUDGET, progress=_quiet) -> Build:
    """Block build of a flat cloud (points (N, 3), rgb (N, 3) uint8) seen from `pose`; no surface
    filling (that needs the image grid: see build_from_grids). Modes as in build_from_grids."""
    progress("placing points")
    p = to_minecraft(points.astype(np.float64), pose)
    lo, hi = _bounds(p)
    if mode == "auto":  # flat clouds have no surfaces to fill: size from the occupied-cell count
        s = _size_for_count(lambda s: p, lo, hi, value * 1000)
    else:
        s = float(value) if mode == "meters" else float((hi - lo).max()) / value
    return _finish(p, rgb, lo, hi, s, pose, fov_deg, budget, progress)


def _size_for_count(samples_at, lo, hi, target: float, progress=_quiet) -> float:
    """Block size giving about `target` blocks. Surfaces: count ~ 1/size², refined a few times."""
    s = float((hi - lo).max()) / 256
    for i in range(4):
        progress("sizing blocks", i, 4)
        n = len(np.unique(_keys(samples_at(s), lo, s, _dims(lo, hi, s))))
        if abs(n / target - 1) < 0.1:
            break
        s *= math.sqrt(n / target)
    return s


def build_from_grids(grids, pose: Pose, fov_deg: float, mode: str = "auto", value: float = 300,
                     budget: int = DEFAULT_BUDGET, progress=_quiet) -> Build:
    """Block build from one or more image grids [(points (H, W, 3), rgb (H, W, 3), mask, camera
    centre)], all in the same cloud frame, viewed from `pose`. Surfaces between pixels are filled.

    Block size by `mode`:
      "auto"   - about `value` thousand blocks: detail follows how much surface the scene has, so a
                 wide scene doesn't get spread thin;
      "meters" - one block = `value` cloud units (metres for metric models);
      "fit"    - longest side of `value` blocks.
    If a build would exceed `budget` blocks, blocks grow until it fits."""
    progress("placing points")
    B, _ = _basis(pose)
    grids = [((P - pose.position) @ B, rgb, mask & np.isfinite(P).all(-1), (c - pose.position) @ B)
             for P, rgb, mask, c in grids]
    lo, hi = _bounds(np.concatenate([P[m] for P, _, m, _ in grids]))

    def samples(s, total=MAX_SAMPLES, stage=None):
        cap = total // len(grids)  # the cap is shared by all frames
        out = []
        for i, (P, rgb, m, c) in enumerate(grids):
            if stage and len(grids) > 1:
                progress(stage, i, len(grids), f"frame {i + 1}/{len(grids)}")  # also lets a cancelled job stop here
            out.append(surface_samples(P, rgb, m, c, s, cap))
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

    if mode == "auto":  # sizing only counts occupied cells, so a sparser fill is close enough
        s = _size_for_count(lambda s: samples(s, SIZING_SAMPLES, "sizing blocks")[0], lo, hi, value * 1000, progress)
    else:
        s = float(value) if mode == "meters" else float((hi - lo).max()) / value
    progress("filling surfaces")
    p, rgb = samples(s, stage="filling surfaces")
    return _finish(p, rgb, lo, hi, s, pose, fov_deg, budget, progress, target=value * 1000 if mode == "auto" else None)


def build_from_image(points, rgb, mask, K, mode="auto", value=300, budget=DEFAULT_BUDGET, progress=_quiet) -> Build:
    """A single-image run: points (H, W, 3) in its camera frame, viewer at that camera."""
    m = mask & (points[..., 2] > 0)
    fov = math.degrees(2 * math.atan(mask.shape[0] / 2 / float(K[1, 1])))
    return build_from_grids([(points.astype(np.float64), rgb, m, np.zeros(3))], IMAGE_POSE, fov, mode, value, budget, progress)


def preview(b: Build, max_cubes: int = PREVIEW_MAX) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Cubes for the browser: (ijk, rgb, cube size, lod). Big builds merge lod³ blocks per cube."""
    rgb = BLOCK_RGB[b.block]
    lod, ijk = 1, b.ijk
    while len(ijk) > max_cubes:
        lod *= 2
        ijk, inv = np.unique(b.ijk // lod, axis=0, return_inverse=True)
    if lod > 1:
        n = np.bincount(inv.ravel())
        rgb = (np.stack([np.bincount(inv.ravel(), rgb[:, k]) for k in range(3)], -1) / n[:, None]).astype(np.uint8)
    return ijk, rgb, b.block_size * lod, lod


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


def varints(v: np.ndarray) -> bytes:
    """Unsigned varints (7 bits per byte, high bit = more follows) for values below 2^14."""
    v = v.astype(np.int64)
    big = v >= 128
    starts = np.concatenate([[0], np.cumsum(1 + big)[:-1]])
    out = np.empty(len(v) + int(big.sum()), np.uint8)
    out[starts] = np.where(big, (v & 0x7F) | 0x80, v)
    out[starts[big] + 1] = v[big] >> 7
    return out.tobytes()


def schematic(ijk: np.ndarray, block: np.ndarray, dims, offset) -> bytes:
    """Sponge schematic v2 of blocks at `ijk` (from the min corner) in a `dims` volume. BlockData is
    ordered x + z*Width + y*Width*Length. WorldEdit pastes block (x, y, z) at
    player + (x, y, z) + WEOffset, so WEOffset carries the camera alignment."""
    W, H, L = (int(d) for d in dims)
    names = ["minecraft:air"] + BLOCK_NAMES
    grid = np.zeros(W * H * L, np.int16)  # 0 = air
    grid[ijk[:, 0] + ijk[:, 2] * W + ijk[:, 1] * W * L] = block + 1
    ox, oy, oz = (int(o) for o in offset)
    root = _compound(
        "Schematic",
        _int("Version", 2),
        _int("DataVersion", DATA_VERSION),
        _short("Width", W), _short("Height", H), _short("Length", L),
        _int_array("Offset", [0, 0, 0]),
        _compound("Metadata", _int("WEOffsetX", ox), _int("WEOffsetY", oy), _int("WEOffsetZ", oz)),
        _int("PaletteMax", len(names)),
        _compound("Palette", *(_int(n, i) for i, n in enumerate(names))),
        _byte_array("BlockData", varints(grid)),
        _name(9, "BlockEntities") + struct.pack(">bi", 10, 0),  # empty list of compounds
    )
    return gzip.compress(root)


def whole(b: Build) -> bytes:
    if math.prod(b.dims) > MAX_SINGLE_CELLS:
        raise ValueError("build too large for one schematic; export tiles")
    return schematic(b.ijk, b.block, b.dims, b.offset)


def tiles(b: Build, size: int = 128) -> list[tuple[int, int, bytes]]:
    """Split into size x size columns (trimmed to their blocks). Each tile's offset includes its
    origin, so pasting every tile from the same standing spot rebuilds the whole build."""
    out = []
    key = (b.ijk[:, 0] // size) * 1_000_003 + b.ijk[:, 2] // size
    order = np.argsort(key, kind="stable")
    keys, starts = np.unique(key[order], return_index=True)
    for k, a, z in zip(keys, starts, np.r_[starts[1:], len(order)]):
        sel = order[a:z]
        local = b.ijk[sel]
        m = local.min(0)
        local = local - m
        offset = np.array(b.offset) + m
        out.append((int(k // 1_000_003), int(k % 1_000_003), schematic(local, b.block[sel], local.max(0) + 1, offset)))
    return out


def tiles_zip(b: Build, base: str, size: int = 128, progress=_quiet) -> bytes:
    progress("writing tiles")
    parts = tiles(b, size)
    lines = [
        f"BlockShot build: {len(parts)} tiles, {b.dims[0]} x {b.dims[1]} x {b.dims[2]} blocks.",
        "Copy the .schem files into WorldEdit's schematics folder. Then stand where the picture",
        f"should be seen from, look along it, and DO NOT MOVE while pasting: /tp @s ~ ~ ~ 0 {b.pitch_deg:.0f}",
        "",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # .schem files are already gzipped
        for i, (tx, tz, data) in enumerate(parts):
            progress("writing tiles", i, len(parts))
            name = f"{base}_x{tx}_z{tz}"
            z.writestr(f"{name}.schem", data)
            lines += [f"//schem load {name}", "//paste"]
        lines += ["", f"Then set FOV to {b.fov_deg:.0f} (FOV Effects off) and press F1."]
        z.writestr("paste-commands.txt", "\n".join(lines) + "\n")
    return buf.getvalue()
