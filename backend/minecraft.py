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
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import boxify

DATA_VERSION = 3465  # Minecraft 1.20.1: the palette was measured from 1.20 blocks
WORLD_HEIGHT = 384  # 1.18+: y = -64 .. 319
EYE = np.array([0.5, 1.62, 0.5])  # player eye position inside the block the player stands in
OUTLIER_PCT = 0.5  # % of points ignored at each end of each axis (stray points)
# mode -> allowed value range. auto: target thousands of blocks; meters: block size; fit: longest side;
# eye: the camera's height above the floor becomes a player's eye height, times `value`.
MODES = {"auto": (20, 8000), "meters": (0.01, 10.0), "fit": (16, 4096), "eye": (0.1, 10.0)}
SHAPES = ("voxels", "boxes")  # boxes: buildings as solid prisms on a grid along their walls (boxify)
EYE_CONE = 60.0  # degrees around straight down in which the floor is looked for
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
    if np.linalg.norm(south) < 0.05:  # looking (nearly) straight down or up: face the image's top
        top = np.array([0.0, -1.0, 0.0])  # an image's top in its camera frame
        south = top - (top @ up) * up
    south /= np.linalg.norm(south)
    east = np.cross(up, south)  # facing south, east is on your left
    pitch = -math.degrees(math.asin(float(np.clip(fwd @ up, -1, 1))))
    return np.stack([east, up, south], 1), pitch


def grid_normals(P: np.ndarray, mask: np.ndarray, step: int = 4) -> np.ndarray:
    """Unit surface normals of an image grid of points (H, W, 3), from neighbouring pixels, every
    `step` pixels, where the pixel and its neighbours are valid. Sign is arbitrary."""
    c = P[step:-step:step, step:-step:step]
    dx = P[step:-step:step, 2 * step::step] - P[step:-step:step, :-2 * step:step]
    dy = P[2 * step::step, step:-step:step] - P[:-2 * step:step, step:-step:step]
    k = min(len(c), len(dx), len(dy)), min(c.shape[1], dx.shape[1], dy.shape[1])
    n = np.cross(dx[:k[0], :k[1]], dy[:k[0], :k[1]])
    m = mask[step:-step:step, step:-step:step][:k[0], :k[1]]
    m = m & mask[step:-step:step, 2 * step::step][:k[0], :k[1]] & mask[2 * step::step, step:-step:step][:k[0], :k[1]]
    n = n[m]
    length = np.linalg.norm(n, axis=1)
    return n[length > 0] / length[length > 0, None]


def _half_sphere(step_deg: float) -> np.ndarray:
    """Unit directions covering a half sphere (a direction and its opposite are the same axis)."""
    out = []
    for th in np.arange(0, 90.01, step_deg):
        k = max(1, int(round(360 * math.sin(math.radians(th)) / step_deg)))
        for ph in np.linspace(0, 2 * math.pi, k, endpoint=False):
            t = math.radians(th)
            out.append((math.sin(t) * math.cos(ph), math.sin(t) * math.sin(ph), math.cos(t)))
    return np.array(out)


def _axis_share(N: np.ndarray, D: np.ndarray, tol: float = 3.0) -> np.ndarray:
    """For each direction in D: share of normals parallel or perpendicular to it (within tol)."""
    d = np.abs(N @ D.T)
    return ((d > math.cos(math.radians(tol))) | (d < math.sin(math.radians(tol)))).mean(0)


def find_axes(N: np.ndarray) -> np.ndarray | None:
    """The three square-to-each-other axes of built things (floors/roofs and the two wall
    directions) from surface normals N: the direction most normals are parallel or perpendicular
    to, then the main wall direction around it. Rows of a (3, 3) array; None if too few normals."""
    if len(N) < 100:
        return None
    N = N[np.random.default_rng(0).choice(len(N), min(len(N), 20000), replace=False)]
    D = _half_sphere(2.0)
    a = D[np.argmax(_axis_share(N, D))]
    jitter = np.random.default_rng(1)
    for spread in (1.5, 0.5):  # refine around the best direction
        cand = a + jitter.normal(0, math.radians(spread), (300, 3))
        cand = np.vstack([a, cand / np.linalg.norm(cand, axis=1, keepdims=True)])
        a = cand[np.argmax(_axis_share(N, cand))]
    # The main wall direction: among surfaces parallel to `a`, the normal direction most share.
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    turns = np.radians(np.arange(0, 180, 0.5))
    E = np.cos(turns)[:, None] * u + np.sin(turns)[:, None] * np.cross(a, u)
    side = N[np.abs(N @ a) < math.sin(math.radians(9))]
    b = E[np.argmax((np.abs(side @ E.T) > math.cos(math.radians(3))).mean(0))] if len(side) > 50 else u
    return np.stack([a, b, np.cross(a, b)])


def find_up(N: np.ndarray, prior: np.ndarray, max_angle: float = 30.0) -> tuple[np.ndarray, float]:
    """Gravity "up" from surface normals. Built things are square: floors, roofs and tables face up,
    walls face sideways along two directions. Find those three axes (find_axes) and take the axis
    nearest `prior`, if it's within max_angle of it; otherwise keep `prior`. Returns (up, share of
    normals lined up with it).

    The walls carry the signal when level surfaces are few: on a Roblox city seen 40 degrees from
    above, averaging level surfaces put "up" 20 degrees off; this finds the direction that makes 23%
    of surfaces exactly vertical instead of 5%. Within half a degree of the truth on a synthetic room."""
    prior = prior / np.linalg.norm(prior)
    axes = find_axes(N)
    if axes is None:
        return prior, 0.0
    k = int(np.argmax(np.abs(axes @ prior)))
    up = axes[k] * np.sign(axes[k] @ prior)
    if up @ prior < math.cos(math.radians(max_angle)):
        return prior, 0.0
    return up, float(_axis_share(N, up[None])[0])


def wall_facing(N: np.ndarray, up: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """The wall direction (from find_axes) nearest `forward`, made level: a grid facing it has
    walls along its block rows. `forward` itself if no wall direction is found."""
    up = up / np.linalg.norm(up)
    axes = find_axes(N)
    if axes is not None:
        axes = axes - (axes @ up)[:, None] * up  # level, and the axis nearest "up" shrinks away
        axes = axes[np.argsort(-np.linalg.norm(axes, axis=1))[:2]]
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        k = int(np.argmax(np.abs(axes @ forward)))
        forward = axes[k] * (np.sign(axes[k] @ forward) or 1.0)
    return forward


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:  # opposite: turn 180 degrees about any axis perpendicular to a
        p = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


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
    pose: Pose = field(default_factory=lambda: IMAGE_POSE)  # sets the grid's axes; the viewer unless view_from moved it
    yaw_deg: float = 0.0  # in-game yaw to look along the camera (0 = south)


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


def _finish(p, rgb, lo, hi, s, pose, fov_deg, budget, progress, limited=False, target=None, clear=True) -> Build:
    """`target`: "auto" builds aim at this many blocks; if the sizing passes undershot the size,
    blocks are enlarged here (the samples are dense enough for bigger blocks, not smaller).
    `clear`: empty the space around the viewer's eyes (view_from does it later otherwise)."""
    # Inside the bounds, and not in the space around the viewer's eyes: anything that close (a
    # third-person character, a railing) would become giant blocks covering the view.
    keep = ((p >= lo) & (p <= hi)).all(1)
    if clear:
        keep &= np.linalg.norm(p, axis=1) > CLEAR_BLOCKS * s
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


def eye_height(grids) -> float:
    """The camera's height above the floor it stands on (grids in Minecraft axes, y up). A level
    floor is the same drop below the camera wherever it is seen, so drops of points inside a cone
    around straight down pile up at each level below; per frame, the nearest level that holds at
    least a third as many points as the biggest one is the floor (looking over a roof's edge, the
    street below can outnumber the roof); the median over frames."""
    t = math.tan(math.radians(EYE_CONE))
    hs = []
    for P, _, m, c in grids:
        d = P[m] - c
        drop = -d[:, 1]
        drop = drop[(drop > 0) & (np.hypot(d[:, 0], d[:, 2]) < t * drop)]
        if len(drop) < 200:
            continue
        n, edges = np.histogram(drop, bins=np.arange(0, 3 * np.median(drop), 0.02 * np.median(drop)))
        sm = np.convolve(n, [1, 1, 1], "same")  # a level split across two bins still counts whole
        peak = (sm >= sm.max() / 3) & (sm >= np.roll(sm, 1)) & (sm >= np.roll(sm, -1))
        k = int(np.argmax(peak))
        k = max(k - 1, 0) + int(np.argmax(n[max(k - 1, 0):k + 2]))  # the fullest bin of that level
        hs.append(edges[k] + 0.01 * np.median(drop))
    if not hs:
        raise ValueError("eye size needs the floor in view: look down at it now and then while recording")
    return float(np.median(hs))


def build_from_grids(grids, pose: Pose, fov_deg: float, mode: str = "auto", value: float = 300,
                     budget: int = DEFAULT_BUDGET, progress=_quiet, clear: bool = True,
                     shape: str = "voxels") -> Build:
    """Block build from one or more image grids [(points (H, W, 3), rgb (H, W, 3), mask, camera
    centre)], all in the same cloud frame, viewed from `pose`. Surfaces between pixels are filled.

    Block size by `mode`:
      "auto"   - about `value` thousand blocks: detail follows how much surface the scene has, so a
                 wide scene doesn't get spread thin;
      "meters" - one block = `value` cloud units (metres for metric models);
      "fit"    - longest side of `value` blocks;
      "eye"    - character-sized: the camera's height above the floor becomes a player's eye
                 height (1.62 blocks), times `value`.
    If a build would exceed `budget` blocks, blocks grow until it fits.

    shape "boxes": the grid is turned to run along the walls and buildings become solid prisms
    (boxify); the result is then seen from `pose` again (view_from)."""
    progress("placing points")
    view = pose
    if shape == "boxes":
        N = np.concatenate([grid_normals(P.astype(np.float64), m & np.isfinite(P).all(-1)) for P, _, m, _ in grids])
        pose = Pose(pose.position, wall_facing(N, pose.up, pose.forward), pose.up)
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
    elif mode == "eye":
        s = eye_height(grids) / (EYE[1] * value)
    else:
        s = float(value) if mode == "meters" else float((hi - lo).max()) / value
    progress("filling surfaces")
    p, rgb = samples(s, stage="filling surfaces")
    b = _finish(p, rgb, lo, hi, s, pose, fov_deg, budget, progress, target=value * 1000 if mode == "auto" else None, clear=clear)
    if shape == "boxes":
        progress("making boxes")
        ijk, block = boxify.boxes(b.ijk, b.block, b.dims)
        b = replace(b, ijk=ijk, block=block)
        b = view_from(b, view, fov_deg) if clear else b  # scenes: the caller places the viewer
    return b


def view_from(b: Build, pose: Pose, fov_deg: float) -> Build:
    """The same blocks, seen from another camera (e.g. another frame of a video scene): where to
    stand (paste offset), yaw, pitch and FOV, with the blocks around that eye cleared. The grid keeps
    its axes, so this costs milliseconds instead of a rebuild."""
    B, _ = _basis(b.pose)
    eye = (np.asarray(pose.position, float) - b.pose.position) @ B  # Minecraft axes of the build
    camera = (eye - b.lo) / b.block_size
    keep = np.linalg.norm(b.ijk + 0.5 - camera, axis=1) > CLEAR_BLOCKS
    f = (pose.forward / np.linalg.norm(pose.forward)) @ B
    yaw = math.degrees(math.atan2(-f[0], f[2]))  # Minecraft: yaw 0 looks south (+z), 90 west (-x)
    pitch = -math.degrees(math.asin(float(np.clip(f[1], -1, 1))))
    offset = tuple(int(v) for v in np.round(EYE - camera))
    return replace(b, ijk=b.ijk[keep], block=b.block[keep], camera=camera, offset=offset, fov_deg=fov_deg,
                   pitch_deg=pitch, yaw_deg=yaw)


def image_up(points: np.ndarray, mask: np.ndarray, above: bool = False) -> np.ndarray:
    """"Up" for a single picture, in its camera frame, from its structure (find_up): the axis of its
    floors, roofs and walls nearest the camera's own up, up to 60 degrees away, so a tilted or
    downward-looking shot still comes out level. `above`: the picture looks (nearly) straight down
    on the ground, like a top-down map; take the axis nearest the camera's backward direction
    instead (straight down, the ground's axis and a wall's are otherwise a toss-up)."""
    m = mask & (points[..., 2] > 0)
    prior = np.array([0.0, 0.0, -1.0]) if above else np.array([0.0, -1.0, 0.0])
    return find_up(grid_normals(points.astype(np.float64), m, step=2), prior, max_angle=60.0)[0]


def build_from_image(points, rgb, mask, K, mode="auto", value=300, budget=DEFAULT_BUDGET, progress=_quiet,
                     up: np.ndarray | None = None, shape: str = "voxels") -> Build:
    """A single-image run: points (H, W, 3) in its camera frame, viewer at that camera, `up` from
    image_up (computed here if not given)."""
    m = mask & (points[..., 2] > 0)
    fov = math.degrees(2 * math.atan(mask.shape[0] / 2 / float(K[1, 1])))
    P = points.astype(np.float64)
    pose = Pose(np.zeros(3), np.array([0.0, 0.0, 1.0]), image_up(points, mask) if up is None else np.asarray(up, float))
    return build_from_grids([(P, rgb, m, np.zeros(3))], pose, fov, mode, value, budget, progress, shape=shape)


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
        f"should be seen from, look along it, and DO NOT MOVE while pasting: /tp @s ~ ~ ~ {b.yaw_deg:.0f} {b.pitch_deg:.0f}",
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
