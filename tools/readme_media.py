"""Render the README's pictures from runs: point clouds splatted with a z-buffer, no GPU needed.

    uv run python tools/readme_media.py triptych <image run> <out.jpg>    input | cloud from the camera | cloud from 30° off
    uv run python tools/readme_media.py build <image run> <out.jpg>       input | cloud from 30° off | its Minecraft build from 30° off
    uv run python tools/readme_media.py scene <scene run> <out.jpg>       frame 0 | the whole scene's cloud from a step back and up | its build from there
    uv run python tools/readme_media.py orbit <image run> <out.gif>       the cloud swinging away from the camera and back
    uv run python tools/readme_media.py walk <scene run> <out.gif>        a walk along the scene's camera path
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import minecraft, store, video  # noqa: E402

BG = (11, 12, 15)
WIDTH = 900  # px per triptych panel
GIF_WIDTH = 560
WALK_WIDTH = 440  # walks have more frames; smaller keeps the GIF under ~5 MB


def splat(P: np.ndarray, C: np.ndarray, size: np.ndarray, K: np.ndarray, w2c: np.ndarray, hw: tuple[int, int]) -> Image.Image:
    """Points P (world) with colours C, each `size` wide (world units), seen by camera w2c with K."""
    h, w = hw
    q = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = q[:, 2]
    ok = z > 0.05
    q, z, C, size = q[ok], z[ok], C[ok], size[ok]
    u = K[0, 0] * q[:, 0] / z + K[0, 2]
    v = K[1, 1] * q[:, 1] / z + K[1, 2]
    ok = (u > -3) & (u < w + 3) & (v > -3) & (v < h + 3)
    u, v, z, C, size = u[ok], v[ok], z[ok], C[ok], size[ok]
    r = np.clip(np.round(K[0, 0] * size / z / 2), 0, 2).astype(int)  # half-width of the square in px
    pix, zs, cols = [], [], []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            m = r >= max(abs(dx), abs(dy))
            x, y = (u[m] + dx).astype(int), (v[m] + dy).astype(int)
            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            pix.append((y * w + x)[inside]); zs.append(z[m][inside]); cols.append(C[m][inside])
    pix, zs, cols = np.concatenate(pix), np.concatenate(zs), np.concatenate(cols)
    order = np.argsort(-zs, kind="stable")  # far to near: numpy keeps the last write to a pixel
    img = np.empty((h * w, 3), np.uint8)
    img[:] = BG
    img[pix[order]] = cols[order]
    return Image.fromarray(img.reshape(h, w, 3))


def _axis_angle(axis: np.ndarray, deg: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def image_cloud(rid: str):
    """An image run's cloud in its camera frame, thinned to about one point per output pixel."""
    a = store.load_arrays(rid)
    rgb = store._rgb(rid)
    h, w = a["mask"].shape
    st = max(1, w // WIDTH)
    m = a["mask"][::st, ::st]
    P = a["points"][::st, ::st][m].astype(np.float64)
    C = rgb[::st, ::st][m]
    K = a["intrinsics"].astype(np.float64)
    size = P[:, 2] * st / K[0, 0] * 1.3  # each point covers its pixel, a little more to close gaps
    return rgb, P, C, size, K, np.array(store.run_up(rid), float)


def orbit_view(P: np.ndarray, up: np.ndarray, deg: float) -> np.ndarray:
    """w2c for the input camera swung `deg` around the cloud's middle, about its "up"."""
    cen = np.array([0, 0, np.median(P[:, 2])])
    R = _axis_angle(up, deg)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = cen - R @ cen
    return w2c


def triptych(rid: str, out: str) -> None:
    rgb, P, C, size, K, up = image_cloud(rid)
    h, w = rgb.shape[:2]
    s = WIDTH / w
    Ko = K * np.array([[s], [s], [1]])
    hw = (round(h * s), WIDTH)
    panels = [Image.fromarray(rgb).resize(hw[::-1], Image.LANCZOS),
              splat(P, C, size, Ko, np.eye(4), hw), splat(P, C, size, Ko, orbit_view(P, up, 30), hw)]
    gap = 12
    fig = Image.new("RGB", (WIDTH * 3 + gap * 2, hw[0]), BG)
    for i, p in enumerate(panels):
        fig.paste(p, (i * (WIDTH + gap), 0))
    fig.save(out, quality=86)
    print(f"{out}: {fig.size[0]}x{fig.size[1]}")


# (axis, side, shade): Minecraft lights tops fully, north/south faces at 0.8, east/west at 0.6.
FACES = [(1, 1, 1.0), (1, 0, 0.5), (2, 0, 0.8), (2, 1, 0.8), (0, 0, 0.6), (0, 1, 0.6)]


def blocks(b: minecraft.Build, K: np.ndarray, w2c: np.ndarray, hw: tuple[int, int]) -> Image.Image:
    """A build's exposed block faces (Minecraft face shading, far to near) seen by camera w2c, in
    the cloud's camera frame like `splat`."""
    h, w = hw
    B, _ = minecraft._basis(b.pose)  # columns: Minecraft x, y, z in cloud coordinates
    occupied = {tuple(p) for p in b.ijk}
    rgb = minecraft.BLOCK_RGB[b.block].astype(float)
    polys = []
    for axis, side, shade in FACES:
        step = np.zeros(3, int)
        step[axis] = 1 if side else -1
        exposed = np.array([tuple(p + step) not in occupied for p in b.ijk])
        a, c = [i for i in range(3) if i != axis]
        quad = np.zeros((4, 3))
        quad[:, axis] = side
        quad[:, a], quad[:, c] = [0, 1, 1, 0], [0, 0, 1, 1]
        mc = (b.ijk[exposed, None, :] + quad[None]) * b.block_size + b.lo
        cam = (mc.reshape(-1, 3) @ B.T + b.pose.position) @ w2c[:3, :3].T + w2c[:3, 3]
        cam = cam.reshape(-1, 4, 3)
        n = np.zeros(3)
        n[axis] = 1 if side else -1
        facing = (cam.mean(1) @ (w2c[:3, :3] @ B @ n)) < 0  # outward normal points back at the camera
        z = cam[..., 2]
        ok = (z > 1e-6).all(1) & facing
        uv = cam[..., :2] / z[..., None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        polys += zip(z[ok].mean(1), uv[ok], rgb[exposed][ok] * shade)
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    for _, q, col in sorted(polys, key=lambda p: -p[0]):
        draw.polygon([tuple(v) for v in q], fill=tuple(int(x) for x in col))
    return img


def build(rid: str, out: str, deg: float = 30, kblocks: int = 300) -> None:
    rgb, P, C, size, K, up = image_cloud(rid)
    a = store.load_arrays(rid)
    b = minecraft.build_from_image(a["points"], rgb, a["mask"], a["intrinsics"], "auto", kblocks, up=up)
    h, w = rgb.shape[:2]
    s = WIDTH / w
    Ko = K * np.array([[s], [s], [1]])
    hw = (round(h * s), WIDTH)
    view = orbit_view(P, up, deg)
    panels = [Image.fromarray(rgb).resize(hw[::-1], Image.LANCZOS), splat(P, C, size, Ko, view, hw), blocks(b, Ko, view, hw)]
    gap = 12
    fig = Image.new("RGB", (WIDTH * 3 + gap * 2, hw[0]), BG)
    for i, p in enumerate(panels):
        fig.paste(p, (i * (WIDTH + gap), 0))
    fig.save(out, quality=86)
    print(f"{out}: {fig.size[0]}x{fig.size[1]}, {len(b.ijk):,} blocks")


def _scene_cloud(sc: dict, stride: int):
    P, C, S = [], [], []
    for i in range(len(sc["depth"])):
        m = sc["mask"][i][::stride, ::stride]
        P.append(video.world_points(sc["depth"][i], sc["K"][i].astype(np.float64), sc["c2w"][i].astype(np.float64))[::stride, ::stride][m])
        C.append(sc["rgb"][i][::stride, ::stride][m])
        S.append(sc["depth"][i][::stride, ::stride][m] * stride / sc["K"][i][0, 0] * 1.3)
    return np.concatenate(P), np.concatenate(C), np.concatenate(S)


def scene(rid: str, out: str, frame: int = 0, deg: float = 0, back: float = 1.5, rise: float = 1.2, tilt: float = 12,
          kblocks: int = 300) -> None:
    sc = store.load_scene(rid)
    frame = int(frame)
    P, C, size = _scene_cloud(sc, 2)
    rgb = sc["rgb"][frame]
    h, w = rgb.shape[:2]
    s = WIDTH / w
    Ko = sc["K"][frame].astype(np.float64) * np.array([[s], [s], [1]])
    hw = (round(h * s), WIDTH)
    # A swing around the middle would put the camera in a wall of a corridor or room, so step back
    # `back` and up `rise` (metres; scenes are levelled with up = -y), turn `deg` and look down `tilt`.
    c2w = sc["c2w"][frame].astype(np.float64).copy()
    up = np.array([0.0, -1.0, 0.0])
    fwd = c2w[:3, 2] - (c2w[:3, 2] @ up) * up
    fwd /= np.linalg.norm(fwd)
    c2w[:3, 3] += rise * up - back * fwd
    c2w[:3, :3] = _axis_angle(up, deg) @ c2w[:3, :3] @ _axis_angle(np.array([1.0, 0, 0]), -tilt)
    view = np.linalg.inv(c2w)
    b = store.minecraft_build(rid, "auto", kblocks, frame=frame)
    panels = [Image.fromarray(rgb).resize(hw[::-1], Image.LANCZOS), splat(P, C, size, Ko, view, hw), blocks(b, Ko, view, hw)]
    gap = 12
    fig = Image.new("RGB", (WIDTH * 3 + gap * 2, hw[0]), BG)
    for i, p in enumerate(panels):
        fig.paste(p, (i * (WIDTH + gap), 0))
    fig.save(out, quality=86)
    print(f"{out}: {fig.size[0]}x{fig.size[1]}, {len(b.ijk):,} blocks")


def _save_gif(frames: list[Image.Image], out: str, ms: int, colors: int = 128) -> None:
    pal = frames[len(frames) // 2].quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=ms, loop=0, optimize=True)
    print(f"{out}: {len(frames)} frames, {Path(out).stat().st_size / 1e6:.1f} MB")


def orbit(rid: str, out: str) -> None:
    rgb, P, C, size, K, up = image_cloud(rid)
    h, w = rgb.shape[:2]
    s = GIF_WIDTH / w
    Ko = K * np.array([[s], [s], [1]])
    hw = (round(h * s), GIF_WIDTH)
    angles = np.concatenate([np.zeros(8), 35 * np.sin(np.linspace(0, 2 * np.pi, 56)), np.zeros(8)])
    _save_gif([splat(P, C, size, Ko, orbit_view(P, up, a), hw) for a in angles], out, 70)


def _slerp_pose(a: np.ndarray, b: np.ndarray, f: float) -> np.ndarray:
    Ra, Rb = a[:3, :3], b[:3, :3]
    M = Ra.T @ Rb
    ang = np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))
    out = np.eye(4)
    if ang < 1e-6:
        out[:3, :3] = Ra
    else:
        ax = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]) / (2 * np.sin(ang))
        out[:3, :3] = Ra @ _axis_angle(ax, np.degrees(ang) * f)
    out[:3, 3] = (1 - f) * a[:3, 3] + f * b[:3, 3]
    return out


def walk(rid: str, out: str, steps: int = 2, stride: int = 3) -> None:
    sc = store.load_scene(rid)
    n, h, w = sc["depth"].shape
    P, C, S = [], [], []
    for i in range(n):
        m = sc["mask"][i][::stride, ::stride]
        P.append(video.world_points(sc["depth"][i], sc["K"][i].astype(np.float64), sc["c2w"][i].astype(np.float64))[::stride, ::stride][m])
        C.append(sc["rgb"][i][::stride, ::stride][m])
        S.append(sc["depth"][i][::stride, ::stride][m] * stride / sc["K"][i][0, 0] * 1.3)
    P, C, S = np.concatenate(P), np.concatenate(C), np.concatenate(S)
    s = WALK_WIDTH / w
    K = sc["K"][0].astype(np.float64) * np.array([[s], [s], [1]])
    hw = (round(h * s), WALK_WIDTH)
    c2w = sc["c2w"].astype(np.float64)
    # Smooth the path a little (VGGT's per-frame poses jitter), then step between frames.
    poses = [_slerp_pose(c2w[i], c2w[i + 1], k / steps) for i in range(n - 1) for k in range(steps)]
    frames = [splat(P, C, S, K, np.linalg.inv(p), hw) for p in poses]
    _save_gif(frames, out, 1000 // (4 * steps), colors=96)


if __name__ == "__main__":
    {"triptych": triptych, "build": build, "scene": scene, "orbit": orbit, "walk": walk}[sys.argv[1]](*sys.argv[2:4], *map(float, sys.argv[4:]))
