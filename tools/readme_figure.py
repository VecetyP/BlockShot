"""Render the README figure: input image | its Minecraft build from the input camera | the build from 30°.

    uv run python tools/readme_figure.py <run id> [size] [out.jpg]

Draws each block's exposed faces (Minecraft face shading, painter's algorithm) with PIL.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import minecraft, store  # noqa: E402

# (axis, side, shade): Minecraft lights tops fully, north/south faces at 0.8, east/west at 0.6.
FACES = [(1, 1, 1.0), (1, 0, 0.5), (2, 0, 0.8), (2, 1, 0.8), (0, 0, 0.6), (0, 1, 0.6)]
WIDTH = 900  # px per panel


def corners(axis: int, side: int) -> np.ndarray:
    """The 4 corners (unit cube, block coordinates) of one face, in winding order."""
    a, b = [i for i in range(3) if i != axis]
    out = []
    for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
        c = np.zeros(3)
        c[axis], c[a], c[b] = side, u, v
        out.append(c)
    return np.array(out)


def render(b: minecraft.Build, K: np.ndarray, hw: tuple[int, int], yaw: float, bg=(17, 19, 23)) -> Image.Image:
    h, w = hw
    occupied = {tuple(p) for p in b.ijk}
    rgb = minecraft.BLOCK_RGB[b.block].astype(float)
    # Cloud centroid (camera frame) to orbit around, as the viewer's off-axis views do.
    cen_mc = b.lo + (b.ijk.mean(0) + 0.5) * b.block_size
    cen = minecraft.to_minecraft_axes(cen_mc)  # the axis flip is its own inverse
    t = np.deg2rad(yaw)
    R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    polys = []
    for axis, side, shade in FACES:
        step = np.zeros(3, int)
        step[axis] = 1 if side else -1
        exposed = np.array([tuple(p + step) not in occupied for p in b.ijk])
        quad = corners(axis, side)
        pts = (b.ijk[exposed, None, :] + quad[None]) * b.block_size + b.lo  # (n, 4, 3) Minecraft axes
        cam = minecraft.to_minecraft_axes(pts.reshape(-1, 3))  # back to the camera frame
        cam = (cam - cen) @ R.T + cen
        cam = cam.reshape(-1, 4, 3)
        # Back-face cull: a face is visible when its outward normal points back at the camera (origin).
        n = np.zeros(3)
        n[axis] = 1 if side else -1
        normal = minecraft.to_minecraft_axes(n) @ R.T
        facing = (cam.mean(1) @ normal) < 0
        z = cam[..., 2]
        ok = (z > 1e-6).all(1) & facing
        uv = cam[..., :2] / z[..., None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        col = rgb[exposed] * shade
        polys += zip(z[ok].mean(1), uv[ok], col[ok])
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    for _, q, c in sorted(polys, key=lambda p: -p[0]):  # far to near
        draw.polygon([tuple(v) for v in q], fill=tuple(int(x) for x in c))
    return img


def main(rid: str, size: int = 192, out: str = "docs/minecraft.jpg") -> None:
    a = store.load_arrays(rid)
    rgb = store._rgb(rid)
    b = minecraft.build(a["points"], rgb, a["mask"], a["intrinsics"], size)
    h, w = a["mask"].shape
    s = WIDTH / w
    K = a["intrinsics"].astype(float) * np.array([[s], [s], [1]])
    hw = (round(h * s), WIDTH)
    panels = [Image.fromarray(rgb).resize(hw[::-1], Image.LANCZOS), render(b, K, hw, 0), render(b, K, hw, 30)]
    gap = 12
    fig = Image.new("RGB", (WIDTH * 3 + gap * 2, hw[0]), (11, 12, 15))
    for i, p in enumerate(panels):
        fig.paste(p, (i * (WIDTH + gap), 0))
    fig.save(out, quality=88)
    print(f"{out}: {fig.size[0]}x{fig.size[1]}, {len(b.ijk):,} blocks")


if __name__ == "__main__":
    main(sys.argv[1], *(int(sys.argv[2]),) if len(sys.argv) > 2 else (), *sys.argv[3:4])
