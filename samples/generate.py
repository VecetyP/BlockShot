"""Draws the synthetic samples: uv run python samples/generate.py"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).parent
SIZE = 768


def penrose_strips(w: float = 0.3, size: int = SIZE) -> list[list[tuple[float, float]]]:
    """The three face strips of a Penrose tribar, as pixel-coordinate polygons in a size x size image."""
    # Edges 0=bottom, 1=right, 2=left of an upward triangle centered at the origin with inradius 1.
    # Each edge has three parallel lines at depth 0 (outer), w (mid), 2w (inner). Every vertex of
    # the figure is the intersection of two such lines; each face strip joins one bar's outer band
    # to the next bar's inner band, which is what makes the figure impossible.
    normals = [np.array([np.cos(a), np.sin(a)]) for a in np.deg2rad([90, 210, 330])]  # inward

    def line(k, d):  # {x : n·x = d - 1}
        return normals[k % 3], d - 1.0

    def meet(l1, l2):
        return np.linalg.solve(np.stack([l1[0], l2[0]]), [l1[1], l2[1]])

    O, M, I = 0.0, w, 2 * w
    strips = []
    for k in range(3):
        b, r, l = k, k + 1, k + 2
        strips.append([
            meet(line(b, O), line(l, M)),
            meet(line(b, O), line(r, O)),
            meet(line(r, O), line(b, M)),
            meet(line(b, M), line(l, I)),
            meet(line(l, I), line(r, M)),
            meet(line(l, M), line(r, M)),
        ])

    s = size * 0.22
    return [[(size / 2 + p[0] * s, size * 0.58 - p[1] * s) for p in poly] for poly in strips]


def penrose_triangle(path: Path) -> None:
    img = Image.new("RGB", (SIZE, SIZE), "white")
    dr = ImageDraw.Draw(img)
    for poly, shade in zip(penrose_strips(), [(90, 90, 90), (170, 170, 170), (230, 230, 230)]):
        dr.polygon(poly, fill=shade, outline="black", width=3)
    img.save(path)


def necker_cube(path: Path) -> None:
    img = Image.new("RGB", (SIZE, SIZE), "white")
    dr = ImageDraw.Draw(img)
    a, off = SIZE * 0.4, SIZE * 0.17
    front = np.array([[0, 0], [a, 0], [a, a], [0, a]]) + [(SIZE - a - off) / 2, (SIZE - a + off) / 2]
    back = front + [off, -off]  # shifted up-right; either face can read as the front
    for face in (front, back):
        dr.polygon([tuple(p) for p in face], outline="black", width=5)
    for p, q in zip(front, back):
        dr.line([tuple(p), tuple(q)], fill="black", width=5)
    img.save(path)


if __name__ == "__main__":
    for draw, name in ((penrose_triangle, "penrose_triangle.png"), (necker_cube, "necker_cube.png")):
        if not (OUT / name).exists():  # never overwrite a sample someone dropped in
            draw(OUT / name)
