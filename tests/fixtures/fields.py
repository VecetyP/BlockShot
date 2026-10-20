"""Analytic normal fields for the integrability tests (OpenCV camera frame)."""
import numpy as np
from PIL import Image, ImageDraw

from samples.generate import penrose_strips


def sphere(n: int = 256, f: float = 300.0, center=(0.2, -0.1, 5.0), radius: float = 1.5):
    """Exact normals of a sphere seen by a perspective camera."""
    K = np.array([[f, 0, n / 2], [0, f, n / 2], [0, 0, 1]], np.float32)
    u, v = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    d = np.stack([(u - n / 2) / f, (v - n / 2) / f, np.ones_like(u)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    c = np.array(center)
    b = d @ c
    disc = b * b - (c @ c - radius**2)
    mask = disc > 0
    t = b - np.sqrt(np.maximum(disc, 0))  # near intersection
    normals = (d * t[..., None] - c) / radius
    return normals.astype(np.float32), K, mask


def penrose(n: int = 256):
    """Hand-built normal field of the Penrose tribar.

    Each of the three face strips is planar with an isometric cube-face gradient (|∇z| = √2, 120°
    apart). The directions are the assignment for which every join between strips is a valid fold
    edge, so the field is (almost) curl-free locally; it still cannot be integrated around the loop.
    A long focal length makes the camera effectively orthographic.
    """
    lab = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(lab)
    for k, poly in enumerate(penrose_strips(size=n)):
        draw.polygon(poly, fill=k + 1)
    lab = np.asarray(lab)
    normals = np.zeros((n, n, 3), np.float32)
    normals[..., 2] = -1
    for k, slot in enumerate((2, 1, 0)):
        a = np.deg2rad(120 * slot)
        v = np.array([-np.sqrt(2) * np.cos(a), -np.sqrt(2) * np.sin(a), -1.0])
        normals[lab == k + 1] = v / np.linalg.norm(v)
    f = 1e4
    K = np.array([[f, 0, n / 2], [0, f, n / 2], [0, 0, 1]], np.float32)
    return normals, K, lab > 0, lab
