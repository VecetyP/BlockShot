"""Integrability of a run's local geometry.

Works on g = ∇ log z per pixel (unit: 1/px). For a perspective camera a surface with normal n at pixel
ray r = ((u-cx)/fx, (v-cy)/fy, 1) satisfies

    ∂ log z/∂x = -n_x / (n·r),   ∂ log z/∂y = -n_y / (n·r)        (x, y normalized image coords)

which is exactly curl-free for any real surface. Near the optical axis n·r ≈ n_z and this is the
orthographic p = -n_x/n_z, q = -n_y/n_z. Using log z makes every number scale-invariant.

Gradients live on grid edges, so the discrete operators line up: for depth-derived gradients the
residual is zero to solver precision (the sanity check), while native normals can carry a
non-integrable part. The Helmholtz–Hodge split solves  min_z ||G z - g||²  over the mask, i.e. the
Poisson equation  Gᵀ G z = Gᵀ g  with natural (Neumann) boundaries, and the residual g - G z is what
no depth map could produce.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

GRAZING = 0.05  # |n·r̂| below this is a silhouette/grazing pixel; its gradient blows up, so mask it


@dataclass
class EdgeField:
    gx: np.ndarray  # (H, W-1) d/du between (i, j) and (i, j+1)
    gy: np.ndarray  # (H-1, W) d/dv between (i, j) and (i+1, j)
    ex: np.ndarray  # (H, W-1) bool, valid x edges
    ey: np.ndarray  # (H-1, W) bool, valid y edges


@dataclass
class Integrability:
    residual_fraction: float  # ||residual||² / ||g||² over valid edges
    curl: np.ndarray  # (H, W) circulation per plaquette (1/px), 0 where undefined
    residual: np.ndarray  # (H, W) per-pixel RMS residual of incident edges (1/px)
    mask: np.ndarray  # (H, W) pixels that took part
    n_edges: int
    grad_rms: float  # RMS of g over valid edges (1/px); curl and residual are best read relative to it


def _rays(h: int, w: int, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]


def from_normals(normals: np.ndarray, K: np.ndarray, mask: np.ndarray) -> EdgeField:
    h, w = mask.shape
    x, y = _rays(h, w, K)
    nr = normals[..., 0] * x + normals[..., 1] * y + normals[..., 2]
    ok = mask & (np.abs(nr) > GRAZING * np.sqrt(x * x + y * y + 1))
    nr = np.where(ok, nr, 1.0)
    p = np.where(ok, -normals[..., 0] / nr / K[0, 0], 0)  # per pixel: d/du = (1/fx) d/dx
    q = np.where(ok, -normals[..., 1] / nr / K[1, 1], 0)
    ex, ey = ok[:, :-1] & ok[:, 1:], ok[:-1] & ok[1:]
    # Trapezoid rule onto the edges between pixel centers.
    return EdgeField((p[:, :-1] + p[:, 1:]) / 2, (q[:-1] + q[1:]) / 2, ex, ey)


def from_depth(depth: np.ndarray, mask: np.ndarray) -> EdgeField:
    ok = mask & (depth > 0)
    lz = np.log(np.where(ok, depth, 1.0))
    ex, ey = ok[:, :-1] & ok[:, 1:], ok[:-1] & ok[1:]
    return EdgeField(np.diff(lz, axis=1), np.diff(lz, axis=0), ex, ey)


def analyze(f: EdgeField) -> Integrability:
    h, w = f.ey.shape[0] + 1, f.ex.shape[1] + 1
    px = np.zeros((h, w), bool)
    px[:, :-1] |= f.ex
    px[:, 1:] |= f.ex
    px[:-1] |= f.ey
    px[1:] |= f.ey
    idx = np.full((h, w), -1)
    idx[px] = np.arange(px.sum())

    ix, jx = np.nonzero(f.ex)
    iy, jy = np.nonzero(f.ey)
    a = np.concatenate([idx[ix, jx], idx[iy, jy]])  # edge tail
    b = np.concatenate([idx[ix, jx + 1], idx[iy + 1, jy]])  # edge head
    g = np.concatenate([f.gx[ix, jx], f.gy[iy, jy]])
    m, n = len(g), int(px.sum())

    curl = np.zeros((h, w), np.float32)
    plaq = f.ex[:-1] & f.ex[1:] & f.ey[:, :-1] & f.ey[:, 1:]
    c = f.gx[:-1] + f.gy[:, 1:] - f.gx[1:] - f.gy[:, :-1]
    curl[:-1, :-1] = np.where(plaq, c, 0)

    if m == 0:
        return Integrability(0.0, curl, np.zeros((h, w), np.float32), px, 0, 0.0)

    rows = np.arange(m)
    G = sp.csr_matrix((np.r_[-np.ones(m), np.ones(m)], (np.r_[rows, rows], np.r_[a, b])), shape=(m, n))
    # Tiny Tikhonov term pins the free constant of each connected component.
    L = (G.T @ G + sp.identity(n) * 1e-9).tocsc()
    z = spla.spsolve(L, G.T @ g)
    r = g - G @ z

    # Per-pixel RMS of the residual over the edges touching it.
    acc, cnt = np.zeros(n), np.zeros(n)
    np.add.at(acc, a, r * r)
    np.add.at(acc, b, r * r)
    np.add.at(cnt, a, 1)
    np.add.at(cnt, b, 1)
    res = np.zeros((h, w), np.float32)
    res[px] = np.sqrt(acc / np.maximum(cnt, 1))

    total = float(g @ g)
    return Integrability(float(r @ r) / total if total > 0 else 0.0, curl, res, px, m, float(np.sqrt(total / m)))
