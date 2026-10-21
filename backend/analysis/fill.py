"""Occlusion gaps and a geometric fill for them.

At an occluding edge (a depth jump between neighbouring pixels) the near surface hides part of the
far one. The hidden part lies behind the near-side pixels close to the edge: the "occluders".
Moving the camera exposes it, and with only one image there is nothing there: a gap.

The fill continues the far surface behind the occluders, assuming it keeps its local behaviour
(the way a person assumes a bowl keeps curving behind its rim). It works in inverse depth, where
a plane is exactly linear in pixel coordinates, and minimises a smoothness energy with the visible
far surface held fixed:
  order 2 ("slope")      min ||L d||²    biharmonic: continues the tilt; planes are reproduced exactly
  order 3 ("curvature")  min ||G L d||²  triharmonic: also continues the bending; noisier
Colours are diffused in from the same visible pixels (order 1, harmonic).
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import binary_dilation, label, maximum_filter, uniform_filter

EDGE_JUMP = 0.05  # neighbours whose depths differ by more than 5% form an occluding edge
RING = 2  # px of visible far surface held fixed around the fill (order 3 needs at least 2)
FAR_SLACK = 0.15  # hidden surface lies between its occluder and ~the depth where the far surface was last seen
SOLID_NEIGHBOURS = 6  # of 9 in a 3x3 window: the far side of an edge must be this solid


@dataclass
class Occlusion:
    occluder: np.ndarray  # (H, W) bool: near-side pixels with hidden far surface behind them
    far_depth: np.ndarray  # (H, W) depth of the nearest far-side edge pixel (inf where none)
    far_edge: np.ndarray  # (H, W) bool: far-side pixels of occluding edges


def _solid(valid: np.ndarray) -> np.ndarray:
    """Pixels inside a real surface, not stray points (e.g. leftovers around a masked background)."""
    return valid & (uniform_filter(valid.astype(np.float32), size=3) * 9 >= SOLID_NEIGHBOURS - 0.5)


_PAIRS = (  # (a, b) slices of horizontally and vertically adjacent pixels
    ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
)


def occlusion(depth: np.ndarray, mask: np.ndarray, reach: float) -> Occlusion:
    """Pixels on the near side of an occluding edge, up to `reach` px into that near surface.

    Grown outward from each edge across the near surface only (neighbours without a depth jump), and
    each grown pixel carries the depth of the far surface it hides. So a slanted wall that is merely
    nearer than some distant edge is not mistaken for an occluder."""
    h, w = mask.shape
    valid = mask & (depth > 0)
    z = np.where(valid, depth, np.nan)
    zfar = np.where(_solid(valid), z, np.nan)
    far = np.full((h, w), np.nan, np.float32)  # depth of the hidden far surface, NaN = not an occluder
    far_edge = np.zeros((h, w), bool)
    links = []
    with np.errstate(invalid="ignore"):
        for a, b in _PAIRS:
            za, zb = z[a], z[b]
            step_b, step_a = zb > za * (1 + EDGE_JUMP), za > zb * (1 + EDGE_JUMP)  # which side is farther
            jump_b, jump_a = step_b & ~np.isnan(zfar[b]), step_a & ~np.isnan(zfar[a])  # ...and solid
            far_edge[b] |= jump_b
            far_edge[a] |= jump_a
            far[a] = np.fmax(far[a], np.where(jump_b, zb, np.nan))
            far[b] = np.fmax(far[b], np.where(jump_a, za, np.nan))
            links.append(~step_a & ~step_b & ~np.isnan(za) & ~np.isnan(zb))  # same surface
    # Geodesic growth across the near surface, one pixel per step.
    for _ in range(int(reach)):
        grown = far.copy()
        for (a, b), link in zip(_PAIRS, links):
            grown[b] = np.where(np.isnan(grown[b]) & link, far[a], grown[b])
            grown[a] = np.where(np.isnan(grown[a]) & link, far[b], grown[a])
        if np.array_equal(np.isnan(grown), np.isnan(far)):
            break
        far = grown
    with np.errstate(invalid="ignore"):
        occluder = mask & ~far_edge & (depth * (1 + EDGE_JUMP) < far)
    return Occlusion(occluder, np.where(occluder, far, np.inf).astype(np.float32), far_edge)


def _operator(nodes: np.ndarray, order: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """Smoothness operator over the 4-connected graph on `nodes`. Returns (D, node index image)."""
    h, w = nodes.shape
    idx = np.full((h, w), -1)
    idx[nodes] = np.arange(nodes.sum())
    ex, ey = nodes[:, :-1] & nodes[:, 1:], nodes[:-1] & nodes[1:]
    ix, jx = np.nonzero(ex)
    iy, jy = np.nonzero(ey)
    a = np.r_[idx[ix, jx], idx[iy, jy]]
    b = np.r_[idx[ix, jx + 1], idx[iy + 1, jy]]
    m, n = len(a), int(nodes.sum())
    rows = np.arange(m)
    G = sp.csr_matrix((np.r_[-np.ones(m), np.ones(m)], (np.r_[rows, rows], np.r_[a, b])), shape=(m, n))
    if order == 1:
        return G, idx
    L = (G.T @ G).tocsr()  # graph Laplacian
    # Where the region ends, a Laplacian row is truncated and no longer vanishes on a plane, which
    # would bend the fill. Those rows get a tiny weight: they only settle what the full rows leave open.
    full = np.asarray(np.abs(L).sum(1)).ravel() == 8  # all 4 neighbours present
    if order == 2:
        return sp.diags(np.where(full, 1.0, 1e-3)) @ L, idx
    GL = (G @ L).tocsr()
    return sp.diags(np.where(full[a] & full[b], 1.0, 1e-3)) @ GL, idx


def _solve(D: sp.csr_matrix, unknown: np.ndarray, known_values: np.ndarray) -> np.ndarray:
    """min ||D x||² over x[unknown] with the other entries fixed. known_values: (n_known, C)."""
    A, B = D[:, unknown], D[:, ~unknown]
    M = (A.T @ A + sp.identity(A.shape[1]) * 1e-9).tocsc()
    rhs = -(A.T @ (B @ known_values))
    lu = spla.splu(M)
    return np.stack([lu.solve(rhs[:, c]) for c in range(rhs.shape[1])], -1)


@dataclass
class Fill:
    rows: np.ndarray  # pixel coordinates of filled samples
    cols: np.ndarray
    depth: np.ndarray  # depth of the hidden surface behind each occluder pixel
    rgb: np.ndarray  # (N, 3) uint8


def fill(depth: np.ndarray, rgb: np.ndarray, mask: np.ndarray, reach: float, order: int = 2) -> Fill:
    occ = occlusion(depth, mask, reach)
    empty = Fill(np.zeros(0, int), np.zeros(0, int), np.zeros(0, np.float32), np.zeros((0, 3), np.uint8))
    if not occ.occluder.any():
        return empty
    # Fixed boundary: pixels beside the fill region that are clearly behind it (the visible far surface),
    # not more of the near surface beyond the reach.
    near = binary_dilation(occ.occluder, iterations=RING) & ~occ.occluder & mask & (depth > 0)
    front = maximum_filter(np.where(occ.occluder, depth, 0), size=2 * RING + 1)
    ring = near & (depth > front * (1 + EDGE_JUMP)) & _solid(mask & (depth > 0))
    # Only solve fill regions that touch the far surface (anything else has nothing to continue).
    lab, _ = label(occ.occluder)
    keep = np.unique(lab[binary_dilation(ring) & occ.occluder])
    hole = np.isin(lab, keep[keep > 0])
    if not hole.any():
        return empty
    nodes = hole | ring
    unknown_img = hole[nodes]

    D, _ = _operator(nodes, order)
    known = ~unknown_img  # node order is row-major, same as depth[nodes]
    d = _solve(D, unknown_img, (1.0 / depth[nodes][known]).astype(np.float64)[:, None])[:, 0]

    G1, _ = _operator(nodes, 1)
    col = _solve(G1, unknown_img, rgb[nodes][known].astype(np.float64))

    rows, cols = np.nonzero(nodes)
    rows, cols = rows[unknown_img], cols[unknown_img]
    with np.errstate(divide="ignore"):
        z = np.where(d > 0, 1.0 / d, np.inf)
    # A hidden surface lies behind its occluder, and near where the surface it continues was last seen:
    # extrapolation that comes forward is pushed back, and a slope that runs away is stopped.
    z = np.clip(z, depth[rows, cols] * (1 + EDGE_JUMP / 2), occ.far_depth[rows, cols] * (1 + FAR_SLACK))
    ok = np.isfinite(z)
    return Fill(rows[ok], cols[ok], z[ok].astype(np.float32), np.clip(col[ok], 0, 255).astype(np.uint8))
