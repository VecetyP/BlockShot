"""Blocks -> box city: buildings as solid prisms, for builds whose grid runs along the walls.

A reconstruction only sees the near side of things, so a build's walls are one block thick and its
back walls missing. Most built things are boxes standing on the ground, so this works on the top-
down height map: small things on roofs (vents, railings) are opened away, holes in footprints are
filled, and every building column is extruded from the ground to its roof as a hollow
shell. Shell blocks take the colour of the nearest block that was actually seen, so windows and
signs survive; everything outside the prisms (streets, props on roofs) stays as it was.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

PROP = 5  # blocks: things narrower than this on a roof are props, not part of the building
MIN_HEIGHT = 3  # blocks above the ground before a column counts as building
PAINT_REACH = 2.0  # blocks: a shell block takes the colour of a seen block this close


def roofs(H: np.ndarray, ground: int, k: int = PROP) -> np.ndarray:
    """Roof height per column from the height map H (x, z; -1 = nothing seen), -1 off buildings."""
    seen = H >= 0
    R = ndimage.grey_opening(np.where(seen, H, ground), size=(k, k))  # drop props narrower than k
    # No closing: it would bridge the narrow gaps between buildings. Only enclosed holes are filled.
    src = R > ground + MIN_HEIGHT
    if not src.any():
        return np.full_like(H, -1)
    on = ndimage.binary_fill_holes(src)
    # Filled columns (nothing seen on them) take the nearest building column's roof.
    _, (ix, iz) = ndimage.distance_transform_edt(~src, return_indices=True)
    return np.where(on, R[ix, iz], -1)


def boxes(ijk: np.ndarray, block: np.ndarray, dims, k: int = PROP) -> tuple[np.ndarray, np.ndarray]:
    """(ijk, block) of a build (x east, y up, z south) -> the same with buildings as prisms."""
    if len(ijk) == 0:
        return ijk, block
    W, _, L = dims
    H = np.full((W, L), -1, np.int64)
    np.maximum.at(H, (ijk[:, 0], ijk[:, 2]), ijk[:, 1])
    ground = int(np.percentile(H[H >= 0], 5))
    R = roofs(H, ground, k)

    # Hollow shell: roof blocks, plus wall blocks wherever a side neighbour is lower.
    lower = np.full((W + 2, L + 2), -1, np.int64)
    lower[1:-1, 1:-1] = R
    nb = np.minimum.reduce([lower[:-2, 1:-1], lower[2:, 1:-1], lower[1:-1, :-2], lower[1:-1, 2:]])
    xs, zs = np.nonzero(R >= 0)
    top, low = R[xs, zs], np.maximum(nb[xs, zs], ground - 1)
    # Per column: walls from above the lowest neighbour's roof (or the ground) up to the roof.
    n = np.maximum(top - low, 1)  # inner columns: just the roof block
    x = np.repeat(xs, n)
    z = np.repeat(zs, n)
    y = np.repeat(top, n) - (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    shell = np.stack([x, y, z], 1)

    # Seen blocks inside the prisms are replaced by the shell; the rest stay (props, streets).
    r_at = R[ijk[:, 0], ijk[:, 2]]
    keep = ijk[:, 1] > r_at
    seen = cKDTree(ijk)
    d, i = seen.query(shell, distance_upper_bound=PAINT_REACH)
    hit = np.isfinite(d)
    paint = np.empty(len(shell), block.dtype)
    paint[hit] = block[i[hit]]
    if (~hit).any():  # nothing seen nearby (back walls): the most common colour of that building
        lab, _ = ndimage.label(R >= 0)
        b_lab = lab[ijk[:, 0], ijk[:, 2]]
        inside = (b_lab > 0) & ~keep
        common = np.full(lab.max() + 1, np.bincount(block).argmax(), block.dtype)
        for b in np.unique(b_lab[inside]):
            common[b] = np.bincount(block[inside & (b_lab == b)]).argmax()
        paint[~hit] = common[lab[shell[~hit, 0], shell[~hit, 2]]]
    out = np.concatenate([ijk[keep], shell.astype(ijk.dtype)])
    return out, np.concatenate([block[keep], paint])
