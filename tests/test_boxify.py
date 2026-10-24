import numpy as np

from backend import boxify


def test_boxes_fill_prisms_seen_from_one_side_and_keep_props():
    # Street at y=0; three buildings (x0, x1, z0, z1, roof) seen only from the south (high z) and
    # from above, as one camera would; a 2x2 vent on the first roof.
    W, Ht, L = 60, 40, 60
    pts = [(x, 0, z, 1) for x in range(W) for z in range(L)]
    towers = [(5, 20, 5, 15, 30), (25, 35, 10, 30, 12), (40, 55, 35, 50, 20)]
    for x0, x1, z0, z1, h in towers:
        pts += [(x, h, z, 2) for x in range(x0, x1) for z in range(z0, z1)]  # roof
        pts += [(x, y, z1 - 1, 3) for x in range(x0, x1) for y in range(1, h)]  # south wall only
    pts += [(x, 31, z, 4) for x in (10, 11) for z in (8, 9)]  # vent
    a = np.array(pts)
    ijk, block = boxify.boxes(a[:, :3], a[:, 3], (W, Ht, L))
    occ = {tuple(p): b for p, b in zip(ijk.tolist(), block.tolist())}
    for x0, x1, z0, z1, h in towers:
        # The unseen north wall now exists, full height, painted like the building's seen wall.
        assert all((x, y, z0) in occ for x in range(x0, x1) for y in range(1, h))
        assert all(occ[(x, h, z)] == 2 for x in range(x0, x1) for z in range(z0, z1))  # roof kept
        assert (x0 + 5, h // 2, z0 + 3) not in occ  # hollow
        assert (x0 - 1, h, z0) not in occ  # footprint not grown
    assert all(occ.get((x, 31, z)) == 4 for x in (10, 11) for z in (8, 9))  # vent kept as is
    assert occ[(0, 0, 0)] == 1  # street untouched
