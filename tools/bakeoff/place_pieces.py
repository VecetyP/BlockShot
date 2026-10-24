"""Place a World's pieces (split_world.py) from the feature matches they share, instead of roof matching.

COLMAP's database holds verified feature matches between frames, including between frames of
different pieces (sequential matching across each break, plus cross_pairs.txt matched with
pycolmap.match_image_pairs). Each match is lifted to 3D with each piece's own depth and camera, so a
pair of pieces gives 3D-3D correspondences; a robust similarity fit (scale, rotation, shift) per pair
says how one piece sits in the other. The pieces are then placed from piece 1 along the strongest links
(most inliers): each child is fitted into the world with a quarter turn, a scale and an offset, the
form World placements take (pieces are levelled and on a grid along their walls).

Checks printed: per link, the inliers and how far its rotation is from the quarter turn used; the start
and end cameras' distance (Test5 starts and ends in the middle of the roof); a loop check where pieces
are linked by more than the tree.

    uv run python tools/bakeoff/place_pieces.py <sfm dir> <world name>
"""
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend import store, video, world  # noqa: E402

PAIR = 2147483647
MIN_LINK = 30  # inlier 3D matches for a link between pieces to count
RANSAC_ITERS = 400
RES_MAX = 3.0  # blocks: a placement fitting its link worse than this is left unconfirmed
SCALE_PRIOR = 2.0  # a link may size a piece at most this far from its own eye-height size (those agree ~20%)
REL_MAX = 0.25  # a link's residual may be at most this share of its matched points' spread


def db_matches(db: Path):
    con = sqlite3.connect(str(db))
    names = dict(con.execute("select image_id, name from images"))
    kp = {i: np.frombuffer(d, np.float32).reshape(r, c)[:, :2] for i, r, c, d in
          con.execute("select image_id, rows, cols, data from keypoints") if d}
    out = []
    for pid, r, c, d in con.execute("select pair_id, rows, cols, data from two_view_geometries where rows > 0"):
        i2 = pid % PAIR
        i1 = (pid - i2) // PAIR
        m = np.frombuffer(d, np.uint32).reshape(r, c)
        out.append((names[i1], names[i2], kp[i1][m[:, 0]], kp[i2][m[:, 1]]))
    return out


def lift(sc, k, xy):
    """Pixels of scene frame k -> 3D points in the scene's frame (nan where no depth)."""
    d = sc["depth"][k]
    h, w = d.shape
    u = np.clip(xy[:, 0].astype(int), 0, w - 1)
    v = np.clip(xy[:, 1].astype(int), 0, h - 1)
    z = d[v, u].astype(np.float64)
    K = sc["K"][k].astype(np.float64)
    P = np.stack([(xy[:, 0] - K[0, 2]) / K[0, 0] * z, (xy[:, 1] - K[1, 2]) / K[1, 1] * z, z], 1)
    c2w = sc["c2w"][k].astype(np.float64)
    P = P @ c2w[:3, :3].T + c2w[:3, 3]
    P[z <= 0] = np.nan
    return P


def sim3_ransac(X, Y, rng):
    """Robust s, R, t with X ~ s R Y + t. Inlier threshold: 10% of the median distance of X from its centre."""
    thr = 0.1 * np.median(np.linalg.norm(X - np.median(X, 0), axis=1))
    best = None
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(len(X), 4, replace=False)
        try:
            s, R, t = video.umeyama(Y[idx], X[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.linalg.norm(X - (s * Y @ R.T + t), axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    s, R, t = video.umeyama(Y[best], X[best])
    inl = np.linalg.norm(X - (s * Y @ R.T + t), axis=1) < thr
    s, R, t = video.umeyama(Y[inl], X[inl])
    return s, R, t, inl


def fit_child(f, parent_is_p, ps, pax, cax):
    """Place the child of link f in the world given its placed parent: yaw about "up" (any angle),
    scale and offset fitted to the inlier matches. -> (median residual in blocks, turns, yaw, scale, offset, n)."""
    s, R, t, inl, A, B = f
    if parent_is_p:  # A in parent, B in child: parent = s R child + t
        y_child, x_parent = B[inl], s * B[inl] @ R.T + t
    else:  # A in child: child = s R parent + t
        y_child, x_parent = A[inl], (A[inl] - t) @ R / s
    target = to_world(ps, x_parent, *pax)
    y = (y_child - cax[0]) @ cax[1]
    a = y[:, [0, 2]] - y[:, [0, 2]].mean(0)
    b = target[:, [0, 2]] - target[:, [0, 2]].mean(0)
    theta = math.atan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    phi = -math.degrees(theta)  # world.rot turns (x, z) the other way round
    turns = int(round(phi / 90)) % 4
    yaw = (phi - 90 * turns + 180) % 360 - 180
    ry = y @ world.rot(turns, yaw).T
    yc, tc = ry - ry.mean(0), target - target.mean(0)
    sc = float((yc * tc).sum() / (yc * yc).sum())
    off = target.mean(0) - sc * ry.mean(0)
    res = float(np.median(np.linalg.norm(sc * ry + off - target, axis=1)))
    rel = res / float(np.median(np.linalg.norm(tc, axis=1)) + 1e-9)  # relative to the matched points' spread
    return res, rel, turns, float(yaw), sc, off, int(inl.sum())


def axes_of(rid):
    _, pose, _, _, B = world._axes(rid)
    return pose.position, B


def to_world(s, x, pos, B):
    return (x - pos) @ B * s["scale"] @ world.srot(s).T + np.asarray(s["offset"], float)


def main():
    d, wname = Path(sys.argv[1]), sys.argv[2]
    cut = [tuple(c) for c in json.loads((d / "pieces.json").read_text())["pieces"]]
    w = world.load(wname)
    rids = [s["scene"] for s in w["sessions"]]
    assert len(rids) == len(cut), "world sessions and pieces differ"
    piece_of = {}
    for p, (a, b) in enumerate(cut):
        for i in range(a, b):
            piece_of[f"{i:04d}.jpg"] = (p, i - a)
    scenes = [store.load_scene(r) | {} for r in rids]  # load_scene caches two; keep our own refs
    scenes = [{k: store.load_scene(r)[k] for k in ("depth", "K", "c2w")} for r in rids]
    links = {}
    for n1, n2, x1, x2 in db_matches(d / "db.db"):
        if n1 not in piece_of or n2 not in piece_of:
            continue
        (p, k), (q, l) = piece_of[n1], piece_of[n2]
        if p == q:
            continue
        A, B = lift(scenes[p], k, x1), lift(scenes[q], l, x2)
        ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        if p > q:
            p, q, A, B = q, p, B, A
        links.setdefault((p, q), [[], []])
        links[(p, q)][0].append(A[ok])
        links[(p, q)][1].append(B[ok])
    rng = np.random.default_rng(0)
    fits = {}
    for (p, q), (A, B) in sorted(links.items()):
        A, B = np.concatenate(A), np.concatenate(B)
        if len(A) < MIN_LINK:
            continue
        s, R, t, inl = sim3_ransac(A, B, rng)
        fits[(p, q)] = (s, R, t, inl, A, B)
        print(f"link {p + 1}-{q + 1}: {len(A)} matches, {inl.sum()} inliers, scale {s:.4g}", flush=True)
    # Tree from piece 1: repeatedly place the unplaced piece whose link to a placed one fits best.
    axes = [axes_of(r) for r in rids]
    placed = {0: w["sessions"][0]}
    while len(placed) < len(rids):
        cands = []
        for (p, q), f in fits.items():
            if f[3].sum() < MIN_LINK or (p in placed) == (q in placed):
                continue
            parent, child = (p, q) if p in placed else (q, p)
            res, rel, turns, yaw, sc, off, n = fit_child(f, parent == p, placed[parent], axes[parent], axes[child])
            eye = w["sessions"][child]["scale"]
            if not (1 / SCALE_PRIOR <= sc / eye <= SCALE_PRIOR) or rel > REL_MAX:
                continue  # collapsed or blown-up scale, or a loose fit: not a believable link
            cands.append((parent, child, res, rel, turns, yaw, sc, off, n))
        if not cands:
            missing = [i + 1 for i in range(len(rids)) if i not in placed]
            print("not linked to the rest:", missing)
            for i in missing:
                placed[i - 1] = {**w["sessions"][i - 1], "confirmed": False}
            break
        parent, child, res, rel, turns, yaw, sc, off, n = max(cands, key=lambda c: c[-1])
        placed[child] = {**w["sessions"][child], "turns": turns, "yaw": yaw, "scale": sc, "offset": off.tolist(),
                         "confirmed": res <= RES_MAX, "score": None, "placed_by": f"matches with piece {parent + 1} ({n} inliers)"}
        print(f"piece {child + 1} from piece {parent + 1}: {n} inliers, yaw {90 * turns + yaw:.1f} deg, scale {sc:.3f} "
              f"(eye-height scale was {w['sessions'][child]['scale']:.3f}), median residual {res:.2f} blocks ({100 * rel:.0f}% of spread)"
              f"{'' if res <= RES_MAX else ' -> left unconfirmed'}", flush=True)
    w["sessions"] = [placed[i] for i in range(len(rids))]
    world.save(w)
    # Start and end: first camera of piece 1, last camera of the last piece.
    cams = [(store.load_scene(r)["c2w"][:, :3, 3].astype(np.float64)) for r in rids]
    start = to_world(w["sessions"][0], cams[0][:1], *axes[0])[0]
    end = to_world(w["sessions"][-1], cams[-1][-1:], *axes[-1])[0]
    path = np.concatenate([to_world(w["sessions"][i], cams[i], *axes[i]) for i in range(len(rids))])
    span = np.linalg.norm(path.max(0) - path.min(0))
    print(f"start-end camera distance {np.linalg.norm(start - end):.1f} blocks (whole path spans {span:.1f} blocks)")
    # Loop check: links not used by the tree, how far apart they put matched points.
    for (p, q), (s, R, t, inl, A, B) in fits.items():
        if inl.sum() < MIN_LINK:
            continue
        a = to_world(w["sessions"][p], A[inl], *axes[p])
        b = to_world(w["sessions"][q], B[inl], *axes[q])
        print(f"  link {p + 1}-{q + 1}: matched points {np.median(np.linalg.norm(a - b, axis=1)):.1f} blocks apart in the world (median)")


if __name__ == "__main__":
    main()
