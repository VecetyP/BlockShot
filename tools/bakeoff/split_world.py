"""A COLMAP model that breaks partway (scale jumps, cameras teleport) -> consistent pieces -> a World.

Breaks: per frame, COLMAP's scale relative to MoGe's (median z_colmap / z_moge at its points). MoGe
is roughly the same scale in every frame, so within a consistent stretch that ratio stays put; where
COLMAP's model breaks it jumps (Test5: ~0.01 for 0-28 s, ~1e-4 for 29-42 s, 0.2-0.6 for 43-63 s).
A camera step far above its neighbours' also breaks. Pieces shorter than MIN_PIECE frames are dropped.

Each piece: MoGe depth fitted per frame to COLMAP's points (fuse_scene.fitted_depth), pixels kept
where the piece's TSDF average agrees (|distance| < half a truncation, i.e. consensus, not a cut),
levelled (path plane when the path covers an area, else floors/walls), saved as a scene run, and added
to a World in order: the first at the origin, the rest placed by roof matching (world.add_scene) and
confirmed when the match scores at least AUTO_CONFIRM.

    uv run python tools/bakeoff/split_world.py <sfm dir> <world name> <source name>
"""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import colmap_text  # noqa: E402
from fuse_scene import HUD, fitted_depth  # noqa: E402
from tsdf_fuse import MAX_VOXELS, tsdf  # noqa: E402
from backend import minecraft, store, video, world  # noqa: E402

SCALE_JUMP = 2.5  # a frame's COLMAP/MoGe scale this far from the running median starts a new piece
STEP_JUMP = 20  # ...or a camera step this many times the running median step
WINDOW = 8  # frames in the running medians
MIN_PIECE = 12  # frames (3 s at 4 fps)
MIN_KEPT = 20_000  # pixels that must survive the consensus for a piece to be used
AUTO_CONFIRM = 0.5  # roof-match score above which a placed piece is confirmed
CONSENSUS = 0.5  # keep pixels within this share of the truncation distance of the averaged surface


def frame_scales(anchors, D, size):
    W, H = size
    h, w = D.shape[1:]
    out = []
    for i, a in enumerate(anchors):
        u = np.clip((a[:, 0] * w / W).astype(int), 0, w - 1)
        v = np.clip((a[:, 1] * h / H).astype(int), 0, h - 1)
        dd = D[i][v, u]
        ok = (dd > 0) & (a[:, 2] > 0) & (a[:, 1] > HUD)
        out.append(np.median(a[ok, 2] / dd[ok]) if ok.sum() > 20 else np.nan)
    return np.array(out)


def pieces(scale, C):
    """[(start, end)] of consecutive frames with a steady scale and no camera teleport."""
    ls = np.log(np.where(np.isfinite(scale) & (scale > 0), scale, np.nan))
    step = np.r_[0, np.linalg.norm(np.diff(C, axis=0), axis=1)]
    cuts, start = [], 0
    for i in range(1, len(C)):
        a = max(start, i - WINDOW)
        ref_s = np.nanmedian(ls[a:i]) if np.isfinite(ls[a:i]).any() else np.nan
        ref_t = np.median(step[a + 1:i]) if i - a > 1 else np.inf
        jump = (np.isfinite(ls[i]) and np.isfinite(ref_s) and abs(ls[i] - ref_s) > np.log(SCALE_JUMP)) or \
               (step[i] > STEP_JUMP * max(ref_t, 1e-12) and i - a > 2)
        if jump:
            cuts.append((start, i))
            start = i
    cuts.append((start, len(C)))
    return [(a, b) for a, b in cuts if b - a >= MIN_PIECE]


def level(c2w, depth, mask, K):
    """Rotation making the piece level: the path plane when the path covers an area, else floors/walls."""
    C = c2w[:, :3, 3]
    prior = -c2w[:, :3, 1].mean(0)
    prior /= np.linalg.norm(prior)
    _, sv, vt = np.linalg.svd(C - C.mean(0))
    if sv[0] > 0 and sv[1] > 0.2 * sv[0] and sv[1] > 1e-6:
        up = vt[2] * np.sign(vt[2] @ prior)
        how = "path plane"
    else:
        N = np.concatenate([minecraft.grid_normals(video.world_points(depth[i], K, c2w[i]), mask[i]) for i in range(len(depth))])
        up, _ = minecraft.find_up(N, prior, max_angle=60.0)
        how = "floors/walls"
    return minecraft.rotation_between(up, np.array([0.0, -1.0, 0.0])), how


def consensus(depth, mask, rgb, K, c2w, anchors):
    z = np.concatenate([a[:, 2] for a in anchors])
    zmed = float(np.median(z[z > 0]))
    pts = []
    v, u = np.mgrid[0:depth.shape[1]:4, 0:depth.shape[2]:4]
    for i in range(len(depth)):
        dd = depth[i][v, u]
        m = dd > 0
        P = np.stack([(u[m] - K[0, 2]) / K[0, 0] * dd[m], (v[m] - K[1, 2]) / K[1, 1] * dd[m], dd[m]], 1)
        pts.append(P @ c2w[i, :3, :3].T + c2w[i, :3, 3])
    pts = np.concatenate(pts)
    lo, hi = np.percentile(pts, 1, 0), np.percentile(pts, 99, 0)
    vox = max(zmed / 30, float(np.prod(hi - lo) / MAX_VOXELS) ** (1 / 3))
    T, Wt, _, dims = tsdf(depth, rgb, K, c2w, lo, hi, vox)
    keep = np.zeros_like(mask)
    H, W = depth.shape[1:]
    vv, uu = np.mgrid[0:H, 0:W]
    for i in range(len(depth)):
        d = depth[i]
        P = np.stack([(uu - K[0, 2]) / K[0, 0] * d, (vv - K[1, 2]) / K[1, 1] * d, d], -1) @ c2w[i, :3, :3].T + c2w[i, :3, 3]
        g = np.floor((P - lo) / vox).astype(int)
        inb = mask[i] & (g >= 0).all(-1) & (g < dims).all(-1)
        gi = g[inb]
        t, w = T[gi[:, 2], gi[:, 1], gi[:, 0]], Wt[gi[:, 2], gi[:, 1], gi[:, 0]]
        k = np.zeros(inb.sum(), bool)
        k[:] = (w >= 0.5) & (np.abs(t) < CONSENSUS)
        keep[i][inb] = k
    return keep


def main():
    d, wname, source = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    names, K, c2w, (W, H), anchors, depth = fitted_depth(d, d / "moge_full.npz")
    D = np.load(d / "moge_full.npz")["depth"]
    scale = frame_scales(anchors, D, (W, H))
    cut = pieces(scale, c2w[:, :3, 3])
    print("pieces (s):", [(round(a / 4, 1), round(b / 4, 1)) for a, b in cut], flush=True)
    rgb_all = [cv2.cvtColor(cv2.imread(str(d / "images" / n)), cv2.COLOR_BGR2RGB) for n in names]
    p = world._path(wname)
    if p.exists():
        p.unlink()
    used = []
    for a, b in cut:
        idx = list(range(a, b))
        dep, cw = depth[a:b].copy(), c2w[a:b].copy()
        if not (dep > 0).any(axis=(1, 2)).sum() >= MIN_PIECE // 2:
            print(f"  {a / 4:.1f}-{b / 4:.1f}s: too few fitted frames, skipped")
            continue
        mask = dep > 0
        rgb = np.stack([rgb_all[i] for i in idx])
        keep = consensus(dep, mask, rgb, K, cw, anchors[a:b])
        if keep.sum() < MIN_KEPT:
            print(f"  {a / 4:.1f}-{b / 4:.1f}s: only {keep.sum()} pixels agree, skipped", flush=True)
            continue
        # Units: this piece's own COLMAP scale; world.add_scene sizes it from the eye height anyway.
        R, how = level(cw, dep, keep, K)
        cw[:, :3, :3] = R @ cw[:, :3, :3]
        cw[:, :3, 3] = cw[:, :3, 3] @ R.T
        cw[:, :3, 3] -= cw[0, :3, 3].copy()
        Ks = np.repeat(K[None], len(idx), 0).astype(np.float32)
        sc = {"rgb": rgb, "depth": dep, "mask": keep, "K": Ks, "c2w": cw.astype(np.float32), "hud": None, "scale_kind": "relative",
              "stats": {"frames": len(idx), "ghosts_removed": round(1 - float(keep.sum() / max(mask.sum(), 1)), 4), "levelled": True,
                        "settings": {"geometry": "colmap+moge, TSDF consensus"}, "level": how}}
        label = f"{source} {a / 4:.1f}-{b / 4:.1f}s"
        sha = hashlib.sha256(f"{label}|{d}".encode()).hexdigest()
        rid = f"{sha[:16]}_scene_colmap03"
        store._scene_clouds.pop(rid, None)
        store._scenes.pop(rid, None)
        store._save_scene(store.RUNS / rid, rid, sc, label, sha)
        w = world.add_scene(wname, rid)
        used.append((a, b))
        s = w["sessions"][-1]
        if not s["confirmed"] and s["score"] is not None and s["score"] >= AUTO_CONFIRM:
            w = world.update(wname, len(w["sessions"]) - 1, confirmed=True)
            s = w["sessions"][-1]
        print(f"  {label}: {len(idx)} frames, kept {100 * keep.sum() / max(mask.sum(), 1):.0f}% by consensus, level by {how}, "
              f"eye scale {s['eye_scale']}, placed turns {s['turns']} scale {s['scale']:.3f} score {s['score']}, confirmed {s['confirmed']}", flush=True)
    # For place_pieces.py: the pieces in World order, and keyframe pairs across pieces to match.
    (d / "pieces.json").write_text(json.dumps({"pieces": used, "names": names}))
    pairs = [f"{names[i]} {names[j]}" for x in range(len(used)) for y in range(x + 1, len(used))
             for i in range(*used[x], 3) for j in range(*used[y], 3) if abs(i - j) > 12]
    (d / "cross_pairs.txt").write_text("\n".join(pairs) + "\n")
    print(f"wrote pieces.json and {len(pairs)} cross-piece pairs to match", flush=True)


if __name__ == "__main__":
    main()
