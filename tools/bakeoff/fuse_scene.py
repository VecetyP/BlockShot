"""COLMAP cameras + dense depth fitted per frame to COLMAP's well-triangulated points -> a BlockShot scene run.

    uv run python tools/bakeoff/fuse_scene.py <sfm dir> <dense .npz> <source name> [min angle deg]
"""
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import align_dense  # noqa: E402
import colmap_text  # noqa: E402
import common  # noqa: E402
from backend import minecraft, store, video  # noqa: E402

HUD = 26
REACH = 1.2  # keep depths up to this times the farthest well-measured point in the frame


def track_angles(text: Path) -> list[np.ndarray]:
    """Per image (sorted by name), the triangulation angle of each of its anchors (as colmap_text orders them)."""
    imgs = {}
    lines = [l for l in (text / "images.txt").read_text().splitlines() if not l.startswith("#")]
    for a, b in zip(lines[0::2], lines[1::2]):
        p = a.split()
        R, t = colmap_text._rot(*map(float, p[1:5])), np.array(list(map(float, p[5:8])))
        imgs[int(p[0])] = (p[9], -R.T @ t, np.array(b.split(), float).reshape(-1, 3))
    ang = {}
    for l in (text / "points3D.txt").read_text().splitlines():
        if not l or l.startswith("#"):
            continue
        p = l.split()
        X, tr = np.array(list(map(float, p[1:4]))), np.array(p[8:], int).reshape(-1, 2)
        r = X - np.stack([imgs[i][1] for i in tr[:, 0]])
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        ang[int(p[0])] = np.degrees(np.arccos(np.clip(r @ r.T, -1, 1).min()))
    return [np.array([ang[int(i)] for i in o[2][o[2][:, 2] >= 0][:, 2]]) for o in sorted(imgs.values(), key=lambda x: x[0])]


def fitted_depth(d: Path, npz: Path, amin: float = 5.0):
    """Every frame's dense depth at full size in COLMAP units, bent onto its well-triangulated points
    (monotone curve), cut beyond the farthest of them. -> names, K, c2w, (W, H), anchors, depth."""
    names, K, c2w, (W, H), anchors = colmap_text.load(d / "text")
    angles = track_angles(d / "text")
    D = np.load(npz)["depth"].astype(np.float64)
    h, w = D.shape[1:]
    depth = np.zeros((len(D), H, W), np.float32)
    for i, a in enumerate(anchors):
        u = np.clip((a[:, 0] * w / W).astype(int), 0, w - 1)
        v = np.clip((a[:, 1] * h / H).astype(int), 0, h - 1)
        dd = D[i][v, u]
        ok = (dd > 0) & (a[:, 2] > 0) & (a[:, 1] > HUD) & (angles[i] >= amin)
        if ok.sum() < 40:
            continue  # frame left out
        f = align_dense.fit("curve", dd[ok], a[ok, 2])
        full = cv2.resize(D[i], (W, H), interpolation=cv2.INTER_NEAREST)
        z = np.where(full > 0, f(full), 0)
        depth[i] = np.where(z <= REACH * a[ok, 2].max(), z, 0)
    depth[:, :HUD] = 0
    return names, K, c2w, (W, H), anchors, depth


def main():
    d, npz, source = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    amin = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    names, K, c2w, (W, H), anchors, depth = fitted_depth(d, npz, amin)
    D = depth
    mask = depth > 0
    mask[:, :HUD] = False
    g = np.median([np.median(a[:, 2]) for a in anchors]) / 5.0  # rough: median anchor ~5 m
    depth /= g
    cw = c2w.copy()
    cw[:, :3, 3] /= g
    Ks = np.repeat(K[None], len(D), 0).astype(np.float32)
    before = mask.sum()
    mask = video.free_space_filter(depth, mask, Ks, cw)
    ghosts = 1 - mask.sum() / before
    rows = common.pair_stats(depth, mask, Ks.astype(float), cw)
    wt, nb = rows[:, 2], np.abs(rows[:, 0] - rows[:, 1]) <= 1
    print(f"frames used {int((depth.reshape(len(D), -1).max(1) > 0).sum())}/{len(D)}, dropped by ghost filter {100 * ghosts:.0f}%, "
          f"neighbour agreement {100 * np.average(rows[nb, 3], weights=wt[nb]):.0f}%, all pairs {100 * np.average(rows[:, 3], weights=wt):.0f}%", flush=True)
    N = np.concatenate([minecraft.grid_normals(video.world_points(depth[i], Ks[i].astype(float), cw[i]), mask[i]) for i in range(len(D))])
    up, share = minecraft.find_up(N, -cw[:, :3, 1].mean(0))
    R = minecraft.rotation_between(up, np.array([0.0, -1.0, 0.0]))
    cw[:, :3, :3] = R @ cw[:, :3, :3]
    cw[:, :3, 3] = cw[:, :3, 3] @ R.T
    cw[:, :3, 3] -= cw[0, :3, 3].copy()
    rgb = np.stack([cv2.cvtColor(cv2.imread(str(d / "images" / n)), cv2.COLOR_BGR2RGB) for n in names])
    sc = {"rgb": rgb, "depth": depth, "mask": mask, "K": Ks, "c2w": cw.astype(np.float32), "hud": None, "scale_kind": "relative",
          "stats": {"frames": len(D), "ghosts_removed": round(float(ghosts), 4), "levelled": True, "level_share": round(share, 3),
                    "settings": {"geometry": f"colmap+{npz.stem}"}}}
    sha = hashlib.sha256(f"{source}|{npz.name}|{amin}".encode()).hexdigest()
    rid = f"{sha[:16]}_scene_colmap02"
    store._save_scene(store.RUNS / rid, rid, sc, source, sha)
    print("saved", rid)


if __name__ == "__main__":
    main()
