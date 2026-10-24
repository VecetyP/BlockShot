"""Fused clouds (tsdf_fuse.py) -> Minecraft builds at character scale, and a side-by-side render.

Up is the normal of the plane through the camera path when the path covers an area (walking on one
roof: 2.6 degrees better than the cameras' average up on Test4, 63 on Test5), else the cameras' average up. The grid is turned along the walls
(minecraft.wall_facing, normals from local point neighbourhoods), and one block is the camera's
height above the floor / 1.62 (a player's eye height). Writes <out>/<cloud>_<shape>.schem and
<out>/builds.jpg.

A cloud name ending in "+floor" also gets the floor the camera walked on: one eye height below the
camera path's median height, over the path's area (its loop filled, widened by FLOOR_MARGIN). Depth models are worst
right under the camera and plain floors give COLMAP almost no points, so this floor is otherwise lost.

    uv run python tools/bakeoff/build_cloud.py <fused_clouds.npz> <out dir> [cloud[+floor] ...]
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend import boxify, minecraft  # noqa: E402


def normals(P, k=12, sample=200_000):
    q = P[np.random.default_rng(0).choice(len(P), min(sample, len(P)), replace=False)]
    _, idx = cKDTree(P).query(q, k=k)
    nb = P[idx] - P[idx].mean(1, keepdims=True)
    _, _, vt = np.linalg.svd(nb, full_matrices=False)
    return vt[:, 2]


AREA_PATH = 0.2  # the path's plane is used as level when its second extent is at least this share of its first
# (Test5, looking down over four ledges: the cameras' average up was 63 degrees off, the path plane right)
FLOOR_MARGIN = 2.5  # player heights (eye heights) around the walked path that are taken as floor too


def walked_floor(Pm, C, cm, eye, step):
    """Floor points (Minecraft axes, y up), flat at one eye height below the path's median camera
    height, over the area the path covers (loop filled) widened by FLOOR_MARGIN eye heights. Colour:
    the median of fused points within a step of that floor."""
    lo = cm[:, [0, 2]].min(0) - FLOOR_MARGIN * eye - step
    n = np.ceil((cm[:, [0, 2]].max(0) + FLOOR_MARGIN * eye + step - lo) / step).astype(int) + 1
    grid = np.zeros(n, bool)
    # The path as a dense line (no gaps between cameras), then widened and its loop filled.
    t = np.linspace(0, len(cm) - 1, 20 * len(cm))
    path = np.stack([np.interp(t, np.arange(len(cm)), cm[:, k]) for k in (0, 2)], 1)
    ij = np.floor((path - lo) / step).astype(int)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = ndimage.binary_fill_holes(ndimage.binary_dilation(grid, iterations=int(round(FLOOR_MARGIN * eye / step))))
    i, j = np.nonzero(grid)
    xz = lo + (np.stack([i, j], 1) + 0.5) * step
    F = np.stack([xz[:, 0], np.full(len(xz), np.median(cm[:, 1]) - eye), xz[:, 1]], 1)
    on = np.abs(Pm[:, 1] - (np.median(cm[:, 1]) - eye)) < step
    col = np.median(C[on], 0) if on.sum() > 50 else np.median(C, 0)
    return F, np.repeat(col[None], len(F), 0).astype(np.uint8)


def render(b, width=900, yaw=35, pitch=40):
    p = b.ijk.astype(float)
    c = minecraft.BLOCK_RGB[b.block].astype(float)
    a, e = math.radians(yaw), math.radians(pitch)
    x = p[:, 0] * math.cos(a) - p[:, 2] * math.sin(a)
    zd = p[:, 0] * math.sin(a) + p[:, 2] * math.cos(a)
    y = -p[:, 1] * math.cos(e) + zd * math.sin(e)
    depth = zd * math.cos(e) + p[:, 1] * math.sin(e)
    k = (width - 20) / (x.max() - x.min() + 1)
    img = np.zeros((int((y.max() - y.min() + 1) * k) + 20, width, 3), np.uint8)
    u = ((x - x.min()) * k + 10).astype(int)
    v = ((y - y.min()) * k + 10).astype(int)
    r = max(1, math.ceil(k))
    for i in np.argsort(depth):
        img[v[i]:v[i] + r, u[i]:u[i] + r] = c[i]
    return img


PITCH = float(__import__("os").environ.get("RENDER_PITCH", 40))


def main():
    z = np.load(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    names = sys.argv[3:] or ["cut", "tsdf"]
    cams = z["cams"].astype(np.float64)
    up = -cams[:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    _, sv, vt = np.linalg.svd(cams[:, :3, 3] - cams[:, :3, 3].mean(0))
    if sv[1] > AREA_PATH * sv[0]:  # the path covers an area (not one line): its plane is level
        up = vt[2] * np.sign(vt[2] @ up)
    pose0 = minecraft.Pose(cams[0, :3, 3], cams[0, :3, 2], up)
    panels = []
    for name in names:
        cloud, floor = name.removesuffix("+floor"), name.endswith("+floor")
        P, C = z[f"{cloud}_points"].astype(np.float64), z[f"{cloud}_rgb"]
        pose = minecraft.Pose(pose0.position, minecraft.wall_facing(normals(P), up, pose0.forward), up)
        B, _ = minecraft._basis(pose)
        Pm = (P - pose.position) @ B
        cm = (cams[:, :3, 3] - pose.position) @ B
        # Eye height from the unfused cloud: fusion (and the cut) lose the plain floor under the camera.
        R = (z["raw_points"].astype(np.float64) - pose.position) @ B
        R = R[np.random.default_rng(0).choice(len(R), min(len(R), 400_000), replace=False)]
        eye = minecraft.eye_height([(R[None], None, np.ones((1, len(R)), bool), c) for c in cm])
        s = eye / minecraft.EYE[1]
        if floor:
            F, FC = walked_floor(Pm, C, cm, eye, s / 2)
            P = np.concatenate([P, F @ B.T + pose.position])
            C = np.concatenate([C, FC])
            print(f"{name}: floor of {len(F)} points, {eye:.3f} units below the camera path", flush=True)
        b = minecraft.build(P, C, pose, 70.0, "meters", s)
        builds = {"voxels": b}
        ijk, block = boxify.boxes(b.ijk, b.block, b.dims)
        builds["boxes"] = minecraft.replace(b, ijk=ijk, block=block)
        for shape, bb in builds.items():
            (out / f"{name}_{shape}.schem").write_bytes(minecraft.whole(bb))
            im = render(bb, pitch=PITCH)
            cv2.putText(im, f"{name} - {shape} - {bb.dims[0]}x{bb.dims[1]}x{bb.dims[2]} blocks", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panels.append(im)
            print(name, shape, bb.dims, len(bb.ijk), f"block = {s:.3f} units (eye {eye:.3f})", flush=True)
    W = max(p.shape[1] for p in panels)
    rows = [np.pad(p, ((0, 0), (0, W - p.shape[1]), (0, 0))) for p in panels]
    pairs = [np.concatenate(rows[i:i + 2], 1) if i + 1 < len(rows) else np.pad(rows[i], ((0, 0), (0, W), (0, 0)))
             for i in range(0, len(rows), 2)]
    H = max(p.shape[0] for p in pairs)
    cv2.imwrite(str(out / "builds.jpg"), cv2.cvtColor(np.concatenate([np.pad(p, ((0, H - p.shape[0]), (0, 0), (0, 0))) for p in pairs], 0), cv2.COLOR_RGB2BGR))
    print("wrote", out / "builds.jpg")


if __name__ == "__main__":
    main()
