"""Video -> one consistent scene.

1. Sample frames at a fixed rate, drop near-duplicates.
2. Find a game HUD: pixels that stay still and textured while the scene moves; they're ignored.
3. VGGT on all frames at once (or overlapping chunks joined by a Sim(3) fit) gives every frame's
   camera pose, intrinsics and depth in one frame of reference: the first frame's camera.
4. Depth: VGGT's own (default), predicted jointly with the poses, so frames agree with each other;
   or MoGe's sharper per-frame depth bent onto VGGT's smooth shape. MoGe on a few frames sets the
   scale in metres either way.
5. Free-space filter: a pixel is a ghost, and dropped, when more frames saw straight through it
   (it floats in front of every surface they saw near there) than saw it.

Measured on a turning-around room clip (24 frames), share of overlapping points that two frames
agree on within 5% / that float in front of another frame's surface ("ghosts"):
  MoGe, one scale per frame (before)  62% / 12.8%
  MoGe bent onto VGGT's shape          70% / 9.9%
  VGGT depth + free-space filter       82% / 2.9%  (keeps 97% of pixels)
The filter's first version (any 2 frames seeing through, exact pixel; together with a cut of VGGT's
least confident 20%) got 0.4% there, but kept only 58% of a hallway clip: the cut took plain walls
and shiny floors, the filter a perforated brick wall (other frames see through its holes). Now 94%.

The scene is kept as per-frame grids (depth, colour, mask, K, camera-to-world) so later steps can
fill surfaces between pixels; `world_points` turns a frame into points in scene coordinates.
"""
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, grey_erosion, label, sobel

from .models import registry
from .models.base import unproject

PIPELINE = 4  # bump when reconstruction changes, so cached scenes are rebuilt (4: no blanket confidence cut)
VGGT_ONE_PASS = 96  # frames VGGT handles in one pass (measured on 16 GB: 64 frames peak at 9.8 GB)
CHUNK, OVERLAP = 64, 16  # longer clips: overlapping chunks, joined on the shared frames
DUP_DIFF = 0.012  # a frame this close (mean abs difference, 0..1) to the last kept one is dropped
LOW_CONF_PCT = 20  # per frame, VGGT's least confident 20% of pixels are ignored...
FAR = 4  # ...if they're also this many times farther than the frame's median depth (sky, haze)
METRIC_FRAMES = 6  # MoGe frames used to put a VGGT-depth scene in metres
SHAPE_BLUR = 0.05  # MoGe geometry: VGGT's shape is followed at this scale (fraction of the width)
FREE_SPACE_TOL = 0.05  # a point this much (relative) in front of another frame's surface "floats"...
FREE_SPACE_WINDOW = 7  # ...in front of every surface that frame saw within this many pixels...
FREE_SPACE_VOTES = 2  # ...for at least this many frames, and more of them than frames that agree with it


@dataclass
class Settings:
    fps: float = 2.0
    max_frames: int = 60
    max_side: int = 640
    hud: bool = True  # ignore a static game HUD
    geometry: str = "vggt"  # "vggt": VGGT depth (frames agree); "moge": MoGe detail on VGGT's shape


def _quiet(*_, **__):
    pass


# --- frames --------------------------------------------------------------------------------------

def sample_frames(path: str, s: Settings, progress=_quiet) -> tuple[list[np.ndarray], dict]:
    """RGB frames at about `s.fps`, near-duplicates dropped, at most `s.max_frames` (evenly spread),
    longest side at most `s.max_side` px."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("could not open the video (unsupported format?)")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, round(src_fps / s.fps))
    kept, last, read, dups, i = [], None, 0, 0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            progress("reading video", i, total)
            ok, bgr = cap.retrieve()
            if not ok:
                break
            read += 1
            h, w = bgr.shape[:2]
            k = min(1.0, s.max_side / max(h, w))
            if k < 1:
                bgr = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
            thumb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA) / 255.0
            if last is not None and np.abs(thumb - last).mean() < DUP_DIFF:
                dups += 1
            else:
                kept.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                last = thumb
        i += 1
    cap.release()
    if len(kept) > s.max_frames:
        kept = [kept[j] for j in np.linspace(0, len(kept) - 1, s.max_frames).round().astype(int)]
    if len(kept) < 2:
        raise ValueError("need at least 2 distinct frames; is the camera moving?")
    return kept, {"video_fps": round(src_fps, 2), "frames_read": read, "duplicates_dropped": dups, "frames": len(kept)}


def hud_mask(frames: list[np.ndarray]) -> np.ndarray | None:
    """Pixels of a static overlay (game HUD, watermark): they barely change while the rest of the
    image moves, and they're textured (text, icons), unlike a plain static sky. None if there's no
    such overlay, or if most of the image is static (then the camera isn't moving, not a HUD)."""
    h, w = frames[0].shape[:2]
    small = np.stack([cv2.resize(f, (w // 4, h // 4), interpolation=cv2.INTER_AREA).mean(-1) for f in frames]) / 255.0
    std = small.std(0)
    mean = small.mean(0)
    grad = np.hypot(sobel(mean, 0), sobel(mean, 1))
    static = std < 0.15 * np.median(std)
    textured = grad > np.percentile(grad, 80)
    seeds = binary_opening(static & textured, iterations=1)  # text and icons
    if seeds.mean() < 0.0005 or static.mean() > 0.5:
        return None
    # Grow from the seeds across the static region they sit in: a HUD panel's flat background
    # is part of the overlay too.
    lab, _ = label(static)
    hud = np.isin(lab, np.unique(lab[seeds])) & (lab > 0)
    hud = binary_dilation(hud, iterations=2)
    return cv2.resize(hud.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


# --- geometry helpers ------------------------------------------------------------------------------

def umeyama(src: np.ndarray, dst: np.ndarray, scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity dst ≈ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (a**2).sum(1).mean()) if scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def c2w_from_extrinsics(E: np.ndarray) -> np.ndarray:
    """(S, 3, 4) camera-from-world -> (S, 4, 4) camera-to-world."""
    R, t = E[:, :, :3], E[:, :, 3]
    out = np.tile(np.eye(4), (len(E), 1, 1))
    out[:, :3, :3] = R.transpose(0, 2, 1)
    out[:, :3, 3] = -np.einsum("sji,sj->si", R, t)
    return out


def world_points(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """(H, W, 3) scene-frame points of one frame."""
    P = unproject(depth, K).astype(np.float64)
    return P @ c2w[:3, :3].T + c2w[:3, 3]


def merge_chunks(chunks: list[dict], starts: list[int], n: int) -> dict:
    """Join VGGT chunk results (each in its own frame and scale) into one, aligning every chunk to
    the frames it shares with what's already placed (Sim(3) on matching pixels' 3D points)."""
    h, w = chunks[0]["depth"].shape[1:]
    c2w = np.zeros((n, 4, 4))
    K = np.zeros((n, 3, 3))
    depth = np.zeros((n, h, w), np.float32)
    conf = np.zeros((n, h, w), np.float32)
    placed = np.zeros(n, bool)
    v, u = np.mgrid[0:h:8, 0:w:8]  # sparse pixel grid for the fit
    for ch, a in zip(chunks, starts):
        idx = np.arange(a, a + len(ch["depth"]))
        cw = c2w_from_extrinsics(ch["extrinsics"])
        shared = [i for i in idx if placed[i]]
        if shared:
            src, dst = [], []
            for i in shared:
                j = i - a
                ok = ((ch["depth_conf"][j] > np.percentile(ch["depth_conf"][j], 50)) & (conf[i] > np.percentile(conf[i], 50)))[v, u]
                src.append(world_points(ch["depth"][j], ch["intrinsics"][j], cw[j])[v, u][ok])
                dst.append(world_points(depth[i], K[i], c2w[i])[v, u][ok])
            s, R, t = umeyama(np.concatenate(src), np.concatenate(dst))
        else:
            s, R, t = 1.0, np.eye(3), np.zeros(3)
        S = np.eye(4)
        S[:3, :3], S[:3, 3] = s * R, t
        D = np.diag([1 / s, 1 / s, 1 / s, 1.0])
        for j, i in enumerate(idx):
            if placed[i]:
                continue  # keep the earlier chunk's estimate for shared frames
            c2w[i] = S @ cw[j] @ D
            K[i] = ch["intrinsics"][j]
            depth[i] = ch["depth"][j] * s
            conf[i] = ch["depth_conf"][j]
            placed[i] = True
    return {"c2w": c2w, "K": K, "depth": depth, "conf": conf}


def vggt_poses(frames: list[np.ndarray], progress=_quiet) -> tuple[dict, int]:
    """Camera-to-world, intrinsics, depth and confidence for every frame, in the first frame's
    camera coordinates. Returns (result, number of VGGT passes)."""
    model = registry.get("vggt")
    n = len(frames)
    if n <= VGGT_ONE_PASS:
        progress("finding camera poses (VGGT)", 0, 1, f"{n} frames in one pass")
        out = model.predict_frames(frames)
        return merge_chunks([out], [0], n), 1
    starts = list(range(0, n - OVERLAP, CHUNK - OVERLAP))
    chunks = []
    for k, a in enumerate(starts):
        progress("finding camera poses (VGGT)", k, len(starts), f"chunk {k + 1}/{len(starts)}")
        chunks.append(model.predict_frames(frames[a:a + CHUNK]))
        registry.free_cache()
    return merge_chunks(chunks, starts, n), len(starts)


def free_space_filter(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, c2w: np.ndarray, progress=_quiet) -> np.ndarray:
    """Mask without ghosts. A pixel "floats" for another frame when it lands more than
    FREE_SPACE_TOL in front of every surface that frame saw within FREE_SPACE_WINDOW pixels (space
    it looked straight through), and is "seen" when that frame's surface there agrees with it. It's
    dropped when it floats for FREE_SPACE_VOTES frames or more, and for more frames than see it.
    The window keeps perforated walls (others see through the holes, not the wall); counting the
    frames that agree keeps floors under reflections (read as depth below the floor)."""
    n, h, w = depth.shape
    keep = mask.copy()
    w2c = np.linalg.inv(c2w)
    nearest = np.stack([grey_erosion(np.where(mask[j], depth[j], np.inf), size=FREE_SPACE_WINDOW) for j in range(n)])
    for i in range(n):
        progress("removing ghosts", i, n)
        P = world_points(depth[i], K[i], c2w[i]).reshape(-1, 3)[mask[i].ravel()]
        votes = np.zeros(len(P), np.int16)
        seen = np.zeros(len(P), np.int16)
        for j in range(n):
            if j == i:
                continue
            q = P @ w2c[j, :3, :3].T + w2c[j, :3, 3]
            z = q[:, 2]
            ok = np.flatnonzero(z > 1e-6)
            u = (K[j][0, 0] * q[ok, 0] / z[ok] + K[j][0, 2]).astype(np.int64)
            v = (K[j][1, 1] * q[ok, 1] / z[ok] + K[j][1, 2]).astype(np.int64)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            ok, u, v = ok[inside], u[inside], v[inside]
            near = nearest[j][v, u]  # inf where that frame has no depth nearby: no evidence either way
            votes[ok[np.isfinite(near) & (z[ok] < near * (1 - FREE_SPACE_TOL))]] += 1
            seen[ok[mask[j][v, u] & (np.abs(z[ok] - depth[j][v, u]) <= FREE_SPACE_TOL * depth[j][v, u])]] += 1
        keep[i][mask[i]] = (votes < FREE_SPACE_VOTES) | (votes <= seen)
    return keep


def _onto_shape(fine: np.ndarray, coarse: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """`fine` depth bent to follow `coarse` depth's smooth shape: fine * blur(coarse / fine)."""
    r = np.where(ok, np.log(coarse / np.maximum(fine, 1e-6)), 0).astype(np.float32)
    wgt = ok.astype(np.float32)
    s = SHAPE_BLUR * fine.shape[1]
    ratio = cv2.GaussianBlur(r * wgt, (0, 0), s) / np.maximum(cv2.GaussianBlur(wgt, (0, 0), s), 1e-6)
    return fine * np.exp(ratio)


# --- the whole pipeline ---------------------------------------------------------------------------

def reconstruct(path: str, s: Settings, progress=_quiet) -> dict:
    """Video file -> scene dict: rgb (F, H, W, 3), depth (F, H, W), mask, K (F, 3, 3), c2w (F, 4, 4),
    hud (H, W) or None, scale_kind, stats."""
    timings = {}
    t = time.perf_counter()
    frames, stats = sample_frames(path, s, progress)
    timings["sample"] = time.perf_counter() - t

    t = time.perf_counter()
    hud = hud_mask(frames) if s.hud else None
    stats["hud_fraction"] = round(float(hud.mean()), 4) if hud is not None else 0.0
    timings["hud"] = time.perf_counter() - t

    t = time.perf_counter()
    vg, stats["vggt_passes"] = vggt_poses(frames, progress)
    registry.free_cache()
    timings["poses"] = time.perf_counter() - t

    n = len(frames)
    # Low confidence alone isn't a reason to drop a pixel: VGGT is unsure of plain walls and shiny
    # floors too (on a hallway clip half the pixels sat at its minimum confidence, and a blanket cut
    # removed 42% of the floor). The ghost filter catches wrong geometry; this only drops unsure
    # pixels far away, which is sky.
    mask = np.zeros(vg["depth"].shape, bool)
    for i in range(n):
        d, valid = vg["depth"][i], vg["depth"][i] > 0
        unsure = vg["conf"][i] <= np.percentile(vg["conf"][i], LOW_CONF_PCT)
        mask[i] = valid & ~(unsure & (d > FAR * np.median(d[valid])))
    if hud is not None:
        mask &= ~hud
    depth = vg["depth"].copy()
    t = time.perf_counter()
    moge = registry.get("moge")
    # MoGe for the scale in metres (a few frames), or for every frame's depth.
    which = range(n) if s.geometry == "moge" else np.unique(np.linspace(0, n - 1, min(n, METRIC_FRAMES)).round().astype(int))
    ratios = []
    for k, i in enumerate(which):
        progress("measuring depth (MoGe)", k, len(which), f"frame {i + 1}/{n}")
        p = moge.predict(frames[i])
        registry.free_cache()
        ok = mask[i] & p.mask & (p.depth > 0)
        if ok.sum() < 100:
            continue
        ratios.append(np.median(vg["depth"][i][ok] / p.depth[ok]))  # MoGe metres -> this scene's units
        if s.geometry == "moge":
            depth[i] = _onto_shape(p.depth, vg["depth"][i], ok)
            mask[i] &= p.mask
    # One global scale to metres, so frames stay consistent with each other.
    R = float(np.median(ratios)) if ratios else 1.0
    depth /= R
    vg["c2w"][:, :3, 3] /= R
    stats["scale_spread"] = round(float(max(ratios) / min(ratios)), 3) if ratios else None
    scale_kind = "metric" if ratios else "relative"
    timings["depth"] = time.perf_counter() - t

    t = time.perf_counter()
    before = int(mask.sum())
    mask = free_space_filter(depth, mask, vg["K"], vg["c2w"], progress)
    stats["ghosts_removed"] = round(1 - int(mask.sum()) / max(before, 1), 4)
    timings["ghosts"] = time.perf_counter() - t

    scene = {"rgb": np.stack(frames), "depth": depth.astype(np.float32), "mask": mask, "K": vg["K"],
             "c2w": vg["c2w"], "hud": hud, "scale_kind": scale_kind}
    stats["timings_s"] = {k: round(v, 1) for k, v in timings.items()}
    stats["settings"] = asdict(s)
    scene["stats"] = stats
    return scene
