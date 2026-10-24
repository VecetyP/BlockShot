"""Worlds: several video scenes placed into one Minecraft map, recorded over any number of sessions.

A world is worlds/<name>/world.json: the scenes it uses and where each one goes. Scenes stay in the
run cache untouched; a world only stores transforms, so it can be rebuilt any time. Each scene is
turned so its walls run along the block grid (minecraft.wall_facing), so between scenes only a
quarter turn, a scale and an offset are left. World coordinates are Minecraft blocks (x east, y up,
z south) with the first scene's first camera at the origin.

Placing a new scene: it's sized character-sized from its camera height (minecraft.eye_height), then
its top-down height map is matched against the world's for each quarter turn and a few scales
around that size (correlation by FFT); the best match is proposed, and the user confirms or nudges
it. Scenes overlap reliably when each recording starts somewhere already in the world.

Long gameplay recordings lose track (VGGT's scale drifts across a fast clip), so add_video cuts one
into short dense pieces that overlap by a few frames, reconstructs each on its own, and places each
piece from the frames it shares with the one before (follow). Pieces whose frames disagree are added
unconfirmed, for the user to keep or drop.
"""
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from . import minecraft, store, video
from .models import registry

WORLDS = store.RUNS.parent / "worlds"
WORLD_POSE = minecraft.Pose(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))  # already Minecraft axes
SCALES = np.geomspace(0.8, 1.25, 9)  # tried around the eye-height size
WIDE_SCALES = np.geomspace(0.25, 4.0, 25)  # when the scene never shows the floor
MIN_OVERLAP = 0.1  # share of the new scene's columns that must land on the world
MIN_VAR = 0.25  # blocks²: overlaps flatter than this can't be told apart
BUDGET = 20_000_000  # ponytail: one dense build; split per region if worlds outgrow memory
MAP_CELLS = 40_000  # cells per scene sent to the map view
PIECE_S = 2.5  # seconds per piece: Test1 (Roblox) at 8 fps agreed 91% between neighbours, 52% at 2 fps over 10 s
PIECE_FPS = 8.0
PIECE_OVERLAP = 4  # frames shared by neighbouring pieces, to place one from the other
GOOD_GHOSTS = 0.3  # pieces losing more points than this to the ghost filter are added unconfirmed


def _path(name: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("world names are letters, digits, - and _")
    return WORLDS / name / "world.json"


def list_worlds() -> list[str]:
    return sorted(p.parent.name for p in WORLDS.glob("*/world.json"))


def load(name: str) -> dict:
    p = _path(name)
    return json.loads(p.read_text()) if p.exists() else {"name": name, "sessions": []}


def save(w: dict) -> dict:
    p = _path(w["name"])
    p.parent.mkdir(parents=True, exist_ok=True)
    store._write_json(p, w)
    return w


def rot(turns: int, yaw: float = 0.0) -> np.ndarray:
    """Quarter turns about "up" (y), plus `yaw` degrees more (a scene whose walls aren't on its grid)."""
    if yaw:
        a = math.radians(90 * turns + yaw)
        c, s = math.cos(a), math.sin(a)
    else:
        c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][turns % 4]
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def srot(s: dict) -> np.ndarray:
    """A placed scene's rotation (its quarter turns and optional extra yaw)."""
    return rot(s["turns"], s.get("yaw", 0.0))


def _axes(rid: str):
    """A scene, its frame-0 pose and FOV, its grids, and B: columns = its own Minecraft axes along
    its walls, in scene coordinates."""
    sc = store.load_scene(rid)
    pose, fov = store._scene_pose(sc, 0, bool(store.load_meta(rid).get("levelled")))
    raw = store._scene_grids(sc)
    N = np.concatenate([minecraft.grid_normals(P.astype(np.float64), m & np.isfinite(P).all(-1)) for P, _, m, _ in raw])
    B, _ = minecraft._basis(minecraft.Pose(pose.position, minecraft.wall_facing(N, pose.up, pose.forward), pose.up))
    return sc, pose, fov, raw, B


def scene_cameras(rid: str) -> tuple[np.ndarray, np.ndarray]:
    """Every frame's camera centre and viewing direction in the scene's own axes (as scene_grids)."""
    sc, pose, _, _, B = _axes(rid)
    c2w = sc["c2w"].astype(np.float64)
    return (c2w[:, :3, 3] - pose.position) @ B, c2w[:, :3, 2] @ B


def scene_grids(rid: str) -> tuple[list, float, float | None]:
    """A scene's grids in its own Minecraft axes along its walls (cloud units, first camera at the
    origin), its vertical FOV, and blocks per unit for character size (None: no floor seen)."""
    sc, pose, fov, raw, B = _axes(rid)
    grids = [((P - pose.position) @ B, rgb, m & np.isfinite(P).all(-1), (c - pose.position) @ B) for P, rgb, m, c in raw]
    try:
        per_unit = minecraft.EYE[1] / minecraft.eye_height(grids)
    except ValueError:
        per_unit = None
    return grids, fov, per_unit


def place(points: np.ndarray, s: dict) -> np.ndarray:
    """Scene points (its own axes, cloud units) -> world blocks."""
    return points * s["scale"] @ srot(s).T + np.asarray(s["offset"], float)


def _cloud(grids, sample: int = 2_000_000) -> np.ndarray:
    p = np.concatenate([P[m] for P, _, m, _ in grids])
    lo, hi = np.percentile(p, minecraft.OUTLIER_PCT, 0), np.percentile(p, 100 - minecraft.OUTLIER_PCT, 0)
    p = p[((p >= lo) & (p <= hi)).all(1)]
    return p[np.random.default_rng(0).choice(len(p), min(sample, len(p)), replace=False)]


def heightmap(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Highest block per (x, z) column of points in blocks: (H, origin xz); NaN where nothing."""
    ij = np.floor(p[:, [0, 2]]).astype(np.int64)
    o = ij.min(0)
    ij -= o
    H = np.full(ij.max(0) + 1, -np.inf)
    np.maximum.at(H, (ij[:, 0], ij[:, 1]), p[:, 1])
    H[np.isinf(H)] = np.nan
    return H, o


def _xcorr(a, b):
    """c[i] = sum_x a[x + d] b[x] with d = i - (b.shape - 1), for every shift d."""
    shape = [a.shape[k] + b.shape[k] - 1 for k in (0, 1)]
    return np.fft.irfft2(np.fft.rfft2(a, shape) * np.fft.rfft2(b[::-1, ::-1], shape), shape)


def match(Hw: np.ndarray, Hn: np.ndarray, min_overlap: float = MIN_OVERLAP) -> tuple[float, np.ndarray, float]:
    """Best shift of height map Hn onto Hw (NaN = unseen): (correlation of heights where both are
    seen, shift d (cell x of Hn lands on cell x + d of Hw), height difference world - new)."""
    Mw, Mn = np.isfinite(Hw).astype(float), np.isfinite(Hn).astype(float)
    hw, hn = np.nan_to_num(Hw), np.nan_to_num(Hn)
    n = _xcorr(Mw, Mn)
    sw, sn = _xcorr(hw, Mn), _xcorr(Mw, hn)
    sww, snn, swn = _xcorr(hw ** 2, Mn), _xcorr(Mw, hn ** 2), _xcorr(hw, hn)
    ok = n > max(20, min_overlap * Mn.sum())
    n = np.where(ok, n, 1)
    vw, vn = sww / n - (sw / n) ** 2, snn / n - (sn / n) ** 2
    ok &= (vw > MIN_VAR) & (vn > MIN_VAR)  # flat ground matches anywhere: needs height differences
    r = np.where(ok, (swn / n - sw * sn / n ** 2) / np.sqrt(np.maximum(vw * vn, 1)) * np.sqrt(np.minimum(1, n / Mn.sum() / 0.3)), -np.inf)
    i = np.unravel_index(np.argmax(r), r.shape)
    return float(r[i]), np.array(i) - (np.array(Hn.shape) - 1), float((sw[i] - sn[i]) / n[i])


def propose(world_pts: np.ndarray, new_pts: np.ndarray, base: float, scales=SCALES) -> dict:
    """Where scene points `new_pts` (own axes, cloud units) fit onto `world_pts` (world blocks):
    the best quarter turn, scale (base * one of `scales`), offset, and its match score."""
    Hw, ow = heightmap(world_pts)
    best = {"score": -np.inf, "turns": 0, "scale": float(base), "offset": [0.0, 0.0, 0.0]}  # no match: place by hand
    for f in scales:
        for t in range(4):
            q = new_pts * base * f @ rot(t).T
            Hn, on = heightmap(q)
            r, d, dy = match(Hw, Hn)
            if r > best["score"]:
                off = ow + d - on
                best = {"score": r, "turns": t, "scale": float(base * f), "offset": [float(off[0]), dy, float(off[1])]}
    best["score"] = best["score"] if np.isfinite(best["score"]) else None
    return best


def add_scene(name: str, rid: str, progress=minecraft._quiet) -> dict:
    """Adds scene `rid` to the world with a proposed placement (confirmed once the user agrees)."""
    w = load(name)
    progress("reading scene")
    grids, fov, per_unit = scene_grids(rid)
    done = [s for s in w["sessions"] if s["confirmed"]]
    if not done:  # the first scene defines the world
        s = {"scene": rid, "turns": 0, "scale": float(per_unit or 1.0), "offset": [0.0, 0.0, 0.0],
             "confirmed": True, "score": None, "eye_scale": per_unit is not None, "fov_deg": fov}
    else:
        world_pts = np.concatenate([place(_cloud(scene_grids(d["scene"])[0], 500_000), d) for d in done])
        known = [d["scale"] for d in done]
        base, scales = (per_unit, SCALES) if per_unit else (float(np.median(known)), WIDE_SCALES)
        progress("matching rooftops")
        s = {"scene": rid, **propose(world_pts, _cloud(grids, 500_000), base, scales), "confirmed": False,
             "eye_scale": per_unit is not None, "fov_deg": fov}
    w["sessions"].append(s)
    return save(w)


def follow(prev: dict, rid: str, ov: int) -> dict:
    """Placement of scene `rid` whose first `ov` frames are the last `ov` of the placed scene `prev`:
    the quarter turn from the shared cameras' headings, the offset from their positions. The scale
    stays: every scene is already in metres (MoGe), while chaining the shared frames' depth ratios
    compounded drift (Test1: 9x after nine pieces)."""
    ca, fa = scene_cameras(prev["scene"])
    cb, fb = scene_cameras(rid)
    a = slice(len(ca) - ov, len(ca))
    dyaw = np.arctan2(fa[a, 0], fa[a, 2]) - np.arctan2(fb[:ov, 0], fb[:ov, 2])
    turns = (prev["turns"] + round(math.degrees(np.angle(np.exp(1j * dyaw).mean())) / 90)) % 4
    shared = place(ca[a], prev) - cb[:ov] * prev["scale"] @ rot(turns).T
    return {"turns": int(turns), "scale": prev["scale"], "offset": shared.mean(0).tolist()}


def add_video(name: str, data: bytes, filename: str | None, piece_s: float = PIECE_S, fps: float = PIECE_FPS,
              fov_deg: float | None = None, progress=minecraft._quiet) -> dict:
    """A long recording -> overlapping pieces of `piece_s` seconds at `fps`, each its own scene run,
    all added to the world: the first like add_scene, the rest following the piece before."""
    s = video.Settings(fps=fps, max_frames=100_000, fov_deg=fov_deg)
    sha = hashlib.sha256(data).hexdigest()
    with tempfile.NamedTemporaryFile(suffix=Path(filename or "video.mp4").suffix or ".mp4", delete=False) as tmp:
        tmp.write(data)
    try:
        frames, stats = video.sample_frames(tmp.name, s, progress)
    finally:
        os.unlink(tmp.name)
    times = stats["frame_times_s"]
    n = max(PIECE_OVERLAP + 2, round(piece_s * fps))
    starts = [a for a in range(0, max(1, len(frames) - PIECE_OVERLAP), n - PIECE_OVERLAP) if len(frames) - a >= PIECE_OVERLAP + 2]
    prev = None
    for k, a in enumerate(starts):
        fr = frames[a:a + n]
        piece_sha = hashlib.sha256(f"{sha}:{a}:{n}".encode()).hexdigest()
        rid = store._scene_id(piece_sha, s)
        if not (store.RUNS / rid / "meta.json").exists():
            say = lambda stage, *x, k=k, **kw: progress(f"piece {k + 1}/{len(starts)}: {stage}", *x, **kw)
            st = {"frames": len(fr), "frame_times_s": [round(t - times[a], 4) for t in times[a:a + n]], "video_fps": stats["video_fps"]}
            with store._gpu_lock:
                try:
                    scene = video.reconstruct_frames(fr, s, st, {}, say)
                finally:
                    registry.free_cache()
            store._save_scene(store.RUNS / rid, rid, scene, f"{filename or 'video'} {times[a]:.1f}-{times[a + len(fr) - 1]:.1f}s", piece_sha)
        good = store.load_meta(rid)["ghosts_removed"] <= GOOD_GHOSTS
        progress("placing pieces", k, len(starts))
        if prev is None:
            w = add_scene(name, rid)
            if not any(x["confirmed"] for x in w["sessions"][:-1]):
                w["sessions"][-1]["confirmed"] = good or len(w["sessions"]) == 1
        else:
            w = load(name)
            _, fov, per_unit = scene_grids(rid)
            w["sessions"].append({"scene": rid, **follow(prev, rid, PIECE_OVERLAP), "confirmed": good, "score": None,
                                  "eye_scale": False, "fov_deg": fov})
        w["sessions"][-1]["ghosts"] = store.load_meta(rid)["ghosts_removed"]
        prev = w["sessions"][-1]
        save(w)
    return load(name)


def update(name: str, i: int, **changes) -> dict:
    w = load(name)
    w["sessions"][i].update({k: v for k, v in changes.items() if k in ("turns", "yaw", "scale", "offset", "confirmed")})
    return save(w)


def remove(name: str, i: int) -> dict:
    w = load(name)
    del w["sessions"][i]
    return save(w)


def top_view(name: str) -> list[dict]:
    """Per scene: its top-down cells (x, z, height in blocks at its current scale, before its turn
    and offset) for the map view, which applies turn, scale changes and offset live."""
    out = []
    for s in load(name)["sessions"]:
        p = _cloud(scene_grids(s["scene"])[0], 1_000_000) * s["scale"]
        H, o = heightmap(p)
        x, z = np.nonzero(np.isfinite(H))
        step = max(1, math.ceil(math.sqrt(len(x) / MAP_CELLS)))
        keep = (x % step == 0) & (z % step == 0)
        out.append({"scale": s["scale"], "cell": step,
                    "cells": np.stack([x[keep] + o[0], z[keep] + o[1], np.round(H[x[keep], z[keep]], 1)], 1).tolist()})
    return out


def build(name: str, shape: str = "voxels", progress=minecraft._quiet) -> minecraft.Build:
    """All confirmed scenes in one build, one block per world block, viewed from the world origin."""
    w = load(name)
    done = [s for s in w["sessions"] if s["confirmed"]]
    if not done:
        raise ValueError("the world has no confirmed scenes yet")
    grids = []
    for i, s in enumerate(done):
        progress("reading scenes", i, len(done))
        R = srot(s) * s["scale"]
        grids += [(P @ R.T + s["offset"], rgb, m, c @ R.T + s["offset"]) for P, rgb, m, c in scene_grids(s["scene"])[0]]
    return minecraft.build_from_grids(grids, WORLD_POSE, done[0]["fov_deg"], "meters", 1.0, BUDGET, progress, shape=shape)


def export(name: str, shape: str = "voxels", progress=minecraft._quiet) -> str:
    """Writes the world's tiles zip into worlds/<name>/exports/ and returns its file name."""
    b = build(name, shape, progress)
    fname = f"{name}_{shape}.zip"
    d = _path(name).parent / "exports"
    d.mkdir(exist_ok=True)
    data = minecraft.tiles_zip(b, fname[:-4], store.TILE, progress)
    store._atomic(d / fname, lambda f: f.write(data))
    return fname
