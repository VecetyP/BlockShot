import dataclasses
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .analysis import integrability, offaxis
from .analysis.discontinuity import discontinuity, top_tears
from .analysis.reproject import reprojection_error
from . import minecraft, video
from .models import registry
from .models.base import Prediction

RUNS = Path(__file__).resolve().parent.parent / "runs"

# Requests run in a thread pool, and a 4-panel grid fires several requests per run at once. Without
# these locks every one of them recomputed the same analysis in parallel (~1-2 GB each), which ran a
# 32 GB laptop out of memory within a minute; and two predictions shared the GPU.
_gpu_lock = threading.Lock()
_analysis_lock = threading.Lock()  # ponytail: one analysis at a time; per-run locks if this queues up

MAX_ANALYSIS_PIXELS = 768 * 768  # the Poisson solve's LU needs ~1.2 GB here and grows faster than linear


def _atomic(path: Path, write) -> None:
    """write(file) to a temp file, then rename over `path`, so readers never see a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def _write_json(path: Path, obj) -> None:
    _atomic(path, lambda f: f.write(json.dumps(obj, indent=2).encode()))


def decode_image(data: bytes) -> np.ndarray:
    """RGB uint8. Transparent pixels are composited onto white (illusion drawings often ship with alpha)."""
    im = Image.open(io.BytesIO(data)).convert("RGBA")
    bg = Image.new("RGBA", im.size, "white")
    return np.asarray(Image.alpha_composite(bg, im).convert("RGB"))


def run_id(image_sha: str, model: str, version: str) -> str:
    return f"{image_sha[:16]}_{model}_{hashlib.sha256(version.encode()).hexdigest()[:8]}"


def predict_cached(data: bytes, model: str, force: bool = False, source: str | None = None, progress=None,
                   sha: str | None = None) -> tuple[str, dict, bool]:
    """Returns (run_id, meta, cached). `progress(stage)` (optional) is told what's happening. `sha`:
    the original upload's hash, when `data` is a stored copy (recompute)."""
    say = progress or (lambda *a, **k: None)
    sha = sha or hashlib.sha256(data).hexdigest()
    version = registry.cls(model).version
    rid = run_id(sha, model, version)
    d = RUNS / rid
    if not force and (d / "meta.json").exists():
        return rid, load_meta(rid), True
    image = decode_image(data)
    say(f"waiting for the GPU ({model})")
    with _gpu_lock:
        if not force and (d / "meta.json").exists():  # an identical request finished while we waited
            return rid, load_meta(rid), True
        try:
            if not registry.is_loaded(model):
                say(f"loading {model} weights (the first run downloads them)")
            m = registry.get(model)
            say(f"running {model}")
            pred = m.predict(image)
        finally:
            registry.free_cache()
        say(f"saving {model}")
        meta = save(d, rid, image, pred, model=model, version=version, image_sha256=sha, source=source)
    return rid, meta, False


def save(d: Path, rid: str, image: np.ndarray, pred: Prediction, **info) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in dataclasses.asdict(pred).items() if isinstance(v, np.ndarray)}
    arrays |= {f"raw_{k}": v for k, v in pred.raw.items()}
    _atomic(d / "prediction.npz", lambda f: np.savez_compressed(f, **arrays))
    Image.fromarray(image).save(d / "input.png")
    write_ply(d / "points.ply", pred.points[pred.mask], image[pred.mask])
    meta = {
        "id": rid,
        **info,
        "scale_kind": pred.scale_kind,
        "height": image.shape[0],
        "width": image.shape[1],
        "intrinsics": pred.intrinsics.tolist(),
        "has_confidence": pred.confidence is not None,
        "has_normals": pred.normals is not None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **pred.meta,
    }
    _write_json(d / "meta.json", meta)  # written last: its presence marks the run as complete
    return meta


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    v = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz.T
    v["red"], v["green"], v["blue"] = rgb.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(v)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(v.tobytes())


def load_meta(rid: str) -> dict:
    return json.loads((RUNS / rid / "meta.json").read_text())


def load_arrays(rid: str) -> np.lib.npyio.NpzFile:
    return np.load(RUNS / rid / "prediction.npz")


def list_runs() -> list[dict]:
    if not RUNS.exists():
        return []
    metas = [json.loads(p.read_text()) for p in RUNS.glob("*/meta.json") if "." not in p.parent.name]  # not a run being recomputed
    return sorted(metas, key=lambda m: m["created"], reverse=True)


MAX_POINTS = 500_000


def _forget(rid: str) -> None:
    """Drops a run from the in-memory caches (call with _build_lock held)."""
    for k in [k for k in _builds if k[0] == rid]:
        del _builds[k]
    _scenes.pop(rid, None)
    _scene_clouds.pop(rid, None)


def run_up(rid: str) -> list[float]:
    """The run's "up" in its cloud's coordinates (OpenCV axes): scenes are levelled already; image
    runs get it from their structure (minecraft.image_up), once, kept in meta.json."""
    if is_scene(rid):
        return [0.0, -1.0, 0.0]
    meta = load_meta(rid)
    if "up" not in meta:
        a = load_arrays(rid)
        meta["up"] = [round(float(v), 6) for v in minecraft.image_up(a["points"], a["mask"], bool(meta.get("above")))]
        _write_json(RUNS / rid / "meta.json", meta)
    return meta["up"]


def set_above(rid: str, above: bool) -> None:
    """Marks an image run as seen from (nearly) straight above, a top-down map, or not; its "up"
    and builds are worked out again."""
    meta = load_meta(rid)
    if bool(meta.get("above")) == above:
        return
    meta["above"] = above
    meta.pop("up", None)
    _write_json(RUNS / rid / "meta.json", meta)
    with _build_lock:
        _forget(rid)


def delete_run(rid: str) -> None:
    """Removes a run (image prediction or video scene) from disk and from the in-memory caches."""
    d = (RUNS / rid).resolve()
    if d.parent != RUNS.resolve() or not (d / "meta.json").exists():
        raise FileNotFoundError(rid)
    with _build_lock:  # not while a build of it is being made
        _forget(rid)
        shutil.rmtree(d)


def recompute(rid: str, progress=None) -> str:
    """Redoes a run with the current code and returns its (possibly new) id. Image runs: the model
    again on the stored input. Video scenes: everything after frame sampling again on the stored
    frames (the video itself isn't kept), with the recommended geometry. The old run is moved aside
    until the new one is saved, so a failure loses nothing."""
    say = progress or (lambda *a, **k: None)
    meta = load_meta(rid)
    old = RUNS / rid
    aside = RUNS / f"{rid}.recomputing"
    frames = data = None
    with _build_lock:
        _forget(rid)
        if is_scene(rid):
            frames = list(load_scene(rid)["rgb"])
            _scenes.pop(rid, None)
        else:
            data = (old / "input.png").read_bytes()
        old.rename(aside)
    try:
        if frames is not None:
            s = video.Settings(**{**meta.get("settings", {}), "geometry": video.Settings.geometry})
            new = _scene_id(meta["image_sha256"], s)
            kept = {k: meta[k] for k in ("video_fps", "frames_read", "duplicates_dropped") if k in meta}
            say("waiting for the GPU")
            with _gpu_lock:
                try:
                    scene = video.reconstruct_frames(frames, s, {**kept, "frames": len(frames)}, {}, say)
                finally:
                    registry.free_cache()
            say("saving the scene")
            _save_scene(RUNS / new, new, scene, meta.get("source"), meta["image_sha256"])
        else:
            new, _, _ = predict_cached(data, model_of(rid), force=True, source=meta.get("source"), progress=say,
                                       sha=meta["image_sha256"])
            set_above(new, bool(meta.get("above")))
    except BaseException:
        if not old.exists():
            aside.rename(old)
        raise
    shutil.rmtree(aside)
    return new


def has_depth(meta: dict) -> bool:
    return not str(meta.get("depth_source", "")).startswith("none")  # normal-only models store a plane


def _analysis_grid(h: int, w: int, K: np.ndarray):
    """Pixel subsampling for the integrability solve: stride s, the slices, and intrinsics for the
    subsampled grid (its pixel centers land exactly on original pixel centers)."""
    s = max(1, math.ceil(math.sqrt(h * w / MAX_ANALYSIS_PIXELS)))
    o = s // 2
    K2 = K.astype(np.float64).copy()
    K2[0, 0], K2[1, 1] = K[0, 0] / s, K[1, 1] / s
    K2[0, 2], K2[1, 2] = (K[0, 2] - o - 0.5) / s + 0.5, (K[1, 2] - o - 0.5) / s + 0.5
    return s, (slice(o, None, s), slice(o, None, s)), K2


def _upsample(x: np.ndarray, s: int, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour back to (h, w): each subsampled pixel fills its s x s block."""
    if s == 1:
        return x
    y = np.repeat(np.repeat(x, s, 0), s, 1)
    y = np.pad(y, ((0, max(0, h - y.shape[0])), (0, max(0, w - y.shape[1]))), mode="edge")
    return y[:h, :w]


def analysis(rid: str, force: bool = False) -> dict:
    """Summary of every Phase 4 analysis for a run, cached as analysis.json (+ analysis.npz maps)."""
    path = RUNS / rid / "analysis.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    with _analysis_lock:
        if path.exists() and not force:  # computed by another request while we waited
            return json.loads(path.read_text())
        return _compute_analysis(rid, path)


def _compute_analysis(rid: str, path: Path) -> dict:
    a, meta = load_arrays(rid), load_meta(rid)
    pts, mask, K = a["points"], a["mask"], a["intrinsics"]
    h, w = mask.shape
    depth_ok = has_depth(meta)
    s, sl, K2 = _analysis_grid(h, w, K)
    out: dict = {
        "reprojection_px": reprojection_error(pts, K, mask) if depth_ok else None,
        "integrability": {},
        "integrability_stride": s,
    }
    maps = {}
    sources = {
        "native": integrability.from_normals(a["normals"][sl], K2, mask[sl]) if "normals" in a else None,
        "derived": integrability.from_depth(a["depth"][sl], mask[sl]) if depth_ok else None,
    }
    for src, field in sources.items():
        if field is None:
            continue
        t0 = time.perf_counter()
        r = integrability.analyze(field)
        out["integrability"][src] = {
            "residual_fraction": r.residual_fraction,
            "edges": r.n_edges,
            "grad_rms": r.grad_rms,
            "seconds": round(time.perf_counter() - t0, 3),
        }
        # Stored relative to the run's RMS gradient: dimensionless and comparable across runs and models.
        scale = r.grad_rms or 1.0
        maps[f"{src}_curl"] = _upsample(r.curl / scale, s, h, w)
        maps[f"{src}_residual"] = _upsample(r.residual / scale, s, h, w)
    out["tears"] = []
    if depth_ok:
        d = discontinuity(pts, mask)
        out["tears"] = [{"u": u, "v": v, "score": s, "xyz": pts[v, u].tolist()} for u, v, s in top_tears(d)]
    _atomic(RUNS / rid / "analysis.npz", lambda f: np.savez_compressed(f, **maps))
    _write_json(path, out)  # last: marks the analysis as complete
    return out


def map_array(rid: str, kind: str) -> np.ndarray | None:
    a = load_arrays(rid)
    if kind == "discontinuity":
        return discontinuity(a["points"], a["mask"])
    if kind in ("curl", "residual"):
        analysis(rid)
        m = np.load(RUNS / rid / "analysis.npz")
        # Native normals are the research signal; fall back to depth-derived (residual ~0 by design).
        for src in ("native", "derived"):
            if f"{src}_{kind}" in m:
                return m[f"{src}_{kind}"]
        return None
    return a[kind] if kind in a else None


def _sample(mask: np.ndarray, stride: int | None):
    """Pixel-grid subsampling to at most MAX_POINTS valid points; shared so every per-point block lines up."""
    min_stride = max(1, int(np.ceil(np.sqrt(mask.sum() / MAX_POINTS))))
    stride = max(stride or 1, min_stride)
    sl = (slice(None, None, stride), slice(None, None, stride))
    return sl, mask[sl]


def points_bin(rid: str, stride: int | None = None) -> bytes:
    """Little-endian: uint32 N, uint32 has_confidence, float32 xyz[N*3], float32 discontinuity[N],
    float32 confidence[N] (only if has_confidence), uint8 rgb[N*3].
    Floats come first so every block stays 4-byte aligned."""
    if is_scene(rid):
        xyz, rgb = scene_points(rid)
        header = np.array([len(xyz), 0], "<u4")
        return header.tobytes() + xyz.astype("<f4").tobytes() + np.zeros(len(xyz), "<f4").tobytes() + rgb.tobytes()
    a = load_arrays(rid)
    pts, mask = a["points"], a["mask"]
    rgb = np.asarray(Image.open(RUNS / rid / "input.png").convert("RGB"))
    sl, m = _sample(mask, stride)
    blocks = [pts[sl][m].astype("<f4"), discontinuity(pts, mask)[sl][m].astype("<f4")]
    if "confidence" in a:
        blocks.append(a["confidence"][sl][m].astype("<f4"))
    blocks.append(rgb[sl][m].astype(np.uint8))
    header = np.array([m.sum(), "confidence" in a], "<u4")
    return header.tobytes() + b"".join(x.tobytes() for x in blocks)


def field_bin(rid: str, kind: str, stride: int | None = None) -> bytes | None:
    """float32[N] for one per-pixel map, in the same point order as points_bin."""
    x = map_array(rid, kind)
    if x is None:
        return None
    sl, m = _sample(load_arrays(rid)["mask"], stride)
    return x[sl][m].astype("<f4").tobytes()


def offaxis_renders(rid: str, progress=None) -> list[str]:
    """Renders at each yaw in offaxis.YAWS into runs/<id>/offaxis/, cached; returns file names."""
    say = progress or (lambda *a, **k: None)
    d = RUNS / rid / "offaxis"
    names = [f"yaw_{y:+03d}.png" for y in offaxis.YAWS]
    if not all((d / n).exists() for n in names):
        with _analysis_lock:
            d.mkdir(exist_ok=True)
            a = load_arrays(rid)
            rgb = np.asarray(Image.open(RUNS / rid / "input.png").convert("RGB"))
            for i, (y, n) in enumerate(zip(offaxis.YAWS, names)):
                say("rendering off-axis views", i, len(names), f"{y:+d}°")
                img = Image.fromarray(offaxis.render(a["points"], rgb, a["mask"], a["intrinsics"], y))
                _atomic(d / n, lambda f: img.save(f, format="PNG"))
    return names


def model_of(rid: str) -> str:
    """Registry name of the model that made a run (meta["model"] holds the weights' repo id)."""
    return rid.split("_", 1)[1].rsplit("_", 1)[0]


def _rgb(rid: str) -> np.ndarray:
    return np.asarray(Image.open(RUNS / rid / "input.png").convert("RGB"))


_builds: dict = {}  # (rid, mode, value) -> Build; a handful of recent builds (scenes: before view_from)
_build_lock = threading.Lock()  # one build at a time, and never the same build twice
MAX_CACHED_BUILDS = 4


def minecraft_build(rid: str, mode: str, value: float, progress=None, frame: int = 0) -> minecraft.Build:
    """Image runs: viewed from the input camera. Scenes: built once, then viewed from frame
    `frame`'s camera (only the standing spot, yaw, pitch and FOV change between frames)."""
    if is_scene(rid):
        return _scene_view(rid, _base_build(rid, mode, value, progress), frame)
    return _base_build(rid, mode, value, progress)


def _base_build(rid: str, mode: str, value: float, progress=None) -> minecraft.Build:
    key = (rid, mode, float(value))
    if key in _builds:
        return _builds[key]
    say = progress or (lambda *a, **k: None)
    # Poll for the lock so a queued build the browser gave up on (progress raises) leaves the queue.
    while not _build_lock.acquire(timeout=0.25):
        say("waiting for another build")
    try:
        if key not in _builds:
            say("reading points")
            if is_scene(rid):
                b = _scene_build(rid, mode, value, say)
            else:
                a = load_arrays(rid)
                b = minecraft.build_from_image(a["points"], _rgb(rid), a["mask"], a["intrinsics"], mode, value, progress=say, up=np.array(run_up(rid)))
            while len(_builds) >= MAX_CACHED_BUILDS:
                _builds.pop(next(iter(_builds)))
            _builds[key] = b
        return _builds[key]
    finally:
        _build_lock.release()


def minecraft_info(rid: str, mode: str, value: float, progress=None, frame: int = 0) -> dict:
    b = minecraft_build(rid, mode, value, progress, frame)
    counts = np.bincount(b.block, minlength=len(minecraft.BLOCK_NAMES))
    top = [(minecraft.BLOCK_NAMES[i], int(counts[i])) for i in np.argsort(-counts)[:8] if counts[i]]
    _, _, _, lod = minecraft.preview(b) if len(b.ijk) > minecraft.PREVIEW_MAX else (None, None, None, 1)
    metric = load_meta(rid).get("scale_kind") == "metric"
    return {
        "dims": b.dims,
        "blocks": len(b.ijk),
        "block_size": b.block_size,
        "block_size_unit": "m" if metric else "units",
        "budget_limited": b.budget_limited,
        "too_tall": b.dims[1] > minecraft.WORLD_HEIGHT,
        "single_file": math.prod(b.dims) <= minecraft.MAX_SINGLE_CELLS,
        "tiles": len(np.unique(b.ijk[:, [0, 2]] // TILE, axis=0)),
        "fov_deg": round(b.fov_deg, 1),
        "pitch_deg": round(b.pitch_deg, 1),
        "yaw_deg": round(b.yaw_deg, 1),
        "offset": b.offset,
        "preview_lod": lod,
        "data_version": minecraft.DATA_VERSION,
        "top_blocks": top,
    }


TILE = 128


def minecraft_preview_bin(rid: str, mode: str, value: float, frame: int = 0) -> bytes:
    """uint32 N, float32 cube_size, float32 lo[3], float32 axes[9], float32 origin[3],
    int16 ijk[N*3], uint8 rgb[N*3] (block colours).
    Cube (i, j, k) spans lo + [ijk, ijk+1) * cube_size in Minecraft axes; a Minecraft-axes point m is
    origin + axes @ m in the cloud's coordinates (axes row-major, columns = east, up, south). Big
    builds are previewed with merged cubes (cube_size is then a multiple of the block size)."""
    b = minecraft_build(rid, mode, value, frame=frame)
    ijk, rgb, size, _ = minecraft.preview(b)
    axes, _ = minecraft._basis(b.pose)
    head = np.uint32(len(ijk)).tobytes() + np.array([size, *b.lo, *axes.ravel(), *b.pose.position], "<f4").tobytes()
    return head + ijk.astype("<i2").tobytes() + rgb.astype(np.uint8).tobytes()


def _export_name(rid: str, mode: str, value: float, frame: int = 0) -> str:
    stem = Path(str(load_meta(rid).get("source") or rid)).stem
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:40] or "build"
    view = f"_f{frame}" if is_scene(rid) else ""
    return f"{safe}_{model_of(rid)}_{mode}{value:g}{view}".replace(".", "p")


def minecraft_schem(rid: str, mode: str, value: float, frame: int = 0) -> tuple[bytes, str]:
    return minecraft.whole(minecraft_build(rid, mode, value, frame=frame)), _export_name(rid, mode, value, frame) + ".schem"


def minecraft_zip(rid: str, mode: str, value: float, progress=None, frame: int = 0) -> str:
    """Writes the tiles zip into runs/<id>/exports/ and returns its file name."""
    b = minecraft_build(rid, mode, value, progress, frame)
    name = _export_name(rid, mode, value, frame) + ".zip"
    d = RUNS / rid / "exports"
    d.mkdir(exist_ok=True)
    data = minecraft.tiles_zip(b, name[:-4], TILE, progress or (lambda *a, **k: None))
    _atomic(d / name, lambda f: f.write(data))
    return name


# --- video scenes --------------------------------------------------------------------------------

SCENE_FILE = "scene.npz"


def is_scene(rid: str) -> bool:
    return (RUNS / rid / SCENE_FILE).exists()


def _scene_id(sha: str, s: video.Settings) -> str:
    """Scenes are cached by video, settings, model versions and pipeline version."""
    version = json.dumps(dataclasses.asdict(s), sort_keys=True) + registry.cls("vggt").version + registry.cls("moge").version + f"p{video.PIPELINE}"
    return f"{sha[:16]}_scene_{hashlib.sha256(version.encode()).hexdigest()[:8]}"


def process_video(data: bytes, filename: str | None, settings: dict, progress=None) -> tuple[str, dict, bool]:
    """Video -> scene run (cached by video + settings). Returns (run id, meta, cached)."""
    say = progress or (lambda *a, **k: None)
    s = video.Settings(**settings)
    sha = hashlib.sha256(data).hexdigest()
    rid = _scene_id(sha, s)
    d = RUNS / rid
    if (d / "meta.json").exists():
        return rid, load_meta(rid), True
    with tempfile.NamedTemporaryFile(suffix=Path(filename or "video.mp4").suffix or ".mp4", delete=False) as tmp:
        tmp.write(data)
    try:
        say("waiting for the GPU")
        with _gpu_lock:
            try:
                scene = video.reconstruct(tmp.name, s, say)
            finally:
                registry.free_cache()
    finally:
        os.unlink(tmp.name)
    say("saving the scene")
    return rid, _save_scene(d, rid, scene, filename, sha), False


def _save_scene(d: Path, rid: str, sc: dict, source: str | None, sha: str) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    (d / "frames").mkdir(exist_ok=True)
    rgb = sc["rgb"]
    for i, f in enumerate(rgb):
        Image.fromarray(f).save(d / "frames" / f"{i}.jpg", quality=88)
    Image.fromarray(rgb[0]).save(d / "input.png")  # thumbnail, and the default viewpoint
    hud = sc["hud"] if sc["hud"] is not None else np.zeros(rgb.shape[1:3], bool)
    _atomic(d / SCENE_FILE, lambda f: np.savez_compressed(
        f, rgb=rgb, depth=sc["depth"], mask=sc["mask"], K=sc["K"].astype(np.float32), c2w=sc["c2w"].astype(np.float32), hud=hud))
    n, h, w = sc["depth"].shape
    meta = {
        "id": rid,
        "kind": "scene",
        "model": "video scene",
        "source": source,
        "image_sha256": sha,
        "scale_kind": sc["scale_kind"],
        "height": h,
        "width": w,
        "frames": n,
        "intrinsics": sc["K"][0].tolist(),
        "cameras": [{"c2w": c.tolist(), "K": k.tolist()} for c, k in zip(sc["c2w"], sc["K"])],
        "has_confidence": False,
        "has_normals": False,
        "depth_source": "video: VGGT poses + " + ("MoGe depth on VGGT's shape" if sc["stats"]["settings"]["geometry"] == "moge" else "VGGT depth"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **sc["stats"],
    }
    _write_json(d / "meta.json", meta)  # last: marks the scene as complete
    return meta


_scenes: dict = {}  # rid -> loaded arrays (the last couple of scenes)


def load_scene(rid: str) -> dict:
    if rid not in _scenes:
        with np.load(RUNS / rid / SCENE_FILE) as z:
            sc = {k: z[k] for k in z.files}
        while len(_scenes) >= 2:
            _scenes.pop(next(iter(_scenes)))
        _scenes[rid] = sc
    return _scenes[rid]


def _scene_grids(sc: dict) -> list:
    return [(video.world_points(sc["depth"][i], sc["K"][i].astype(np.float64), sc["c2w"][i].astype(np.float64)),
             sc["rgb"][i], sc["mask"][i], sc["c2w"][i, :3, 3].astype(np.float64)) for i in range(len(sc["depth"]))]


_scene_clouds: dict = {}


def scene_points(rid: str) -> tuple[np.ndarray, np.ndarray]:
    """All frames fused into one cloud (scene coordinates = frame 0's camera), averaged on a voxel
    grid sized to give at most MAX_POINTS points."""
    if rid not in _scene_clouds:
        sc = load_scene(rid)
        P = np.concatenate([g[0][g[2]] for g in _scene_grids(sc)])
        C = np.concatenate([g[1][g[2]] for g in _scene_grids(sc)])
        lo, hi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
        keep = ((P >= lo) & (P <= hi)).all(1)
        P, C = P[keep], C[keep]
        s = minecraft._size_for_count(lambda s: P, lo, hi, MAX_POINTS * 0.9)
        dims = minecraft._dims(lo, hi, s)
        cells, inv = np.unique(minecraft._keys(P, lo, s, dims), return_inverse=True)
        n = np.bincount(inv)
        xyz = np.stack([np.bincount(inv, P[:, k]) for k in range(3)], -1) / n[:, None]
        rgb = np.stack([np.bincount(inv, C[:, k]) for k in range(3)], -1) / n[:, None]
        while len(_scene_clouds) >= 2:
            _scene_clouds.pop(next(iter(_scene_clouds)))
        _scene_clouds[rid] = (xyz.astype(np.float32), rgb.round().astype(np.uint8))
    return _scene_clouds[rid]


def _scene_pose(sc: dict, frame: int, levelled: bool) -> tuple[minecraft.Pose, float]:
    """Frame `frame`'s camera and its vertical FOV. "Up" is the scene's -y when it was levelled by its
    floor (video.reconstruct does that), else the cameras' up averaged over all frames."""
    c2w = sc["c2w"].astype(np.float64)
    frame = int(np.clip(frame, 0, len(c2w) - 1))
    up = np.array([0.0, -1.0, 0.0]) if levelled else -c2w[:, :3, 1].mean(0)
    fov = math.degrees(2 * math.atan(sc["depth"].shape[1] / 2 / float(sc["K"][frame][1, 1])))
    return minecraft.Pose(c2w[frame, :3, 3], c2w[frame, :3, 2], up), fov


def _scene_build(rid: str, mode: str, value: float, say) -> minecraft.Build:
    """The scene's blocks, with the grid lined up with frame 0 (any frame works: view_from moves the
    viewer). The space around the eye is cleared per frame by view_from."""
    sc = load_scene(rid)
    pose, fov = _scene_pose(sc, 0, bool(load_meta(rid).get("levelled")))
    return minecraft.build_from_grids(_scene_grids(sc), pose, fov, mode, value, progress=say, clear=False)


def _scene_view(rid: str, b: minecraft.Build, frame: int) -> minecraft.Build:
    return minecraft.view_from(b, *_scene_pose(load_scene(rid), frame, bool(load_meta(rid).get("levelled"))))
