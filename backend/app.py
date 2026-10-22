import io
import json
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")

from . import jobs, minecraft, store  # noqa: E402
from .models import registry  # noqa: E402

app = FastAPI(title="BlockShot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _run_or_404(rid: str) -> None:
    if not (store.RUNS / rid / "meta.json").exists():
        raise HTTPException(404, f"no run {rid}")


def _image_run(rid: str) -> None:
    """Per-image analysis tools don't apply to video scenes."""
    _run_or_404(rid)
    if store.is_scene(rid):
        raise HTTPException(400, "not available for video scenes")


_STARTED = time.time()


@app.get("/health")
def health():
    """`stale`: the backend's code changed on disk after this server started (it needs a restart)."""
    return {"stale": any(p.stat().st_mtime > _STARTED for p in Path(__file__).parent.rglob("*.py"))}


@app.get("/models")
def models():
    return registry.status()


@app.post("/models/unload")
def unload(name: str | None = None):
    registry.unload(name)
    return registry.status()


def _job(kind: str, fn) -> dict:
    return {"job_id": jobs.submit(kind, fn).id}


@app.get("/jobs/{job_id}")
def job(job_id: str):
    """Progress of a background job: state, stage, done/total, message; result once done."""
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, f"no job {job_id}")
    return jobs.as_dict(j)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    jobs.cancel(job_id)
    return {}


@app.post("/predict")
def predict(image: UploadFile = File(...), models: list[str] = Form(...), force: bool = Form(False), above: bool = Form(False),
            truth: str | None = Form(None)):
    """Starts a job; its result is [{id, cached, meta}] per model. `above`: the picture looks (nearly)
    straight down, a top-down map; also applied to runs that were already cached."""
    unknown = [m for m in models if m not in registry.MODELS]
    if unknown:
        raise HTTPException(400, f"unknown models: {unknown}")
    data, source = image.file.read(), image.filename
    truth_obj = _parse_truth(truth)

    def work(progress):
        out = []
        for i, m in enumerate(models):  # sequential on purpose: one model's activations on the GPU at a time
            rid, meta, cached = store.predict_cached(
                data, m, force=force, source=source,
                progress=lambda stage, *_: progress(stage, i, len(models)),
            )
            store.set_above(rid, above)
            if truth_obj:
                store.attach_truth(rid, truth_obj)
            out.append({"id": rid, "cached": cached, "meta": store.load_meta(rid)})
        return out

    return _job("predict", work)


@app.post("/video")
def process_video(
    video: UploadFile = File(...),
    fps: float = Form(2.0),
    max_frames: int = Form(60),
    max_side: int = Form(640),
    hud: bool = Form(True),
    geometry: str = Form("vggt"),
    fov_deg: float | None = Form(None),
    truth: str | None = Form(None),
):
    """Starts a job reconstructing one scene from a video; its result is {id, cached, meta}."""
    if not (0.2 <= fps <= 15 and 2 <= max_frames <= 300 and 256 <= max_side <= 1280 and geometry in ("moge", "vggt")):
        raise HTTPException(400, "fps 0.2-15, max_frames 2-300, max_side 256-1280, geometry moge|vggt")
    if fov_deg is not None and not 20 <= fov_deg <= 150:
        raise HTTPException(400, "fov_deg (vertical field of view) 20-150")
    data, source = video.file.read(), video.filename
    settings = {"fps": fps, "max_frames": max_frames, "max_side": max_side, "hud": hud, "geometry": geometry, "fov_deg": fov_deg}
    truth_obj = _parse_truth(truth)

    def work(progress):
        rid, meta, cached = store.process_video(data, source, settings, progress)
        if truth_obj:
            store.attach_truth(rid, truth_obj)
            meta = store.load_meta(rid)
        return {"id": rid, "cached": cached, "meta": meta}

    return _job("video", work)


def _parse_truth(truth: str | None) -> dict | None:
    """A playground recording's ground truth (see backend/playground.py), checked for the basics."""
    if not truth:
        return None
    try:
        t = json.loads(truth)
        assert t["frames"] and all(len(f["p"]) == 3 and len(f["q"]) == 4 for f in t["frames"])
        assert isinstance(t["objects"], list) and t["fov_deg"] > 0
    except (ValueError, KeyError, TypeError, AssertionError):
        raise HTTPException(400, "truth: expected the playground's JSON (frames with p and q, objects, fov_deg)")
    return t


@app.get("/runs/{rid}/score")
def run_score(rid: str):
    """A playground run's scores against its ground truth."""
    _run_or_404(rid)
    if not store.has_truth(rid):
        raise HTTPException(404, "this run has no ground truth (only playground recordings do)")
    try:
        return store.score(rid)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/runs/{rid}/frames/{i}.jpg")
def frame_jpg(rid: str, i: int):
    path = store.RUNS / rid / "frames" / f"{i}.jpg"
    if not path.is_file():
        raise HTTPException(404)
    return Response(path.read_bytes(), media_type="image/jpeg")


@app.get("/runs")
def runs():
    return store.list_runs()


@app.get("/runs/{rid}")
def run(rid: str):
    """Run metadata, plus "up": the direction the 3D view and the builds treat as up."""
    _run_or_404(rid)
    return {**store.load_meta(rid), "up": store.run_up(rid)}


@app.delete("/runs/{rid}")
def delete_run(rid: str):
    try:
        store.delete_run(rid)
    except FileNotFoundError:
        raise HTTPException(404, f"no run {rid}")
    return {}


@app.post("/runs/{rid}/recompute")
def recompute(rid: str):
    """Starts a job redoing a run with the current code; its result is the run's (possibly new) id."""
    _run_or_404(rid)
    return _job("recompute", lambda progress: store.recompute(rid, progress))


@app.get("/runs/{rid}/points.bin")
def points(rid: str, stride: int | None = None):
    _run_or_404(rid)
    return Response(store.points_bin(rid, stride), media_type="application/octet-stream")


@app.get("/runs/{rid}/input.png")
def input_png(rid: str):
    _run_or_404(rid)
    return Response((store.RUNS / rid / "input.png").read_bytes(), media_type="image/png")


@app.get("/runs/{rid}/maps/{kind}.png")
def map_png(rid: str, kind: str):
    _image_run(rid)
    if kind == "normals":
        n = store.map_array(rid, "normals")
        if n is None:
            raise HTTPException(404, f"no normals for {rid}")
        # Conventional normal-map colors: R=right, G=up, B=toward viewer (stored normals are OpenCV).
        rgb = ((n * np.array([1, -1, -1]) + 1) * 127.5).clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")
    x = store.map_array(rid, kind) if kind in ("depth", "confidence", "discontinuity", "curl", "residual") else None
    if x is None:
        raise HTTPException(404, f"no {kind} map for {rid}")
    # Display normalization only; stored arrays stay raw.
    x, mask = x.astype(np.float32), store.load_arrays(rid)["mask"]
    if kind in ("curl", "residual"):
        # Already relative to the RMS gradient: fixed scale, so ~0 shows as 0 and runs compare directly.
        x, (lo, hi) = np.abs(x), (0.0, 1.0)
    else:
        vals = x[mask] if mask.any() else x.ravel()
        lo, hi = np.percentile(vals, [1, 99]) if vals.size else (0.0, 1.0)
    img = np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
    img = np.where(mask, (img * 255).astype(np.uint8), 0)
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/runs/{rid}/analysis")
def analysis(rid: str, force: bool = False):
    _image_run(rid)
    return store.analysis(rid, force)


@app.get("/runs/{rid}/field/{kind}.bin")
def field(rid: str, kind: str, stride: int | None = None):
    """float32 per point, same order as points.bin."""
    _image_run(rid)
    b = store.field_bin(rid, kind, stride) if kind in ("curl", "residual") else None
    if b is None:
        raise HTTPException(404, f"no {kind} field for {rid}")
    return Response(b, media_type="application/octet-stream")


def _check_minecraft(rid: str, mode: str, value: float) -> None:
    _run_or_404(rid)
    if not store.has_depth(store.load_meta(rid)):
        raise HTTPException(400, "this model has no depth to build from")
    if mode not in minecraft.MODES:
        raise HTTPException(400, f"mode must be one of {list(minecraft.MODES)}")
    lo, hi = minecraft.MODES[mode]
    if not lo <= value <= hi:
        raise HTTPException(400, f"value for {mode} must be {lo}..{hi}")


@app.post("/runs/{rid}/minecraft/build")
def minecraft_build(rid: str, mode: str = "auto", value: float = 300, frame: int = 0):
    """Starts a job that builds the blocks; its result is the build summary."""
    _check_minecraft(rid, mode, value)
    return _job("minecraft", lambda progress: store.minecraft_info(rid, mode, value, progress, frame))


@app.get("/runs/{rid}/minecraft.json")
def minecraft_info(rid: str, mode: str = "auto", value: float = 300, frame: int = 0):
    """Build summary: dimensions, block count, block size, FOV, pitch, paste offset, tiles."""
    _check_minecraft(rid, mode, value)
    return store.minecraft_info(rid, mode, value, frame=frame)


@app.get("/runs/{rid}/minecraft.bin")
def minecraft_preview(rid: str, mode: str = "auto", value: float = 300, frame: int = 0):
    _check_minecraft(rid, mode, value)
    return Response(store.minecraft_preview_bin(rid, mode, value, frame), media_type="application/octet-stream")


@app.get("/runs/{rid}/minecraft.schem")
def minecraft_schem(rid: str, mode: str = "auto", value: float = 300, frame: int = 0):
    """One Sponge v2 schematic for WorldEdit; //paste puts the input camera at your eyes."""
    _check_minecraft(rid, mode, value)
    try:
        data, filename = store.minecraft_schem(rid, mode, value, frame)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(data, media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.post("/runs/{rid}/minecraft/tiles")
def minecraft_tiles(rid: str, mode: str = "auto", value: float = 300, frame: int = 0):
    """Starts a job writing the build as a zip of tiles; its result is the download URL."""
    _check_minecraft(rid, mode, value)
    return _job("tiles", lambda progress: f"/runs/{rid}/exports/{store.minecraft_zip(rid, mode, value, progress, frame)}")


@app.get("/runs/{rid}/exports/{name}")
def export_file(rid: str, name: str):
    path = (store.RUNS / rid / "exports" / name).resolve()
    if not path.is_relative_to((store.RUNS / rid / "exports").resolve()) or not path.is_file():
        raise HTTPException(404)
    return Response(path.read_bytes(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/runs/{rid}/offaxis")
def offaxis(rid: str):
    """Starts a job; its result is the URLs of the renders."""
    _image_run(rid)
    if not store.has_depth(store.load_meta(rid)):
        raise HTTPException(400, "run has no depth to re-render")
    return _job("offaxis", lambda progress: [f"/runs/{rid}/offaxis/{n}" for n in store.offaxis_renders(rid, progress)])


@app.get("/runs/{rid}/offaxis/{name}")
def offaxis_png(rid: str, name: str):
    path = (store.RUNS / rid / "offaxis" / name).resolve()
    if not path.is_relative_to(store.RUNS.resolve()) or path.parent.name != "offaxis" or not path.is_file():
        raise HTTPException(404)
    return Response(path.read_bytes(), media_type="image/png")
