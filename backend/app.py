import io
import logging

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")

from . import jobs, store  # noqa: E402
from .models import registry  # noqa: E402

app = FastAPI(title="BlockShot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _run_or_404(rid: str) -> None:
    if not (store.RUNS / rid / "meta.json").exists():
        raise HTTPException(404, f"no run {rid}")


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


@app.post("/predict")
def predict(image: UploadFile = File(...), models: list[str] = Form(...), force: bool = Form(False)):
    """Starts a job; its result is [{id, cached, meta}] per model."""
    unknown = [m for m in models if m not in registry.MODELS]
    if unknown:
        raise HTTPException(400, f"unknown models: {unknown}")
    data, source = image.file.read(), image.filename

    def work(progress):
        out = []
        for i, m in enumerate(models):  # sequential on purpose: one model's activations on the GPU at a time
            rid, meta, cached = store.predict_cached(
                data, m, force=force, source=source,
                progress=lambda stage, *_: progress(stage, i, len(models)),
            )
            out.append({"id": rid, "cached": cached, "meta": meta})
        return out

    return _job("predict", work)


@app.get("/runs")
def runs():
    return store.list_runs()


@app.get("/runs/{rid}")
def run(rid: str):
    _run_or_404(rid)
    return store.load_meta(rid)


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
    _run_or_404(rid)
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
    _run_or_404(rid)
    return store.analysis(rid, force)


@app.get("/runs/{rid}/field/{kind}.bin")
def field(rid: str, kind: str, stride: int | None = None):
    """float32 per point, same order as points.bin."""
    _run_or_404(rid)
    b = store.field_bin(rid, kind, stride) if kind in ("curl", "residual") else None
    if b is None:
        raise HTTPException(404, f"no {kind} field for {rid}")
    return Response(b, media_type="application/octet-stream")


def _check_minecraft(rid: str, size: int) -> None:
    _run_or_404(rid)
    if not store.has_depth(store.load_meta(rid)):
        raise HTTPException(400, "this model has no depth to build from")
    if not 16 <= size <= 512:
        raise HTTPException(400, "size must be 16..512 blocks")


@app.get("/runs/{rid}/minecraft.json")
def minecraft_info(rid: str, size: int = 128):
    """Build summary: dimensions, block count, FOV and paste offset."""
    _check_minecraft(rid, size)
    return store.minecraft_info(rid, size)


@app.get("/runs/{rid}/minecraft.bin")
def minecraft_preview(rid: str, size: int = 128):
    _check_minecraft(rid, size)
    return Response(store.minecraft_preview_bin(rid, size), media_type="application/octet-stream")


@app.get("/runs/{rid}/minecraft.schem")
def minecraft_schem(rid: str, size: int = 128):
    """Sponge v2 schematic for WorldEdit; //paste puts the input camera at your eyes."""
    _check_minecraft(rid, size)
    data, filename = store.minecraft_schem(rid, size)
    return Response(data, media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.post("/runs/{rid}/offaxis")
def offaxis(rid: str):
    """Starts a job; its result is the URLs of the renders."""
    _run_or_404(rid)
    if not store.has_depth(store.load_meta(rid)):
        raise HTTPException(400, "run has no depth to re-render")
    return _job("offaxis", lambda progress: [f"/runs/{rid}/offaxis/{n}" for n in store.offaxis_renders(rid, progress)])


@app.get("/runs/{rid}/offaxis/{name}")
def offaxis_png(rid: str, name: str):
    path = (store.RUNS / rid / "offaxis" / name).resolve()
    if not path.is_relative_to(store.RUNS.resolve()) or path.parent.name != "offaxis" or not path.is_file():
        raise HTTPException(404)
    return Response(path.read_bytes(), media_type="image/png")
