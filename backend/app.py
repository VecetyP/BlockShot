import io
import logging

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")

from . import store  # noqa: E402
from .models import registry  # noqa: E402

app = FastAPI(title="Point Cloud Playground")
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


@app.post("/predict")
def predict(image: UploadFile = File(...), models: list[str] = Form(...), force: bool = Form(False)):
    # Plain def: FastAPI runs it in a worker thread, so History and images keep loading during inference.
    unknown = [m for m in models if m not in registry.MODELS]
    if unknown:
        raise HTTPException(400, f"unknown models: {unknown}")
    data = image.file.read()
    out = []
    for m in models:  # sequential on purpose: one model's activations on the GPU at a time
        rid, meta, cached = store.predict_cached(data, m, force=force, source=image.filename)
        out.append({"id": rid, "cached": cached, "meta": meta})
    return out


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


def _check_fill(rid: str, reach: int | None, order: str | None) -> None:
    _run_or_404(rid)
    if not store.has_depth(store.load_meta(rid)):
        raise HTTPException(400, "run has no depth to work with")
    if order is not None and (order not in store.FILL_ORDERS or reach is None or not 1 <= reach <= 50):
        raise HTTPException(400, f"order must be one of {list(store.FILL_ORDERS)} with reach (% of width) in 1..50")


@app.get("/runs/{rid}/fill.bin")
def fill(rid: str, reach: int = store.DEFAULT_REACH, order: str = "slope"):
    """Gap-fill points behind occluders, up to `reach` % of the image width from each occluding edge:
    uint32 N, f32 xyz[3N], u8 rgb[3N]."""
    _check_fill(rid, reach, order)
    return Response(store.fill_bin(rid, reach, order), media_type="application/octet-stream")


@app.post("/runs/{rid}/offaxis")
def offaxis(rid: str, reach: int | None = None, order: str | None = None):
    """Renders at ±5/15/30°, gaps in magenta; with reach+order also renders with the fill and coverage."""
    _check_fill(rid, reach, order)
    stats = store.offaxis_renders(rid, reach, order)
    for e in stats:
        for k in ("gaps", "filled"):
            if k in e:
                e[k] = f"/runs/{rid}/offaxis/{e[k]}"
    return stats


@app.get("/runs/{rid}/offaxis/{variant}/{name}")
def offaxis_png(rid: str, variant: str, name: str):
    path = (store.RUNS / rid / "offaxis" / variant / name).resolve()
    if not path.is_relative_to((store.RUNS / rid / "offaxis").resolve()) or path.suffix != ".png" or not path.is_file():
        raise HTTPException(404)
    return Response(path.read_bytes(), media_type="image/png")
