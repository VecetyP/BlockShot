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
async def predict(image: UploadFile = File(...), models: list[str] = Form(...), force: bool = Form(False)):
    unknown = [m for m in models if m not in registry.MODELS]
    if unknown:
        raise HTTPException(400, f"unknown models: {unknown}")
    data = await image.read()
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
    x = store.map_array(rid, kind) if kind in ("depth", "confidence", "discontinuity") else None
    if x is None:
        raise HTTPException(404, f"no {kind} map for {rid}")
    # Display normalization only; stored arrays stay raw.
    x, mask = x.astype(np.float32), store.load_arrays(rid)["mask"]
    vals = x[mask] if mask.any() else x.ravel()
    lo, hi = np.percentile(vals, [1, 99]) if vals.size else (0.0, 1.0)
    img = np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
    img = np.where(mask, (img * 255).astype(np.uint8), 0)
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
