# Image → 3D Point Cloud Playground

Run single-image depth/geometry models on images (especially impossible figures), cache the
outputs in one uniform format, and inspect them.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Torch is pulled from the CUDA 12.4 wheel index;
the device is picked at startup (CUDA → MPS → CPU).

```bash
uv sync
uv run python samples/generate.py   # draws penrose_triangle.png, necker_cube.png
```

## Use

```bash
uv run python -m backend.cli predict samples/penrose_triangle.png --model depth_anything
uv run uvicorn backend.app:app --port 8000
uv run pytest
```

Each run lands in `runs/<image sha>_<model>_<version hash>/` with `prediction.npz`,
`points.ply` (RGB-colored, camera frame: +X right, +Y down, +Z forward), `input.png` and
`meta.json`. Re-running the same image and model hits the cache; pass `--force` to recompute.

## API

| Route | |
|---|---|
| `GET /models` | models, load status, device, scale kind |
| `POST /models/unload?name=` | free a model (or all) |
| `POST /predict` | multipart `image`, `models` (repeatable), `force` |
| `GET /runs`, `GET /runs/{id}` | run metadata |
| `GET /runs/{id}/points.bin?stride=` | `uint32 N`, `float32 xyz[N*3]`, `uint8 rgb[N*3]`, ≤500k points |
| `GET /runs/{id}/maps/{depth,confidence}.png` | display-normalized 2D maps |

## Models

| Name | Weights | License | Scale | VRAM |
|---|---|---|---|---|
| `depth_anything` | Depth-Anything-V2-Small-hf (set `DEPTH_ANYTHING_SIZE=Base\|Large`) | Small: Apache-2.0, Base/Large: CC-BY-NC-4.0 | affine-invariant disparity | <1 GB (Small) |

Depth Anything predicts disparity. `depth = 1/disparity` (raw disparity is kept in
`prediction.npz` as `raw_disparity`); intrinsics assume a 60° horizontal FOV.
