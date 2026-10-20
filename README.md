# Image → 3D Point Cloud Playground

Run single-image depth/geometry models on images (especially impossible figures), cache the
outputs in one uniform format, and inspect them.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Torch is pulled from the CUDA 13.0 wheel index (NVIDIA driver 580+);
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

Viewer (needs Node and pnpm; `corepack pnpm` works without a global install):

```bash
cd frontend && corepack pnpm install && corepack pnpm dev   # http://localhost:5173, proxies /api to :8000
```

The viewer renders with a projection built from the run's intrinsics, so **Snap to input camera**
(camera at the origin looking down +Z) reproduces the input image pixel-for-pixel. Color modes:
RGB, depth, confidence (when the model has it), discontinuity.

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
| `GET /runs/{id}/points.bin?stride=` | `uint32 N, has_conf`, `f32 xyz[3N]`, `f32 discontinuity[N]`, `f32 conf[N]` (if any), `u8 rgb[3N]`; ≤500k points |
| `GET /runs/{id}/maps/{depth,confidence,discontinuity}.png` | display-normalized 2D maps |

## Models

![comparison view](docs/comparison.jpg)

Tick up to four runs in History (or run several models at once) for the comparison grid. Orbiting
one panel orbits all of them; poses are shared in units of each cloud's median depth, so metric and
relative models stay aligned. Each panel has its own Snap.

VRAM measured on an RTX 3080 Laptop (16 GB), CUDA. All four resident ≈ 10.6 GB.

| Name | Weights | License | Output | Weights / peak VRAM |
|---|---|---|---|---|
| `depth_anything` | Depth-Anything-V2-Small-hf (`DEPTH_ANYTHING_SIZE=Base\|Large`) | Small: Apache-2.0; Base/Large: CC-BY-NC-4.0 | affine-invariant disparity, assumed 60° FOV | 0.1 / 0.2 GB |
| `moge` | Ruicheng/moge-3-vitl (MoGe 3.0.0) | MIT (DINOv2 parts Apache-2.0) | metric pointmap, intrinsics, mask, normals | 1.4 / 2.4 GB |
| `marigold_normals` | prs-eth/marigold-normals-v1-1 (4 steps, ensemble 3) | CreativeML Open RAIL++-M | normals + uncertainty only (points are a placeholder plane) | 2.4 / 3.8 GB |
| `vggt` | facebook/VGGT-1B, single image | weights non-commercial; code see repo | depth, confidence, intrinsics (camera head) | 4.7 / 7.0 GB |

Notes:
- Depth Anything predicts disparity. `depth = 1/disparity`; raw disparity is kept as `raw_disparity`.
- MoGe-3's refinement stage needs CUDA (Triton kernels via flex-gemm). On CPU it runs with
  `refine_steps=0`; the setting is part of the cache key and `meta.refine_steps`.
- Marigold normals are converted from (right, up, toward viewer) to the OpenCV camera frame.
- VGGT's `depth_conf` (≥ 1, unbounded) is stored raw as `raw_depth_conf`; `confidence = 1 - 1/depth_conf`.
- PyTorch comes from the CUDA 13.0 wheel index (MoGe requires it); `vggt`'s `numpy<2` pin is overridden.
- DUSt3R/MASt3R (optional in the plan) are not included.
