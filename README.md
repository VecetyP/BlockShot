# Blockshot

**Turn any image into a Minecraft build that looks exactly like it from one spot.**

Drop in a screenshot, a photo or a painting. A monocular 3D model rebuilds the scene as a point
cloud, Blockshot turns that cloud into blocks, and you get a WorldEdit schematic. Paste it, stand
where the camera was, and the build lines up with the original image. Walk away, and it falls apart
into the façade it really is.

![A Minecraft screenshot, its block build seen from the original camera, and the same build from 30° to the side](docs/minecraft.jpg)

*Left: the input screenshot. Middle: the generated build, seen from the input camera. Right: the same
build from 30° to the side.*

## Quick start

You need [uv](https://docs.astral.sh/uv/), Node.js, and ideally an NVIDIA GPU. It was developed on
an RTX 3080 Laptop (16 GB); everything also runs on CPU, slowly. PyTorch comes from the CUDA 13.0
wheel index, which needs NVIDIA driver 580 or newer.

```bash
uv sync
```

**Windows:** double-click **`run.bat`**. It starts the backend (:8000) and the web app (:5173) in two
windows and opens http://localhost:5173. Close both windows to stop.

**Anywhere else:**

```bash
uv run uvicorn backend.app:app --port 8000
cd frontend && corepack pnpm install && corepack pnpm dev   # http://localhost:5173
```

Model weights download from Hugging Face on first use.

## Make a build

1. Drop an image on the drop zone, keep **moge** ticked (recommended), and press **Run**.
2. In the **Minecraft build** bar under the 3D view, pick a **Size**: blocks along the build's
   longest side, 32–256.
3. Tick **Show as blocks** to preview the build in 3D, and press **Snap to input camera** to see it
   from the spot where it matches.
4. Press **Download .schem** on the run's card.

Then, in Minecraft with [WorldEdit](https://enginehub.org/worldedit):

1. Put the file in `.minecraft/config/worldedit/schematics/` (singleplayer mod) or the server's
   `plugins/WorldEdit/schematics/`.
2. Stand where the picture should be seen from, and face south: `/tp @s ~ ~ ~ 0 0`.
3. `//schem load <file name>`, then `//paste`. The paste is offset so the image's camera lands at
   your eyes.
4. Set **Options → FOV** to the number on the card (the camera's vertical field of view), turn
   **FOV Effects** off, and press **F1** to hide the HUD.

Schematics use Minecraft 1.16.5 block data and load in 1.16 and every newer version.

## How a build is made

`backend/minecraft.py`:

1. **Point cloud.** The chosen model predicts a 3D point for every pixel, in the camera's frame.
2. **Axes.** The cloud is turned into Minecraft's axes with the camera facing south (+z), so the
   image's right side goes west.
3. **Blocks.** Points are snapped to a grid sized so the longest side has *Size* blocks. The
   outermost 0.5% of points on each axis are ignored, so stray points don't shrink the build.
4. **Colours.** Each block's colour is the mean of its points in linear light. The block placed is
   the palette entry closest in CIELAB, a perceptual colour space.
5. **Schematic.** A Sponge v2 `.schem` with WorldEdit's paste offset set from the camera position,
   and the vertical FOV from the model's estimated camera.

The palette (`backend/minecraft_palette.json`) holds 107 blocks: all concrete, wool, terracotta and
planks, plus stones, bricks, storage blocks and others that look the same on every face and don't
fall, glow, or let light through. Each entry is the mean colour of the block's texture, measured from
a Minecraft jar by `tools/minecraft_palette.py`. Only those numbers are in this repo, no game files.

**What to expect.** A build is a single-view sculpture: surfaces are one block thick, and nothing
exists behind what the camera saw. The image's own lighting is baked into the block choice, and
Minecraft shades block faces again, so faces can come out slightly darker in game.

## Models

| Model | Weights | Output | Licence | VRAM (weights / peak) |
|---|---|---|---|---|
| **`moge`** · *recommended* | Ruicheng/moge-3-vitl (MoGe 3.0.0) | metric point map, camera intrinsics, mask, normals | MIT (DINOv2 parts Apache-2.0) | 1.4 / 2.4 GB |
| `depth_anything` | Depth-Anything-V2-Small-hf (`DEPTH_ANYTHING_SIZE=Base\|Large`) | relative depth; assumes a 60° camera | Small: Apache-2.0; Base/Large: CC-BY-NC-4.0 | 0.1 / 0.2 GB |
| `vggt` | facebook/VGGT-1B, single image | depth, confidence, intrinsics | weights non-commercial | 4.7 / 7.0 GB |
| `marigold_normals` | prs-eth/marigold-normals-v1-1 | surface normals only (no depth, so no build) | CreativeML Open RAIL++-M | 2.4 / 3.8 GB |

MoGe gives the best-looking builds: it estimates the real camera, so the FOV and proportions come out
right, and it masks out sky and empty background. All four models loaded together use about 10.6 GB
of VRAM.

Notes:

- Depth Anything predicts disparity; `depth = 1/disparity`, and the raw disparity is kept.
- MoGe's refinement stage needs CUDA (Triton kernels). On CPU it runs without refinement, and that
  setting is part of the run's cache key.
- VGGT's depth confidence is stored raw and mapped to `1 − 1/conf` for display.
- `vggt`'s `numpy<2` pin is overridden in `pyproject.toml`; it runs fine on numpy 2.

## Also in the box: a single-image 3D lab

Blockshot grew out of a playground for studying how these models handle images, especially optical
illusions and impossible figures. All of that is still here.

![comparison view](docs/comparison.jpg)

- **Exact camera.** The 3D view projects through each run's own intrinsics, so **Snap to input
  camera** reproduces the input pixel for pixel.
- **Comparison grid.** Tick up to four runs in History. Orbiting one panel orbits all of them;
  poses are shared in units of each cloud's median depth, so metric and relative models stay aligned.
- **Navigation.** Drag to orbit, right-drag to pan, scroll to zoom. **W/A/S/D** fly, **Q/E**
  down/up, **Shift** faster, with sensitivity sliders for each.
- **Colour modes:** RGB, depth, confidence, discontinuity, curl, residual.
- **2D strip:** input, depth, confidence, normals and analysis maps. Click any image to enlarge it.
- **Integrability** (`backend/analysis/integrability.py`): how much of a model's predicted normal
  field no real surface could produce.
  - It uses a Helmholtz–Hodge split (a Poisson solve) on the perspective-exact gradient
    `∇log z = −(nₓ, n_y)/(n·r)`, where r is the pixel's viewing ray.
  - Depth-derived gradients score 0 by construction.
  - A synthetic sphere scores 0.05%, and a hand-built Penrose-triangle normal field scores 22%,
    spread around the loop.
- **Tears** (`discontinuity.py`): depth jumps scaled by distance, with the top 20 marked in 3D and on
  the image.
- **Off-axis renders** (`offaxis.py`): the cloud re-rendered at ±5°, ±15° and ±30°, to watch an
  illusion fall apart as you leave its viewpoint.
- **Reprojection check** (`reproject.py`): 0.000 px for every depth model.

`samples/generate.py` draws a Penrose triangle and a Necker cube; `samples/` also has a few rendered
3D figures as controls. A CLI writes runs too:

```bash
uv run python -m backend.cli predict samples/penrose_triangle.png --model moge
```

Each run is cached in `runs/<image sha>_<model>_<version hash>/` as `prediction.npz`, `points.ply`
(RGB, camera frame: +X right, +Y down, +Z forward), `input.png` and `meta.json`.

## API

| Route | |
|---|---|
| `GET /models` · `POST /models/unload?name=` | models, load status, device · free GPU memory |
| `POST /predict` | multipart `image`, `models` (repeatable), `force` |
| `GET /runs` · `GET /runs/{id}` | run metadata |
| `GET /runs/{id}/minecraft.json?size=` | build summary: size, block count, FOV, paste offset |
| `GET /runs/{id}/minecraft.schem?size=` | Sponge v2 schematic |
| `GET /runs/{id}/minecraft.bin?size=` | block preview for the viewer |
| `GET /runs/{id}/points.bin?stride=` | point cloud for the viewer (≤ 500k points) |
| `GET /runs/{id}/maps/{kind}.png` | depth, confidence, normals, discontinuity, curl, residual |
| `GET /runs/{id}/analysis` · `GET /runs/{id}/field/{curl,residual}.bin` | integrability, reprojection, tears · per-point fields |
| `POST /runs/{id}/offaxis` | ±5/15/30° renders |

## Development

```bash
uv run pytest                                            # model tests download weights on first run
uv run python tools/readme_figure.py <run id> [size]     # regenerate docs/minecraft.jpg
uv run python tools/minecraft_palette.py <minecraft.jar> # re-measure the block palette
```

Minecraft is a trademark of Mojang Studios. Blockshot is not affiliated with Mojang or Microsoft.
