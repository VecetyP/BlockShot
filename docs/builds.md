# How a build is made, and the models

[← README](../README.md)

## Build options

In the **Minecraft build** bar under the 3D view, choose how big a block is:
- **Auto** (default): about *N* thousand blocks in total, 50k–4M. Detail follows the scene, so a
  city and a room both get a few hundred thousand blocks.
- **Real size**: one block = *x* metres. Metric models (MoGe, video scenes) only.
- **Fit longest side**: the longest side is *N* blocks.
- **Character size**: the camera's height above the floor becomes a player's eye height (1.62
  blocks), times a size factor. A building ten people long comes out ten players long, whatever the
  model thinks a metre is: a game's character (a Roblox R6 eye is about 4.5 studs, so 1 block ≈ 2.8
  studs) or a person filming. Needs the floor in view now and then.

**Shape**: *As seen* (default) places blocks only where surfaces were seen. *Boxes (buildings)* turns
the grid to run along the walls and rebuilds every building as a hollow prism from the ground to its
roof, so hidden back walls exist too ([backend/boxify.py](../backend/boxify.py)). Shell blocks take
the colour of the nearest seen block (windows, signs); things on roofs narrower than 5 blocks and
everything outside buildings stay as seen. Gaps between buildings are never bridged. Slanted roofs
and spires come out stepped.

A budget of 4M blocks enlarges the blocks, with a warning, if a setting would go past it. **Show as
blocks** previews the build in 3D; previews above 400k blocks merge blocks into bigger cubes, and
downloads are always full detail. Builds wider or longer than 128 blocks download as a **zip of
128 × 128 tiles** with a `paste-commands.txt`.

Schematics use Minecraft 1.20.1 block data (DataVersion 3465) and load in 1.20 and newer. A build
taller than 384 blocks (the world's build height) gets a warning.

## The pipeline

`backend/minecraft.py`:

1. **Point cloud.** The chosen model predicts a 3D point for every pixel, in the camera's frame.
2. **Axes.** The cloud is turned into Minecraft's axes with the camera facing south (+z), so the
   image's right side goes west. "Up" comes from the picture's structure: floors, roofs and walls
   are square to each other, so the build finds those three axes from the surface directions and
   takes the one nearest the camera's own up. A tilted photo, or a city shot 40 degrees from above,
   still builds level, and the card gives the pitch to look down at it. The 3D view is levelled the
   same way. For a map shot straight down, tick **Top-down map** next to Run (straight down, the
   ground's axis and a wall's are a toss-up); it is kept with the picture's runs. Video scenes are
   levelled when they're made (see [video](video.md)) and built once: choosing another frame only
   changes where you stand, your yaw and pitch, and the FOV.
3. **Blocks.** The surface between neighbouring pixels is filled in, except across depth jumps, and
   snapped to a grid whose spacing comes from the block-size setting. The outermost 0.5% of points on
   each axis are ignored, so stray points don't shrink the build, and blocks within 4 of the eye are
   dropped so they don't fill the view in game.
4. **Colours.** Each block's colour is the mean of its points in linear light. The block placed is
   the palette entry closest in CIELAB, a perceptual colour space.
5. **Schematic.** A Sponge v2 `.schem` with WorldEdit's paste offset set from the camera position,
   and the vertical FOV from the model's estimated camera. Tiles each carry the global offset plus
   their own origin.

The palette (`backend/minecraft_palette.json`) holds 153 blocks from 1.20.1: all concrete, wool,
terracotta and planks, plus deepslate, tuff, mud bricks, waxed copper, stones, bricks, storage blocks
and others that look the same on every face and don't fall, glow, or let light through. Each entry is
the mean colour of the block's texture, measured from a Minecraft jar by `tools/minecraft_palette.py`.
Only those numbers are in this repo, no game files.

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
