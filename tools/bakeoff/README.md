# Gameplay-footage bake-off

Can any newer model reconstruct frantic first-person game footage well enough to build a "Gameplay
mode" around it? This folder measures that on a clip, without touching the app.

The test clip was first-person Roblox: 35 s, 2560×1440 at 120 fps, with swinging arms, changing HUD
bars, frantic turns, and a death and respawn.

## Result (2026-10-22): no winner

Each cell is neighbour agreement / all-pairs agreement / ghosts, with the player's arms blanked. Agreement
is the share of points two frames agree on within 5%. A ghost is a point that floats in space another frame
saw as empty. For reference, the hallway clip that maps well scores 93% / 85% / 3%.

| window | VGGT (app today) | Pi3X | MapAnything |
|---|---|---|---|
| calm 8 fps 1–3 s | 84 / 63 / 19% | 90 / 57 / 25% | 56 / 44 / 19% |
| mid 8 fps 10–12 s | 81 / 43 / 33% | 88 / 66 / 17% | 67 / 50 / 26% |
| frantic 8 fps 29–31 s | 33 / 14 / 50% | 59 / 27 / 20% | 60 / 29 / 31% |
| running 2 fps 0–8 s | 27 / 16 / 30% | 48 / 23 / 29% | 38 / 29 / 36% |
| respawn 2 fps 17–25 s | 8 / 4 / 15% | 14 / 12 / 18% | 12 / 16 / 24% |
| end 2 fps 28–35 s | 29 / 14 / 14% | 22 / 19 / 26% | 12 / 11 / 33% |
| across respawn 4 fps 12–28 s | 18 / 6 / 20% | 35 / 9 / 25% | 22 / 9 / 16% |

The bar, set before the run, was:
- **Calm windows:** at least 85% neighbours, 60% all pairs, and at most 10% ghosts.
- **2 fps windows:** at least 50% neighbours.

Pi3X is the best of the three, but even its calmest window has 17–25% ghosts, and it loses track
through fast turns and the respawn. The renders in `out/renders/` show the same thing: the floor comes out
doubled into bands and buildings smear, even in the calm stretch.

The obstacles that stay are:
- the character's arms and body, 10–20% of each frame, which move with the camera;
- HUD bars that change, so the static-overlay detector misses them;
- turns of up to 130° between frames 0.5 s apart;
- the teleport.

## Rerunning (e.g. a new clip, or newer models later)

The baseline uses the app's own VGGT code and venv. The contenders need their own venv, kept out of the
app's:

```
uv venv --python 3.11 tools/bakeoff/.venv
uv pip install --python tools/bakeoff/.venv/Scripts/python.exe torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
git clone --depth 1 https://github.com/yyfz/Pi3.git tools/bakeoff/vendor/Pi3
git clone --depth 1 https://github.com/facebookresearch/map-anything.git tools/bakeoff/vendor/map-anything
uv pip install --python tools/bakeoff/.venv/Scripts/python.exe -e tools/bakeoff/vendor/Pi3 -e tools/bakeoff/vendor/map-anything scipy
```

Then run each model on the clip and print the table and verdict:

```
uv run python tools/bakeoff/run.py vggt "<clip.mp4>"
tools/bakeoff/.venv/Scripts/python.exe tools/bakeoff/run.py pi3x "<clip.mp4>"
tools/bakeoff/.venv/Scripts/python.exe tools/bakeoff/run.py mapanything "<clip.mp4>"
python tools/bakeoff/report.py
```

Notes:
- **GPU memory:** stop the BlockShot server first. `run.py` refuses to start with less than 9 GB of free GPU
  memory.
- **Windows and Pi3X:** Pi3X runs in float16 here, because Pi3 forces FlashAttention for bfloat16, which
  Windows PyTorch builds lack.
- **Weights:** they download from Hugging Face on first use. Pi3X and VGGT-1B are non-commercial;
  `map-anything-apache` is Apache 2.0.
- **Outputs:** frames, scores (`out/<model>.json`) and renders go to `out/`. The venv, the cloned code and
  `out/` are gitignored.

## Calm game footage: COLMAP cameras + MoGe depth (2026-10-24/25)

The scripts below are the path measured in [docs/game-maps-research.md](../../docs/game-maps-research.md)
(Test4/Test5, Parkour Legacy at a fixed 120 FOV setting = 114° vertical). Run COLMAP steps in the
bake-off venv (`pycolmap`), the rest with the app's (`uv run python`).

| Step | Script |
|---|---|
| Frames at 4 fps + COLMAP with the camera fixed at a known vertical FOV | `sfm.py <clip> <out dir> 4 114` |
| COLMAP model as text | `pycolmap.Reconstruction(...).write_text(<out dir>/text)` |
| Dense depth per frame (MoGe told the FOV; MapAnything / DA3 given COLMAP's cameras) | `dense_moge.py`, `dense_mapanything.py`, `dense_da3.py` |
| Depth vs COLMAP's well-triangulated points | `score_anchors.py` (also `align_dense.py`, `score_dense.py`) |
| Fit depth per frame to COLMAP's points; cut-based fusion as a scene run | `fuse_scene.py` |
| TSDF averaging and its scores | `tsdf_fuse.py` |
| Build at character scale (+ the walked floor), render | `build_cloud.py` |
| Split a model that breaks into consistent pieces, add them to a World | `split_world.py` |
| Place those pieces from shared feature matches | `place_pieces.py`, after matching the `cross_pairs.txt` that `split_world.py` writes: `pycolmap.match_image_pairs(db, pairing_options=ImportedPairingOptions(match_list_path=...))` |
