# Game map → Minecraft: what we know, what others do, and a plan

[← README](../README.md) · written 2026-10-24, after Test1/Test2 (Parkour Legacy) failed

Goal: the Parkour Legacy city in Minecraft, at character scale, mostly box buildings with props,
recorded over several sessions and adjustable by hand. This page starts from what we have measured,
then what other people use for the same problem, then a plan with decisions made up front.

## 1. What we know (measured, in this repo)

| Fact | Evidence |
|---|---|
| Our pipeline is fine on footage it was built for | Hallway/IRL clips lose 3–10% of points to the ghost filter; playground recordings score 0.7° / 0.4 m camera error |
| No code regression | Three playground recordings replayed through today's code give the scores they got when recorded (0.68°/0.36 m, 0.83°/0.57 m, 5.25°/8.25 m) |
| VGGT loses track on Parkour footage | Test1/Test2 lose 45–57% of points; VGGT's scale wanders 2.7–6.9× across a clip (1.1–1.6× on good clips) |
| Settings don't fix it | 6 fps, entered 70° FOV, chunked VGGT, MoGe geometry, grey sky: best 42% lost |
| Short dense pieces half-fix it | Test1 in 2.5 s pieces at 8 fps: 4 of 9 pieces clean (3–18%); the open-rooftop stretch (3–11 s) fails even so |
| Failure is in what's on screen, not only speed | An 8 fps stretch with 0.17° median turn per frame still failed (38% neighbour agreement); camera steps shrank 10× at constant running speed |
| VGGT's FOV guess jumps between pieces | 52.6°–73.1° vertical across Test1's pieces |
| Other models weren't better on Roblox | `tools/bakeoff`: Pi3X and MapAnything also failed on gameplay (17–33% ghosts in calm windows) |
| MoGe single images look good | Image runs of Roblox screenshots build well (README tower) |

What's on screen in the failing parts: half the frame is a plain lit floor, a third is night sky
crossed by zip-line cables, windows repeat, and the camera mostly moves straight forward
(little parallax).

### Step 0 results (Test4: calm walk around one rooftop, FOV setting 120)

| Fact | Evidence |
|---|---|
| Dynamic FOV stops at the max setting | User check in game |
| The real FOV is ~114° vertical / ~140° horizontal | Measured from the footage (rotation between frames); COLMAP self-calibration says 116° |
| VGGT assumed 66° vertical | Its focal length was 2.4× off; our "Vertical FOV" field only relabels the camera after VGGT ran |
| VGGT still fails on calm footage | 50–74% of points dropped, full frames or normal-lens centre crops |
| The footage is trackable with classic features | SIFT: median 196 (min 72) geometric matches between frames 0.5 s apart |
| **COLMAP solves the cameras** | Fixed camera at 114°: 109/109 frames in one model, 0.73 px error, 10k points; the loop closes |
| MoGe's dense depth is the remaining gap | At COLMAP's points, MoGe is a median 15–20% off after its best per-frame scale (crop or not); fused depth still drops 47–61% |

So for calm footage at a known fixed FOV, camera tracking is solved (COLMAP). What's missing is
dense geometry that agrees with those cameras.

### Dense depth at COLMAP's cameras (Test4)

Scored against COLMAP's points seen from at least 10° apart (the far, barely-triangulated points are
COLMAP's own uncertainty, not the models'), on 30% held-out points, after fitting each frame to the
rest with one scale or a monotone depth curve. Scripts: `tools/bakeoff/score_anchors.py`.

| Depth (given the true FOV) | Cameras given? | Median error, scale | Median error, curve | Within 5% (curve) |
|---|---|---|---|---|
| **MoGe-3** (single image) | no | **5.8%** | **4.7%** | **51%** |
| MapAnything (Apache) | yes, COLMAP's | 7.0% | 5.4% | 48% |
| Depth Anything 3 Large 1.1 (CC BY-NC) | yes, COLMAP's | 8.1% | 5.8% | 47% |
| Depth Anything 3 Base (Apache) | yes, COLMAP's | 14.5% | 8.2% | 40% |

Including points seen from 5°+: MoGe 7.6%, MapAnything 9.0%, DA3-L 8.8%, DA3-B 12.5% (curve).
Up close (the floor under you) and far (skyline, 4x+ the median distance) all models are much worse:
22–46% near, and far things put 1.9–2.6x too close.

Fused at COLMAP's cameras (MoGe, curve-fitted, cut beyond the measured range), the build shows flat
facades, the crane and billboards, but stays noisy, and the free-space filter still drops 53%: at
~5% depth error it is too strict for this footage.

### Averaging (TSDF) instead of cutting (Test4)

`tools/bakeoff/tsdf_fuse.py` integrates every frame's fitted MoGe depth into a truncated
signed-distance grid on the GPU (weighted by nearness) and takes the surface where the average
crosses zero. Scored against COLMAP's points seen from 10°+ apart (distance to the nearest fused
point, relative to the camera distance), one point per voxel for all three:

| Fusion | Points | Accuracy (median) | Within 2% | Within 5% | Isolated specks |
|---|---|---|---|---|---|
| All frames, no filter | 1.10M | 2.5% | 42% | 79% | 8.7% |
| Cut (free-space filter) | 0.74M | 3.3% | 32% | 65% | 10.5% |
| **TSDF average** (min weight 0.5, truncation 6 voxels) | 0.56M | 5.3% | 17% | 48% | **1.0%** |

(Dense noisy clouds score well on accuracy/coverage because some point lands near every anchor;
the speck rate is what catches their noise.) Tighter TSDF settings (weight 2, truncation 4) keep
only 12% within 2%.

Built at character scale (`tools/bakeoff/build_cloud.py`, block = camera height / 1.62), TSDF walls
are thinner and cleaner, with almost no specks, but it covers less. In **both**, the rooftop
floor you walked on is mostly missing: it's plain, seen at grazing angles, has almost no COLMAP
points to fit to, and near-camera depth is the models' worst range (22–46% off). The camera path
itself says where that floor is (camera height minus eye height), which is the obvious next fix.

### The walked floor (Test4)

`build_cloud.py` with a `+floor` cloud adds a flat floor one eye height below the path's median
camera height, over the area the path covers (loop filled, widened by 2.5 eye heights). "Up" is now
the normal of the plane through the camera path when it's flat: on Test4 that's 2.6° from the
cameras' average up and 7.6° from find_up on the fused walls, and it gives the smallest camera height
spread (0.29 vs 0.41 and 1.34 units; still ~half an eye height, so COLMAP's path isn't perfectly level).

The floor lands where expected (under the path, 472 block columns), but **Boxes can't make the
building a box: nothing below the roof was ever seen.** The roof is the lowest thing in the scene
(block y 2, boxify's ground y 1), so boxify takes it for the ground; 2% of its columns count as
building. A building's height needs its base: either look over the edge at the street while
recording, or take the ground from another scene in the World, or set it by hand.

### Test5: walk to all four ledges and look down (98 s)

- COLMAP (camera fixed at 114°) "registers" all 391 frames at 0.68 px (self-calibration: 115°), but
  **the model breaks in the middle (≈29–62 s)**: camera positions jump and depth scale swings from
  ~0.2 to ~8 units between stretches. COLMAP's global mapper (GLOMAP-style) breaks the same way.
- The break starts where one tall facade of identical window stripes fills the frame (frames 110–118),
  the textbook failure for feature matching: repeated patterns give confident wrong matches. The
  look-downs themselves are mostly rotation on the spot, which carries no scale.
- MoGe per frame is fine (3.7% median error at well-measured points, 5.9% at 5°+), so depth is
  not the problem here; the cameras are.
- Two fixes found on the way: "up" must come from the plane through the camera path (averaging
  camera tilts was 63° off when looking down a lot), and eye height must take the *nearest* strong
  level below the camera (the street seen over a ledge outnumbers the roof); `minecraft.eye_height`
  does that now (tested).

#### Splitting Test5 at the breaks (`tools/bakeoff/split_world.py`)

Per frame, COLMAP's scale relative to MoGe's (median z_colmap / z_moge at its points) sits in steady
regimes and jumps where the model breaks: ~0.01 (0–28 s), ~1e-4 (29–42 s, collapsed), 0.2–0.6 (43–63 s),
~0.01 (65–88 s), ~0.002 (89–98 s). Splitting where it jumps 2.5× (or a camera step spikes 20×) gave 8
pieces of 3 s+; one (30–40 s) had no usable depth and was dropped. Each piece keeps the pixels where
its own TSDF average agrees (68–74%), is levelled by its path plane (or floors/walls when the path is
one line), saved as a scene, and added to a World: sized by eye height, placed by roof matching.

- Sizing is consistent: blocks per MoGe metre came out 0.57–0.84 across all 7 pieces, although their
  COLMAP scales differ 300×.
- All 7 placements scored 0.63–0.90 and were auto-confirmed, and the build (182×84×170 blocks) shows
  far more of the block than Test4, with the street under the buildings so Boxes extrudes them.
- **But placement is not right**: the first piece (0–12 s) and the last (89–98 s) are both "standing in
  the middle of the roof" and should overlap; they land about a third of the map apart, and the green
  Cafe sign appears twice. Roof matching alone finds plausible, wrong spots.

The principled fix is to place neighbouring pieces from the frames they share with each other:
COLMAP already matched features across every break (sequential matching), and MoGe gives those
matches 3D positions, so each break gives a 3D-3D fit (turn, scale, offset) instead of a search.

#### Placing Test5's pieces from shared feature matches (`tools/bakeoff/place_pieces.py`)

Keyframes (every 3rd) of every pair of pieces were matched with COLMAP (4,539 pairs, 104 s); each
verified match lifted to 3D with each piece's fitted depth; a robust similarity fit per pair of pieces;
then pieces placed from piece 1 along believable links. World placements gained an optional `yaw`
(degrees on top of quarter turns): the pieces' wall grids came out 10–57° apart, so rounding to
quarter turns threw real rotation away. Without guards, "best fitting link" picked collapsing scales
(a piece shrunk to nothing fits perfectly), so a link must keep the piece within 2× of its eye-height
size and fit within 25% of its matched points' spread.

- Piece 2 from piece 1: 27,658 matched points, 0.95 blocks median error (8% of spread). Confirmed.
- Piece 7 (the end) from piece 1: 12,542 points but 4.6 blocks (15%); left unconfirmed. Start and
  end cameras land 9.6 blocks apart; the build with it shows the burger billboard doubled.
- Pieces 3–6 (the ledge look-downs, 25–65 s) have **no believable link**: every fit collapses or
  blows up their size. They come from the stretch where COLMAP's model broke, so their own geometry
  is distorted; linking can't repair that.
- The confirmed pieces 1+2 give the clearest build yet (109×78×111 blocks): the building front with
  its billboards, the green Cafe sign on the side, Boxes extruding to the street.

So: splitting + linking recovers the parts of a recording COLMAP got right, and refuses the rest.
The look-downs need either better capture (diagonal, roof edge in view, steps along the ledge) or a
tracker that handles rotation-only stretches.

Capture lesson: keep something distinctive (billboard, roof props, the roof edge) in frame; never let
one repeated facade fill the view; tilt down diagonally so the roof edge stays in view.

## 2. What we don't know yet (check before building)

1. **Is the in-game FOV constant?** Many parkour games widen the FOV with speed. If Parkour Legacy
   does, every method that assumes one camera (all of them below) is fed wrong geometry. Check: the
   game's settings, or stand still vs sprint and compare a screenshot.
2. **What the FOV actually is.** Roblox's default is 70° vertical
   ([Camera docs](https://create.roblox.com/docs/pt-br/workspace/camera)); the game may change it.
3. **Can we get the geometry legitimately?** See 3A.
4. **Can we record calmly?** Walking survey passes (no jumps, slow turns, buildings in view) vs parkour runs.

## 3. How others solve it

### A. Don't reconstruct: export the real geometry
Roblox Studio exports a place or a selection as OBJ (File → Export Place / Export Selection)
([guide](https://nilo.io/articles/export-obj-files-roblox-studio)), and
[ObjToSchematic](https://github.com/LucasDower/ObjToSchematic) turns OBJ into .schem/.litematic.
That gives the exact map. **But** it needs the place file: PARKOUR Legacy (A Mongoose Studio,
[game page](https://www.roblox.com/games/445664957/PARKOUR-Legacy)) isn't open, and ripping
another creator's place breaks Roblox's terms and their copyright. The legitimate version is
**asking A Mongoose Studio** for the map (or permission) for the video. For a YouTube video this is
a real option, and a cheap one to try first.

### B. Classic structure-from-motion
COLMAP (4.0 absorbed GLOMAP's fast global mapper) with learned features (hloc:
SuperPoint/DISK + [LightGlue](https://arxiv.org/pdf/2306.13643)). Strong when frames are sharp,
static and overlapping, and it can take the camera as known and fixed (one camera, our FOV). Known
weak spots are ours too: weak texture and repetitive structure (LightGlue's authors say so); line
features help in texture-poor scenes
([LightGlueStick](https://arxiv.org/pdf/2510.16438)). Sparse: geometry would come from MoGe/VGGT
depth placed at COLMAP's cameras.

### C. Learned SLAM for long videos
- [VGGT-Long](https://arxiv.org/pdf/2507.16443): chunks, Sim(3) alignment between chunks, loop
  closure; km-scale outdoor driving. Its repo warns that **very dense input with small motion
  between frames increases drift**.
- [VGGT-Motion](https://arxiv.org/html/2602.05508) (2026): says VGGT-Long/VGGT-SLAM still drift in
  scale at submap boundaries; motion-aware chunking, 85–95% less drift (self-reported).
- [VGGT-SLAM 2.0](https://arxiv.org/pdf/2601.19887), [Mamba-VGGT](https://arxiv.org/html/2605.17478v1): same family.
- [MASt3R-SLAM](https://arxiv.org/pdf/2412.12392): calibration-free, 15 fps, robust "in the wild";
  beaten by DROID-SLAM where calibration is known.
- [MegaSaM](https://arxiv.org/html/2412.04463v2) (DROID-SLAM based): built for casual video with
  **little parallax** and unknown FOV, our forward-running case.
- Streaming models ([TTT3R](https://arxiv.org/pdf/2509.26645), CUT3R, StreamVGGT): CUT3R drifts;
  TTT3R reduces it. All self-reported.

### D. Stronger feed-forward models
- [Depth Anything 3](https://arxiv.org/abs/2511.10647) (ICLR 2026): poses + depth from any number of
  views; claims 35–44% better camera pose than VGGT (own benchmark). Not yet tried here.
- Pi3X, MapAnything: tried in `tools/bakeoff`, no winner on Roblox footage.
- No benchmark tests these on low-texture, repetitive or game footage (searched; none found).

### E. Single images + registration (the screenshot idea)
Per-screenshot depth (MoGe), then place screenshots against each other by feature matches lifted
to 3D, or by matching geometry. This is what B does, with MoGe supplying depth. It skips drift
inside a clip but still needs overlap and features between neighbours.

### F. Fit the shape you want
Box-world reconstruction is an old, solved-ish problem once a point cloud exists:
[Li, Wonka, Nan 2016](https://3d.bk.tudelft.nl/liangliang/publications/2016/manhattan/manhattan.html)
picks boxes from candidates (code available);
[He et al. 2021](https://diglib.eg.org/handle/10.1111/cgf14421) fits cubes from SfM lines and
corners. Our `boxify.py` is a simpler 2.5D version. Fitting boxes also forgives a noisy cloud,
but not a wrongly *placed* one.

### Takeaways
- The bottleneck is camera tracking on low-texture, low-parallax game footage. Nobody publishes
  results on this kind of footage, so we must measure ourselves.
- The most robust published trackers for our conditions are DROID-family (MegaSaM) and
  COLMAP with a known fixed camera; the most promising new model is DA3.
- Capture matters as much as the model: every method above prefers calm, overlapping, textured
  views over parkour runs.

## 4. Plan

### Step 0 — truth checks (an evening, no code)
1. Ask A Mongoose Studio for the map or permission. If yes, plan A wins: OBJ → our boxes/palette or ObjToSchematic.
2. Check whether the game's FOV changes with speed, and what it is.
3. Record two survey clips (walk, slow turns, buildings in view, look down sometimes), 30 s each,
   overlapping one another.

### Step 1 — a Roblox benchmark with ground truth (the missing piece)
We judge methods by eye today. Build a small **box city in Roblox Studio ourselves** (free, our own
place): night lighting, plain rooftops, repeated windows, cables, like Parkour Legacy. Export its
OBJ (exact geometry) and record footage in it with a fixed known FOV: one survey walk and one
parkour run. Score each method on camera error and surface error, as the playground does. This is real
Roblox rendering with exact truth, and it keeps us honest.

### Step 2 — bake-off on that benchmark (decide by numbers)
Candidates, cheapest first:
1. Our pipeline today (VGGT) and the short-piece World (baseline).
2. COLMAP/GLOMAP + LightGlue, one fixed camera at the known FOV, MoGe depth at its cameras.
3. MegaSaM.
4. Depth Anything 3 (check its licence).
5. VGGT-Long (or VGGT-Motion if code is out).

Bar, set now: on the survey clip, camera error < 1° and < 2% of path length, surface error
< 0.5 block at character scale, and no piece needing manual placement more than once per
30 s. A method that passes on the survey clip but not the parkour run means "record survey
passes for the map", which is fine.

### Step 3 — integrate the winner, keep what works
The winner replaces only the camera tracking step. Worlds (sessions, manual placement), character
size and boxes stay. If nothing passes, plan E with manual placement as the main tool becomes the
fallback: screenshots + MoGe + World map editing.

### Step 4 — better boxes
Once placement is right, consider a Manhattan box fitter (Li et al. 2016) instead of 2.5D boxify,
judged on the same benchmark.
