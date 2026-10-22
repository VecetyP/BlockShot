# Video → one scene

[← README](../README.md)

Drop an `mp4`, `mov` or `webm` instead of an image. The options:

- **Frames/s** (default 2) and **Max frames** (default 60): how densely the clip is sampled.
  Near-duplicate frames (a camera standing still) are dropped.
- **Geometry:** *VGGT depth* (default: the frames agree with each other) or *MoGe detail* (sharper
  per-frame depth bent onto VGGT's shape, but with more ghosting).
- **Ignore a static game HUD** (on): pixels that stay still while the picture moves, such as a
  crosshair, hotbar or minimap, are left out of the scene.
- **Vertical FOV** (empty = auto): the camera's vertical field of view in degrees, if you know it.
  Games usually show it in their settings (Roblox and Minecraft default to 70). Otherwise VGGT
  guesses it, and it tends to guess wide views too narrow. That bends every frame so they don't line
  up: on a playground loop the surfaces came out 10.8 cm off with the guess, and 8.5 cm with the
  real FOV.

The result is a **scene**: one point cloud from all frames, shown with a **View from frame** slider
and a strip of the sampled frames. Click a frame to view the scene, and its build, from that frame's
camera. The Minecraft build is seen from the chosen frame, with floors levelled from the average
camera "up". Stand there in game the same way as for an image.

A scene keeps the result of every step below, so improvements to them only reach scenes made
afterwards. History tags older ones **older version**; tick them and press **Recompute** to redo
them. The video isn't kept, so they are redone from their stored frames. Image runs only store the
model's output, so they are tagged only when that model changes; levelling, builds and analysis are
worked out fresh whenever a run is viewed.

## Recording tips

- Move slowly, with each moment overlapping the last. Fast motion, motion blur and blank walls can
  throw the camera poses off.
- Circling an object up close is the hardest case: on a 29 s clip walking around a chair at about
  1 m (462° of turning), neighbouring frames agreed on only 55% of points (93% on a hallway walk), and
  ghost removal leaves holes where the views disagree.
- For games: hide the HUD if the game allows it, keep the character's own body out of view (a free
  camera, or looking ahead rather than down), avoid teleports and respawns (record each area as its
  own clip), and enter the game's field of view.

**Gameplay recordings.** Frantic first-person gameplay doesn't map. A 35 s Roblox run with swinging
arms, changing HUD bars, snap turns and a respawn reached 8–29% neighbour agreement, and no newer model
fixed it. Pi3X and MapAnything were tried; see [tools/bakeoff](../tools/bakeoff/README.md). The
[playground](playground.md) reproduces the problem in a controlled way.

## How it works

`backend/video.py`:

1. VGGT reads all frames in one pass, up to 48k image patches: the default 60 frames, landscape or
   portrait, with each frame's longer side at 518 px (21 s, peak 9.4 GB of GPU memory). Longer clips go
   in overlapping chunks of that size, joined with a similarity fit on the shared frames, but one pass
   is better: on a playground clip, two chunks put the cameras 99 cm off where one pass got 57 cm.
   It gives every frame's camera pose and depth in one frame of reference, so a frame that comes back
   to something already seen agrees with the first view of it.
2. MoGe, on six frames, sets the scale in metres. A quick coarse MoGe pass on every frame (about
   0.25 s each) masks out the sky.
3. **Levelling.** The scene is turned so its floors are horizontal and walls vertical, found the same
   way as for images, so a phone held at an angle (36 degrees in one test clip) still gives a level
   scene and build.
4. **Ghost removal.** A point is dropped when more frames saw straight through it (it floats in front
   of everything they saw near there) than saw it. Glass reflections and the edges of objects go;
   walls, perforated screens and shiny floors that many frames agree on stay. It can also remove a
   person walking through the shot. Each frame is checked against up to 24 others spread over the
   clip: about 20 s for 50 frames.
5. Each frame's pixels are placed with its VGGT pose. Builds fill the surface between neighbouring
   pixels of each frame, so close walls don't come out as sparse dots.

**Why VGGT's depth and not MoGe's.** On a 24-frame clip of a kitchen, walking and turning around, two
frames agree within 5% on 62% of the points they share when each frame uses MoGe's depth. 12.8% of
points float in space another frame saw as empty, and those ghosts are what makes objects come out
doubled and see-through. MoGe's guess bends the room a little differently in every frame, and no
per-frame scale fixes a bend. VGGT's depth is predicted together with the poses; with ghost removal,
82% agree and 2.9% float, keeping 97% of the pixels. It is faster too: 47 s instead of 104 s for
that clip. (A stricter first version reached 0.4%, but kept only 58% of a hallway clip: dropping
VGGT's least confident pixels took plain walls and shiny floors with them, and the filter took a
perforated brick wall. Now 94% of that hallway stays. Unsure pixels are only dropped when far away.)

**The sky.** A plain sky gives VGGT nothing to measure, so it places it just past the walls. On a
playground clip, where sky covered 40% of each frame, up to 66% of it stayed in the scene as a pale
dome. With MoGe's mask, 1% stays, and three indoor clips lost none of their points to it.

## Measurements

On an RTX 3080 Laptop (16 GB) with a synthetic 10 s room walkthrough rendered from known camera poses
(20 frames at 640 × 360):

| | |
|---|---|
| Total time (MoGe depth on every frame, before ghost removal) | 51 s: poses 15 s, depth 34 s |
| Camera rotation error | 0.7° (median) |
| Camera position error | 2.8 cm (RMS, after aligning scale) |
| Surface error | 24–28 cm (median) |
| VGGT, 64 frames in one pass | 26 s, 9.8 GB peak VRAM |

MoGe's metres were 1.56× off on the synthetic room, so treat real-size builds from video as
approximate. The [playground](playground.md) has further measurements against exact ground truth.
