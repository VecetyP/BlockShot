# Playground: test against the ground truth

[← README](../README.md)

Switch the sidebar from **Upload** to **Playground** for a small first-person world built in the app.
Because the app draws it, every recording knows exactly where the camera was in every frame and where
every surface is, so each run gets scored against the truth instead of judged by eye.

![Recording the Courtyard preset, then its ground-truth score card](playground.gif)

- **Scenes:** presets (Room, Courtyard, Block city, Obstacle course), **Shuffle** for a random one, and
  an editor: **F** places the chosen shape under the crosshair, **X** deletes, **1–4** pick box,
  pyramid, sphere or cylinder, **[ ]** size, **T** material, **Y** rotate. Your scene is kept in the
  browser.
- **Moving:** click the view to look around with the mouse, **WASD** to walk, **Space** to jump,
  **Shift** to sprint. You can stand on boxes.
- **Recording:** **R** starts and stops a recording, which goes to the video pipeline with the options
  under "Recordings go to", plus the playground camera's field of view (as you'd enter it for a
  game). **P** sends a snapshot to the models ticked under "Snapshots go to".
  **Autopilot** records a repeatable path from the scene's start: a loop around the scene, a turn on the
  spot, or a zig-zag with quick turns.
- **Game nuisances:** HUD panels with a changing health bar, swinging arms, head bob and fast turns are
  drawn into the recording, to test the problems gameplay footage has.

Each playground run gets a **Ground truth** card next to its Minecraft card:
- **Camera angle error:** median, after lining the predicted camera path up with the true one.
- **Camera position error:** RMS.
- **Depth error:** median relative error.
- **Surface error:** median distance from the reconstructed points to the true surfaces.
- **Scale vs metres:** how far the scene's metres were off.
- **Recording offset:** the best match between the video's timing and the camera log, or "not
  measurable" when the motion is too steady to tell.

The scoring is in `backend/playground.py`: a ray caster gives the true depth of every pixel, and the
predicted cameras are lined up with the true ones (rotation, then scale from the depths, then
position) before anything is compared.

## Measurements

On the Courtyard preset (60 frames at 4 per second, VGGT, before the one-pass and field-of-view
changes):

| | camera angle | camera position | depth | surface |
|---|---|---|---|---|
| Autopilot loop | 0.8° | 44 cm over a 72 m path | 2.6% | 13 cm |
| Zig-zag | 1.1° | 47 cm | 3.5% | 17 cm |
| Zig-zag with HUD, arms, head bob, fast turns | 31.8° | 7.8 m | 6.7% | 32 cm |

The game nuisances wreck the camera tracking while each frame's depth stays decent. That matches what
happened with real Roblox footage (see [tools/bakeoff](../tools/bakeoff/README.md)), now measured in a
controlled way.

What the playground has found so far:
- **Sky:** a plain sky stayed in scenes as a dome until the MoGe sky mask (surface error 18.6 → 12.8 cm).
- **One VGGT pass:** joining two chunks put cameras 99 cm off; one pass of all 60 frames, 57 cm.
- **Field of view:** VGGT and MoGe both guess the playground's wide view about 8% too narrow; entering
  the real FOV took a loop's surface error from 10.8 to 8.5 cm.
- **Camera refinement (ICP) didn't help:** fitting each frame's depth onto the others made frames
  agree more but moved them away from the truth while the field of view was wrong, and gained under
  1 cm with the right one.
