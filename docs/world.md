# Worlds: many videos, one map

[← README](../README.md)

The **World** tab puts several video scenes into one Minecraft map, so a big place can be recorded
over many sessions. A world is `worlds/<name>/world.json`: which scenes it uses and where each goes
(quarter turn, scale, offset in blocks). Scenes stay as they are in `runs/`; a world can be rebuilt
or rearranged any time.

1. Record and upload a video on the **Upload** tab as usual.
2. On **World**, create or open a world, pick the scene and press **Add scene**. The first scene sets
   the world's origin (its first camera).
3. Each later scene is sized character-sized (see *Character size* in [builds.md](builds.md)), then
   its top-down height map is matched against the world's for every quarter turn and a few sizes
   around that. The best match is shown on the map, highlighted.
4. Check it: drag it, turn it 90°, make it bigger/smaller or raise it, then **Confirm**. A scene
   without a match ("place it by hand") lands at the origin.
5. **Export tiles** writes every confirmed scene as one build (one block per world block, *As seen*
   or *Boxes*), as 128 × 128 tiles with `paste-commands.txt`.

**Recording for a world**: start every new video somewhere already in the world (a rooftop you've
recorded) and look around it before moving on; the matching needs shared roofs or walls with
different heights. Flat ground alone can't be placed. Look down at the floor now and then, so the
size can be measured.

How it works: [backend/world.py](../backend/world.py). Matching is a correlation of roof heights
over the overlap, computed for every shift at once by FFT; the walls of each scene are lined up with
the block grid first, so only quarter turns are left to try.
