"""Measure the block palette for the Minecraft export from a Minecraft client jar you own.

    uv run python tools/minecraft_palette.py "%APPDATA%/.minecraft/versions/1.16.5/1.16.5.jar"

Writes backend/minecraft_palette.json: each block's mean texture colour, averaged in linear light
(how a block's pixels blend together at a distance). Only numbers are stored, no textures.
Blocks are full cubes that look the same on every face and don't fall, glow, or let light through.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

COLORS = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
          "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black"]
WOODS = ["oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "crimson", "warped"]
BLOCKS = (
    [f"{c}_concrete" for c in COLORS]
    + [f"{c}_wool" for c in COLORS]
    + [f"{c}_terracotta" for c in COLORS] + ["terracotta"]
    + [f"{w}_planks" for w in WOODS]
    + """stone smooth_stone cobblestone mossy_cobblestone andesite polished_andesite diorite
    polished_diorite granite polished_granite stone_bricks mossy_stone_bricks cracked_stone_bricks
    bricks nether_bricks red_nether_bricks end_stone end_stone_bricks purpur_block prismarine
    prismarine_bricks dark_prismarine snow_block clay packed_ice blue_ice dirt coarse_dirt
    netherrack soul_soil polished_blackstone polished_blackstone_bricks obsidian coal_block
    iron_block gold_block diamond_block emerald_block lapis_block netherite_block quartz_bricks
    smooth_quartz smooth_sandstone smooth_red_sandstone nether_wart_block warped_wart_block
    honeycomb_block brown_mushroom_block red_mushroom_block mushroom_stem""".split()
)
TEXTURE = {  # where a block's texture name differs from its block id
    "snow_block": "snow",
    "smooth_quartz": "quartz_block_bottom",
    "smooth_sandstone": "sandstone_top",
    "smooth_red_sandstone": "red_sandstone_top",
}


def to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def main(jar: str) -> None:
    palette = {}
    with zipfile.ZipFile(jar) as z:
        for block in BLOCKS:
            name = f"assets/minecraft/textures/block/{TEXTURE.get(block, block)}.png"
            px = np.asarray(Image.open(io.BytesIO(z.read(name))).convert("RGBA"), np.float64) / 255
            px = px.reshape(-1, 4)
            px = px[px[:, 3] > 0.5, :3]  # animated textures are frame strips; every frame counts
            rgb = to_srgb(to_linear(px).mean(0))
            palette[f"minecraft:{block}"] = [round(float(v) * 255) for v in rgb]
    out = Path(__file__).resolve().parent.parent / "backend" / "minecraft_palette.json"
    rows = ",\n".join(f'  "{k}": {json.dumps(v)}' for k, v in palette.items())
    out.write_text(f'{{"source": "{Path(jar).name}", "blocks": {{\n{rows}\n}}}}\n')
    print(f"{len(palette)} blocks -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
