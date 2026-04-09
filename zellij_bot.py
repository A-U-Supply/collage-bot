"""Collage stencil zellij bot.

Inspired by Moroccan zellij tilework. Divides the image into a classic Islamic
6-pointed star grid on a triangular lattice. Stars and shield-shaped connectors
between adjacent pairs of stars are cut from the source image, shuffled within
their respective groups, and pasted back without rotation.

Posts a color result and an Otsu binary version.
"""
import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

SQRT3 = math.sqrt(3)


def apply_zellij(img: Image.Image) -> Image.Image:
    w, h = img.size
    S = max(60, min(w, h) // 6)

    R_out = S * 0.42
    R_in = R_out * 0.50
    box = int(R_out) + 2
    star_size = box * 2

    # Flat-top 6-pointed star mask: outer points at 0°, 60°, 120°, ...
    star_mask = Image.new("L", (star_size, star_size), 0)
    scx = scy = box
    star_pts = []
    for i in range(12):
        angle = math.radians(i * 30)
        r = R_out if i % 2 == 0 else R_in
        star_pts.append((scx + r * math.cos(angle), scy + r * math.sin(angle)))
    ImageDraw.Draw(star_mask).polygon(star_pts, fill=255)

    # Triangular lattice: e1=(S,0), e2=(S/2, S*√3/2)
    cx_img, cy_img = w / 2.0, h / 2.0
    margin = 2 * S
    steps = int(max(w, h) / S) + 4

    lattice = {}
    for r in range(-steps, steps + 1):
        for q in range(-steps, steps + 1):
            px = cx_img + S * q + S * 0.5 * r
            py = cy_img + S * SQRT3 / 2 * r
            if -margin < px < w + margin and -margin < py < h + margin:
                lattice[(q, r)] = (px, py)

    logger.info(f"S={S}px, {len(lattice)} star positions")

    # Reflect-pad source so edge crops never go out of bounds
    pad = int(S * 0.6)
    src = Image.fromarray(np.pad(np.array(img), ((pad, pad), (pad, pad), (0, 0)), mode='reflect'))

    # Star crops — all lattice positions, cropped from padded source
    star_crops = []
    star_pastes = []
    for px, py in lattice.values():
        x0, y0 = int(px) - box, int(py) - box
        star_crops.append(src.crop((x0 + pad, y0 + pad, x0 + pad + star_size, y0 + pad + star_size)))
        star_pastes.append((x0, y0))

    # Shield crops: 3 edge directions × 2 sides (upper/lower) = 6 orientation groups
    # For edge A→B at angle theta:
    #   upper shield (sign=+1): A outer@θ, A inner@θ+30°, A outer@θ+60°,
    #                            B outer@θ+120°, B inner@θ+150°, B outer@θ+180°
    #   lower shield (sign=-1): same with negative offsets
    neighbor_dirs = [(1, 0, 0.0), (0, 1, 60.0), (-1, 1, 120.0)]

    shield_groups = [[] for _ in range(6)]  # 0-2 upper, 3-5 lower

    def make_shield_item(ax, ay, bx, by, theta, sign):
        pts = [
            (ax + R_out * math.cos(theta),
             ay + R_out * math.sin(theta)),
            (ax + R_in * math.cos(theta + sign * math.pi / 6),
             ay + R_in * math.sin(theta + sign * math.pi / 6)),
            (ax + R_out * math.cos(theta + sign * math.pi / 3),
             ay + R_out * math.sin(theta + sign * math.pi / 3)),
            (bx + R_out * math.cos(theta + sign * 2 * math.pi / 3),
             by + R_out * math.sin(theta + sign * 2 * math.pi / 3)),
            (bx + R_in * math.cos(theta + sign * 5 * math.pi / 6),
             by + R_in * math.sin(theta + sign * 5 * math.pi / 6)),
            (bx + R_out * math.cos(theta + math.pi),
             by + R_out * math.sin(theta + math.pi)),
        ]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = int(min(xs)) - 1
        y0 = int(min(ys)) - 1
        x1 = int(max(xs)) + 2
        y1 = int(max(ys)) + 2
        mw, mh = x1 - x0, y1 - y0
        local = [(x - x0, y - y0) for x, y in pts]
        mask = Image.new("L", (mw, mh), 0)
        ImageDraw.Draw(mask).polygon(local, fill=255)
        crop = src.crop((x0 + pad, y0 + pad, x1 + pad, y1 + pad))
        return crop, mask, (x0, y0)

    for dir_idx, (dq, dr, theta_deg) in enumerate(neighbor_dirs):
        theta = math.radians(theta_deg)
        for (q, r), (ax, ay) in lattice.items():
            nb = (q + dq, r + dr)
            if nb not in lattice:
                continue
            bx, by = lattice[nb]
            for sign, group_offset in [(+1, 0), (-1, 3)]:
                crop, mask, paste_xy = make_shield_item(ax, ay, bx, by, theta, sign)
                shield_groups[dir_idx + group_offset].append((crop, mask, paste_xy))

    total_shields = sum(len(g) for g in shield_groups)
    logger.info(f"Shields: {total_shields} in 6 orientation groups")

    result = img.copy()

    # Paste shields first, stars on top
    for group in shield_groups:
        crops = [c for c, _, _ in group]
        random.shuffle(crops)
        for (_, mask, paste_xy), crop in zip(group, crops):
            if crop.size != mask.size:
                crop = crop.resize(mask.size, Image.LANCZOS)
            result.paste(crop, paste_xy, mask)

    random.shuffle(star_crops)
    for crop, paste_xy in zip(star_crops, star_pastes):
        result.paste(crop, paste_xy, star_mask)

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil zellij bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./zellij-bot-output"))
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages
    from stencil_transform import make_stencil

    source_dir = args.output_dir / "source"
    out_dir = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching 1 image from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 1, source_dir))
    img = Image.open(source_paths[0]).convert("RGB")

    logger.info(f"Image size: {img.width}×{img.height}")
    result = apply_zellij(img)

    dest = out_dir / "zellij_result.png"
    result.save(dest)
    logger.info(f"Saved {dest.name}")

    binary = make_stencil(result).convert("RGB")
    dest_binary = out_dir / "zellij_binary.png"
    binary.save(dest_binary)
    logger.info(f"Saved {dest_binary.name}")

    if not args.no_post:
        post_collages(token, args.post_channel, [dest, dest_binary], bot_name="collage-stencil-zellij-bot", threaded=False)
        logger.info(f"Posted to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
