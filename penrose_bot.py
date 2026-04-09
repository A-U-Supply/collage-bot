"""Collage stencil Penrose bot.

Divides the image into an aperiodic Penrose P3 tiling (thick and thin
rhombuses) generated via Robinson triangle inflation. Tiles of each type
are shuffled within their groups and pasted back — creating a scrambled
mosaic with 5-fold symmetry and no periodic repetition.

Posts a color result and an Otsu binary version.
"""
import argparse
import cmath
import logging
import math
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

PHI = (1 + math.sqrt(5)) / 2


def inflate(triangles: list) -> list:
    """One step of Penrose P3 Robinson triangle inflation."""
    result = []
    for kind, A, B, C in triangles:
        if kind == 0:  # acute triangle (part of thick rhombus)
            P = A + (B - A) / PHI
            result += [(0, C, P, B), (1, P, C, A)]
        else:  # obtuse triangle (part of thin rhombus)
            Q = B + (C - B) / PHI
            R = B + (A - B) / PHI
            result += [(1, R, C, A), (1, Q, R, B), (0, R, Q, A)]
    return result


def apply_penrose(img: Image.Image) -> Image.Image:
    w, h = img.size

    # Cover the whole image from center with some margin
    diag = math.sqrt(w ** 2 + h ** 2) / 2 * 1.3
    target_tile = min(w, h) * 0.06
    N = max(3, round(math.log(diag / target_tile) / math.log(PHI)))
    logger.info(f"Penrose inflations: {N}, target tile: {target_tile:.0f}px")

    # Start: 10 acute triangles in a sun wheel of radius 1
    triangles = []
    for i in range(10):
        B = cmath.rect(1, (2 * i - 1) * math.pi / 10)
        C = cmath.rect(1, (2 * i + 1) * math.pi / 10)
        if i % 2 == 0:
            B, C = C, B
        triangles.append((0, 0j, B, C))

    for _ in range(N):
        triangles = inflate(triangles)

    logger.info(f"Generated {len(triangles)} triangles")

    cx_img, cy_img = w / 2.0, h / 2.0

    def to_px(z: complex) -> tuple:
        return (cx_img + z.real * diag, cy_img + z.imag * diag)

    # Compute tile half-size analytically: each inflation step shrinks by PHI
    tile_side = diag / (PHI ** N)
    tile_half = max(4, int(tile_side * 1.2))  # slightly larger than tile for margin
    size = tile_half * 2
    logger.info(f"Tile half-size: {tile_half}px")

    thick_data = []  # (crop, mask, paste_xy)
    thin_data = []

    for kind, A, B, C in triangles:
        Apx, Bpx, Cpx = to_px(A), to_px(B), to_px(C)
        centroid_x = (Apx[0] + Bpx[0] + Cpx[0]) / 3
        centroid_y = (Apx[1] + Bpx[1] + Cpx[1]) / 3

        # Skip tiles fully outside image
        if (centroid_x + tile_half < 0 or centroid_x - tile_half > w or
                centroid_y + tile_half < 0 or centroid_y - tile_half > h):
            continue

        local_pts = [
            (Apx[0] - centroid_x + tile_half, Apx[1] - centroid_y + tile_half),
            (Bpx[0] - centroid_x + tile_half, Bpx[1] - centroid_y + tile_half),
            (Cpx[0] - centroid_x + tile_half, Cpx[1] - centroid_y + tile_half),
        ]

        crop = img.crop((
            centroid_x - tile_half,
            centroid_y - tile_half,
            centroid_x + tile_half,
            centroid_y + tile_half,
        ))

        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).polygon(local_pts, fill=255)

        paste_xy = (int(centroid_x - tile_half), int(centroid_y - tile_half))

        if kind == 0:
            thick_data.append((crop, mask, paste_xy))
        else:
            thin_data.append((crop, mask, paste_xy))

    logger.info(f"Thick tiles: {len(thick_data)}, thin tiles: {len(thin_data)}")

    # Shuffle crops within each type group
    thick_crops = [c for c, _, _ in thick_data]
    thin_crops = [c for c, _, _ in thin_data]
    random.shuffle(thick_crops)
    random.shuffle(thin_crops)

    result = img.copy()

    for (_, mask, paste_xy), crop in zip(thick_data, thick_crops):
        result.paste(crop, paste_xy, mask)

    for (_, mask, paste_xy), crop in zip(thin_data, thin_crops):
        result.paste(crop, paste_xy, mask)

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil Penrose bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./penrose-bot-output"))
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
    result = apply_penrose(img)

    dest = out_dir / "penrose_result.png"
    result.save(dest)
    logger.info(f"Saved {dest.name}")

    binary = make_stencil(result).convert("RGB")
    dest_binary = out_dir / "penrose_binary.png"
    binary.save(dest_binary)
    logger.info(f"Saved {dest_binary.name}")

    if not args.no_post:
        post_collages(token, args.post_channel, [dest, dest_binary], bot_name="collage-stencil-penrose-bot", threaded=False)
        logger.info(f"Posted to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
