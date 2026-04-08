"""Collage stencil kaleidoscope bot.

Bullseye variant: same concentric ring structure (thirds) but each ring
gets N-fold radial symmetry applied before rotation — creating a mandala /
kaleidoscope look. Each ring draws from the original image independently
so the symmetry is clean per ring. All rings share the same fold count.

Posts a color result and an Otsu binary version.
"""
import argparse
import logging
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def apply_kaleidoscope(crop: Image.Image, folds: int) -> Image.Image:
    """Apply N-fold radial symmetry to a square circular crop.

    Folds polar coordinates into one wedge and mirrors alternating wedges
    for seamless joins — like looking through a kaleidoscope tube.
    """
    arr = np.array(crop.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = x_grid - cx
    dy = y_grid - cy

    theta = np.arctan2(dy, dx)           # -π to π
    r = np.sqrt(dx ** 2 + dy ** 2)

    wedge = 2 * np.pi / folds
    theta_pos = theta % (2 * np.pi)      # 0 to 2π
    theta_folded = theta_pos % wedge     # fold into first wedge

    # Mirror alternating wedges for seamless joins
    wedge_idx = (theta_pos / wedge).astype(int)
    theta_folded = np.where(wedge_idx % 2 == 1, wedge - theta_folded, theta_folded)

    src_x = np.clip(cx + r * np.cos(theta_folded), 0, w - 1).astype(np.float32)
    src_y = np.clip(cy + r * np.sin(theta_folded), 0, h - 1).astype(np.float32)

    result = cv2.remap(arr, src_x, src_y, cv2.INTER_LINEAR)
    return Image.fromarray(result.astype(np.uint8))


def bullseye_centers(w: int, h: int, r1: int) -> list:
    """Compute bullseye centers: short axis centered, long axis pushed to a random edge."""
    short = min(w, h)
    long_dim = max(w, h)
    cell = short
    margin = short // 2 - r1

    n = max(1, long_dim // cell)
    if random.choice([True, False]):
        first = r1 + margin
    else:
        first = (long_dim - r1 - margin) - (n - 1) * cell

    centers = []
    for i in range(n):
        long_pos = first + i * cell
        if w <= h:
            centers.append((w // 2, long_pos))
        else:
            centers.append((long_pos, h // 2))

    logger.info(f"Fitting {n} bullseye(s)")
    return centers


def paste_kaleidoscope_bullseye(
    img_src: Image.Image,
    result: Image.Image,
    cx: int,
    cy: int,
    radii: list,
    folds: int,
):
    """Apply kaleidoscope symmetry + rotation to each ring and paste into result.

    Each ring is extracted from the original image, folded into N-fold
    symmetry, rotated independently, then pasted largest → smallest.
    """
    for r in radii:
        size = r * 2
        left, top = cx - r, cy - r

        # Extract from original at this ring's position
        crop = img_src.crop((left, top, left + size, top + size))

        # Apply N-fold kaleidoscope symmetry
        kaleido = apply_kaleidoscope(crop, folds)

        # Rotate independently
        angle = random.uniform(30, 330)
        rotated = kaleido.rotate(angle, resample=Image.BICUBIC)

        # Circular mask
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size - 1, size - 1), fill=255)

        result.paste(rotated, (left, top), mask)


def apply_kaleidoscope_bullseye(img: Image.Image, folds: int = 6) -> Image.Image:
    w, h = img.size
    short = min(w, h)

    r1 = int(short * 0.46)
    radii = [r1, int(r1 * 2 / 3), int(r1 / 3)]

    centers = bullseye_centers(w, h, r1)
    logger.info(f"Folds: {folds}, radii: {radii}")

    result = img.copy()
    for cx, cy in centers:
        logger.info(f"Bullseye center: ({cx}, {cy})")
        paste_kaleidoscope_bullseye(img, result, cx, cy, radii, folds)

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil kaleidoscope bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./kaleidoscope-bot-output"))
    parser.add_argument("--folds", type=int, default=6,
                        help="Symmetry fold count (3=triangular, 6=hexagonal, 8=octagonal, 12=fine)")
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
    result = apply_kaleidoscope_bullseye(img, folds=args.folds)

    dest = out_dir / "kaleidoscope_result.png"
    result.save(dest)
    logger.info(f"Saved {dest.name}")

    binary = make_stencil(result).convert("RGB")
    dest_binary = out_dir / "kaleidoscope_binary.png"
    binary.save(dest_binary)
    logger.info(f"Saved {dest_binary.name}")

    if not args.no_post:
        post_collages(token, args.post_channel, [dest, dest_binary], bot_name="collage-stencil-kaleidoscope-bot", threaded=False)
        logger.info(f"Posted to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
