"""Collage stencil lathe bot.

Bullseye variant using many thin concentric ring bands (annuli) instead of
thick circles. Each ring band is extracted from the original image, rotated
independently 30–330°, and pasted using a donut-shaped mask. Creates a
sliced-wood / agate / lathe-turned appearance.

Posts a color result and an Otsu binary version.
"""
import argparse
import logging
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def bullseye_centers(w: int, h: int, r1: int) -> list:
    """Short axis centered, long axis pushed to a random edge."""
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

    logger.info(f"Fitting {n} lathe(s)")
    return centers


def paste_lathe(img_src: Image.Image, result: Image.Image, cx: int, cy: int, radii: list):
    """Rotate and paste each ring band (annulus) from the original image.

    Processes outermost ring first. Each ring is extracted from the original,
    rotated independently, then masked to show only its band.
    """
    for i, r_outer in enumerate(radii):
        r_inner = radii[i + 1] if i + 1 < len(radii) else 0
        size = r_outer * 2
        left, top = cx - r_outer, cy - r_outer

        # Extract from original
        crop = img_src.crop((left, top, left + size, top + size))
        angle = random.uniform(30, 330)
        rotated = crop.rotate(angle, resample=Image.BICUBIC)

        # Annular mask: outer circle with inner circle cut out
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size - 1, size - 1), fill=255)
        if r_inner > 0:
            offset = r_outer - r_inner
            draw.ellipse((offset, offset, size - 1 - offset, size - 1 - offset), fill=0)

        result.paste(rotated, (left, top), mask)


def apply_lathe(img: Image.Image, rings: int = 20) -> Image.Image:
    w, h = img.size
    long_dim = max(w, h)

    r1 = int(long_dim * 0.46)
    radii = [max(1, int(r1 * (rings - i) / rings)) for i in range(rings)]

    centers = bullseye_centers(w, h, r1)
    logger.info(f"Rings: {rings}, outermost r={r1}px, band width={r1 // rings}px")

    result = img.copy()
    for cx, cy in centers:
        logger.info(f"Lathe center: ({cx}, {cy})")
        paste_lathe(img, result, cx, cy, radii)

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil lathe bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./lathe-bot-output"))
    parser.add_argument("--rings", type=int, default=80, help="Number of concentric ring bands")
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
    result = apply_lathe(img, rings=args.rings)

    dest = out_dir / "lathe_result.png"
    result.save(dest)
    logger.info(f"Saved {dest.name}")

    binary = make_stencil(result).convert("RGB")
    dest_binary = out_dir / "lathe_binary.png"
    binary.save(dest_binary)
    logger.info(f"Saved {dest_binary.name}")

    if not args.no_post:
        post_collages(token, args.post_channel, [dest, dest_binary], bot_name="collage-stencil-lathe-bot", threaded=False)
        logger.info(f"Posted to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
