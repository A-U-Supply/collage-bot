"""Collage stencil bullseye bot.

Fetches one image from Slack and applies four concentric circular
cut-and-rotate operations centered on the same point.

Circle diameters are 92%, 69%, 46%, and 23% of the shorter image dimension.
The center is fixed at the midpoint of the short axis; on the long axis it
floats randomly while keeping the outer circle's margin consistent on all sides.

Each circle is extracted from the original image, rotated independently
30–330°, then pasted back largest-to-smallest — creating a bullseye of
independently rotated rings.
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
    """Compute center points for as many bullseyes as fit along the long axis.

    Short axis: exactly centered.
    Long axis: arrangement is pushed as close to one edge as the margin allows,
    randomly choosing which end. Gap between bullseyes matches the short-side margin.
    """
    short = min(w, h)
    long_dim = max(w, h)
    cell = short  # cell size = 2*r1 + 2*margin = short
    margin = short // 2 - r1

    n = max(1, long_dim // cell)

    # Push to a random end — first bullseye near start, or last near end
    if random.choice([True, False]):
        first = r1 + margin  # near start
    else:
        first = (long_dim - r1 - margin) - (n - 1) * cell  # near end

    centers = []
    for i in range(n):
        long_pos = first + i * cell
        if w <= h:  # portrait
            centers.append((w // 2, long_pos))
        else:  # landscape
            centers.append((long_pos, h // 2))

    logger.info(f"Fitting {n} bullseye(s), pushed to {'start' if first == r1 + margin else 'end'}")
    return centers


def paste_bullseye(img_src: Image.Image, result: Image.Image, cx: int, cy: int, radii: list):
    """Extract crops from img_src, rotate independently, paste into result."""
    crops = []
    for r in radii:
        left, top = cx - r, cy - r
        size = r * 2
        crops.append(img_src.crop((left, top, left + size, top + size)))

    for crop, r in zip(crops, radii):
        angle = random.uniform(30, 330)
        size = r * 2
        rotated = crop.rotate(angle, resample=Image.BICUBIC)

        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size - 1, size - 1), fill=255)

        result.paste(rotated, (cx - r, cy - r), mask)


def apply_bullseye(img: Image.Image) -> Image.Image:
    w, h = img.size
    short = min(w, h)

    r1 = int(short * 0.46)  # diameter ≈ 92% of short side
    radii = [r1, int(r1 * 2 / 3), int(r1 / 3)]

    centers = bullseye_centers(w, h, r1)
    result = img.copy()

    for cx, cy in centers:
        logger.info(f"Bullseye center: ({cx}, {cy}), radii: {radii}")
        paste_bullseye(img, result, cx, cy, radii)

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil bullseye bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./bullseye-bot-output"))
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
    result = apply_bullseye(img)

    dest = out_dir / "bullseye_result.png"
    result.save(dest)
    logger.info(f"Saved {dest.name}")

    binary = make_stencil(result).convert("RGB")
    dest_binary = out_dir / "bullseye_binary.png"
    binary.save(dest_binary)
    logger.info(f"Saved {dest_binary.name}")

    if not args.no_post:
        post_collages(token, args.post_channel, [dest, dest_binary], bot_name="collage-stencil-bullseye-bot", threaded=False)
        logger.info(f"Posted to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
