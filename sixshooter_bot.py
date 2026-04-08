"""Collage stencil six shooter bot.

Fetches one image from Slack and applies six circular cut-and-rotate operations.

Layout depends on image proportions:
- Landscape (w > h by >25%): 3 cols × 2 rows grid
- Portrait  (h > w by >25%): 2 cols × 3 rows grid
- Near-square: 6 circles at the vertices of a regular hexagon, centered,
  with space between them

Each circle is centered on its section/vertex, rotated 30–330°, and placed back.
"""
import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)



def grid_circles(w: int, h: int, cols: int, rows: int) -> list:
    """Return (cx, cy, r) for a cols×rows grid of circles centered in each cell."""
    cell_w = w // cols
    cell_h = h // rows
    r = min(cell_w, cell_h) // 2
    circles = []
    for row in range(rows):
        for col in range(cols):
            cx = col * cell_w + cell_w // 2
            cy = row * cell_h + cell_h // 2
            circles.append((cx, cy, r))
    return circles


def hex_circles(w: int, h: int) -> list:
    """Return (cx, cy, r) for 6 circles at vertices of a regular hexagon.

    Hexagon is centered in the image. Circle radius is 30% of the hex radius
    so there's visible space between neighbors.
    """
    # Hex radius: largest R such that circles (r=0.42*R) stay within image
    # Outermost point = R + r = R + 0.42R = 1.42R ≤ min(w, h) / 2
    R = int(min(w, h) / 2 / 1.42)
    r = max(4, int(R * 0.42))

    cx0, cy0 = w // 2, h // 2
    circles = []
    for i in range(6):
        angle_deg = 90 + i * 60  # start at top, go clockwise
        angle_rad = math.radians(angle_deg)
        cx = cx0 + int(R * math.cos(angle_rad))
        cy = cy0 - int(R * math.sin(angle_rad))
        circles.append((cx, cy, r))
    return circles


def apply_six_shooter(img: Image.Image) -> Image.Image:
    w, h = img.size
    aspect = w / h

    if aspect > 1.25:
        logger.info("Landscape — 3×2 grid")
        circles = grid_circles(w, h, cols=3, rows=2)
    elif aspect < 0.80:
        logger.info("Portrait — 2×3 grid")
        circles = grid_circles(w, h, cols=2, rows=3)
    else:
        logger.info("Near-square — hexagonal arrangement")
        circles = hex_circles(w, h)

    # Extract all crops from the original before any modifications
    crops = []
    for cx, cy, r in circles:
        left, top = cx - r, cy - r
        size = r * 2
        crops.append(img.crop((left, top, left + size, top + size)))

    # Shuffle which crop lands at which position
    src_indices = list(range(6))
    random.shuffle(src_indices)

    result = img.copy()
    for dest_idx, src_idx in enumerate(src_indices):
        cx, cy, r = circles[dest_idx]
        size = r * 2
        angle = random.uniform(30, 330)
        rotated = crops[src_idx].rotate(angle, resample=Image.BICUBIC)

        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size - 1, size - 1), fill=255)

        result.paste(rotated, (cx - r, cy - r), mask)

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil six shooter bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./sixshooter-bot-output"))
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages

    source_dir = args.output_dir / "source"
    out_dir = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching 1 image from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 1, source_dir))
    img = Image.open(source_paths[0]).convert("RGB")

    logger.info(f"Image size: {img.width}×{img.height}, aspect={img.width/img.height:.2f}")
    result = apply_six_shooter(img)

    dest = out_dir / "sixshooter_result.png"
    result.save(dest)
    logger.info(f"Saved {dest.name}")

    if not args.no_post:
        post_collages(token, args.post_channel, [dest], bot_name="collage-stencil-sixshooter-bot", threaded=False)
        logger.info(f"Posted to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
