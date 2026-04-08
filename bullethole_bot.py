"""Collage stencil bullet hole bot.

Fetches one image from Slack and punches circular "bullet holes" into it.
A --chaos knob (0.0–1.0) controls both count and size inversely:
  0.0 = 2–3 large circles (up to w/3 diameter)
  0.5 = 5–7 medium circles
  1.0 = 13–15 small circles (down to w/12 diameter)
Each circle is cut from a random position, rotated 30–330° around its center,
and composited back in place. Posts the result to img-junkyard.
"""
import argparse
import logging
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def chaos_params(w: int, h: int, chaos: float) -> tuple:
    """Derive hole count and radius range from chaos value and image size.

    chaos=0.0 → 2–3 large holes (~33% of image width)
    chaos=1.0 → 85–87 small holes (~5% of image width)
    """
    # Count: 2 at chaos=0, 86 at chaos=1, with ±1 jitter
    n = max(2, round(2 + chaos * 84 + random.uniform(-1, 1)))

    # Radius as fraction of width: lerp 0.33 → 0.05
    frac = 0.33 + (0.05 - 0.33) * chaos
    max_r = max(4, round(w * frac))
    min_r = max(2, round(max_r * 0.6))

    return n, min_r, max_r


def apply_bullet_holes(img: Image.Image, chaos: float = 0.5) -> Image.Image:
    """Cut circular sections at chaos-determined count/size, rotate, place back."""
    w, h = img.size
    n, min_r, max_r = chaos_params(w, h, chaos)
    logger.info(f"chaos={chaos:.2f} → {n} holes, radius {min_r}–{max_r}px")

    result = img.copy()

    # All holes the same size — pick once per run
    radius = random.randint(min_r, max_r)
    logger.info(f"radius={radius}px for all holes")

    for _ in range(n):
        # Constrain center so the full circle stays within the image
        cx = random.randint(radius, w - radius)
        cy = random.randint(radius, h - radius)
        angle = random.uniform(30, 330)

        left = cx - radius
        top = cy - radius
        size = radius * 2

        crop = result.crop((left, top, left + size, top + size))
        rotated = crop.rotate(angle, resample=Image.BICUBIC)

        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size - 1, size - 1), fill=255)

        result.paste(rotated, (left, top), mask)

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil bullet hole bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./bullethole-bot-output"))
    parser.add_argument("--chaos", type=float, default=0.5,
                        help="0.0 = few large circles, 1.0 = many small circles")
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

    result = apply_bullet_holes(img, chaos=args.chaos)

    dest = out_dir / "bullethole_result.png"
    result.save(dest)
    logger.info(f"Saved {dest.name}")

    binary = make_stencil(result).convert("RGB")
    dest_binary = out_dir / "bullethole_binary.png"
    binary.save(dest_binary)
    logger.info(f"Saved {dest_binary.name}")

    if not args.no_post:
        post_collages(token, args.post_channel, [dest, dest_binary], bot_name="collage-stencil-bullethole-bot", threaded=False)
        logger.info(f"Posted to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
