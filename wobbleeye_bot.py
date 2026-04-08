"""Collage stencil wobble eye bot.

Like bullseye but each successive inner circle drifts from the previous
center in a random direction — rings wobble off-axis like a badly printed
target. Each circle is still rotated independently 30–330°.

Uses inverse mapping so each output pixel has exactly one source pixel —
no pixels are duplicated, only moved by rotation within their ring.

Posts a color result and an Otsu binary black and white version.
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


def drift_center(cx: int, cy: int, drift: int) -> tuple:
    """Drift a center point exactly `drift` pixels in a random direction."""
    angle = random.uniform(0, 2 * math.pi)
    return (
        cx + int(drift * math.cos(angle)),
        cy + int(drift * math.sin(angle)),
    )


def wobble_centers(cx0: int, cy0: int, radii: list) -> list:
    """Compute drifted center for each ring.

    Each center drifts 85–95% of the band width so the inner circle's far
    edge nearly grazes the outer circle — almost touching but not quite.
    """
    centers = [(cx0, cy0)]
    for i in range(1, len(radii)):
        band_width = radii[i - 1] - radii[i]
        drift = int(band_width * random.uniform(0.85, 0.95))
        cx, cy = drift_center(*centers[-1], drift)
        centers.append((cx, cy))
    return centers


def base_center(w: int, h: int, r1: int) -> tuple:
    """Place the outermost circle center: midpoint on short axis,
    pushed to a random edge on the long axis."""
    short = min(w, h)
    long_dim = max(w, h)
    margin = short // 2 - r1

    near_start = r1 + margin
    near_end = long_dim - r1 - margin

    long_pos = near_start if random.choice([True, False]) else near_end

    if w <= h:
        return (w // 2, long_pos)
    else:
        return (long_pos, h // 2)


def apply_wobbleeye(img: Image.Image, rings: int = 5) -> Image.Image:
    """Apply wobble-eye effect sequentially, outermost ring first.

    Each ring is cut from the current state of the result (after previous
    rings have been applied), rotated, and pasted back. Pixels cascade
    through rings — nothing is ever sampled twice from the original.
    """
    w, h = img.size
    short = min(w, h)

    r1 = int(short * 0.46)
    radii = [max(1, int(r1 * (rings - i) / rings)) for i in range(rings)]

    cx0, cy0 = base_center(w, h, r1)
    centers = wobble_centers(cx0, cy0, radii)

    logger.info(f"Rings: {rings}, radii: {radii}")
    for i, ((cx, cy), r) in enumerate(zip(centers, radii)):
        logger.info(f"  Ring {i + 1}: center=({cx}, {cy}), r={r}")

    result = img.copy()

    # Process outermost → innermost, each ring cut from current result
    for (cx, cy), r in zip(centers, radii):
        angle = random.uniform(30, 330)
        size = r * 2
        left, top = cx - r, cy - r

        crop = result.crop((left, top, left + size, top + size))
        rotated = crop.rotate(angle, resample=Image.BICUBIC)

        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size - 1, size - 1), fill=255)

        result.paste(rotated, (left, top), mask)

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil wobble eye bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./wobbleeye-bot-output"))
    parser.add_argument("--rings", type=int, default=5, help="Number of rings (3–10)")
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
    result = apply_wobbleeye(img, rings=args.rings)

    dest = out_dir / "wobbleeye_result.png"
    result.save(dest)
    logger.info(f"Saved {dest.name}")

    binary = make_stencil(result).convert("RGB")
    dest_binary = out_dir / "wobbleeye_binary.png"
    binary.save(dest_binary)
    logger.info(f"Saved {dest_binary.name}")

    if not args.no_post:
        post_collages(token, args.post_channel, [dest, dest_binary], bot_name="collage-stencil-wobbleeye-bot", threaded=False)
        logger.info(f"Posted to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
