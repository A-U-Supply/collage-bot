"""Collage stencil cyanotype bot.

Like collage-stencil-silver-bot but tones fill images with a prussian
blue cyanotype palette: deep blue shadows, bright bleached highlights,
and paper grain. The stencil image remains full color for mask generation.
Posts all 6 variations as a single message.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Cyanotype tonal curve: deepen shadows, push highlights toward white
_CYN_IN  = [  0,  40, 128, 200, 255]
_CYN_OUT = [  0,  20, 120, 230, 255]
_CYN_LUT = np.interp(np.arange(256), _CYN_IN, _CYN_OUT).astype(np.uint8)

# Cyan color palette mapped across 0-255 grayscale
_CYN_R = np.interp(np.arange(256), [0, 64, 128, 192, 255], [0,  10,  30, 130, 220]).astype(np.uint8)
_CYN_G = np.interp(np.arange(256), [0, 64, 128, 192, 255], [40, 90, 160, 220, 248]).astype(np.uint8)
_CYN_B = np.interp(np.arange(256), [0, 64, 128, 192, 255], [60, 130, 200, 240, 255]).astype(np.uint8)


def to_cyanotype(img: Image.Image) -> Image.Image:
    """Apply cyanotype treatment: prussian blue palette with bleached highlights.

    1. Grayscale conversion
    2. Tonal curve (deepen shadows, lift highlights)
    3. Highlight bloom (screen-blend blurred highlights for glow/bleed)
    4. Prussian blue color LUT
    5. Paper grain
    """
    gray = np.array(img.convert("L"))

    curved = _CYN_LUT[gray]

    highlights = np.where(curved > 180, curved.astype(np.float32), 0.0)
    halo = cv2.GaussianBlur(highlights, (0, 0), sigmaX=18)
    a = curved.astype(np.float32)
    bloomed = 255.0 - (255.0 - a) * (255.0 - halo) / 255.0
    bloomed = np.clip(bloomed, 0, 255).astype(np.uint8)

    r = _CYN_R[bloomed]
    g = _CYN_G[bloomed]
    b = _CYN_B[bloomed]

    grain = np.random.normal(0, 5, gray.shape).astype(np.float32)
    r = np.clip(r.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) + grain * 0.5, 0, 255).astype(np.uint8)

    return Image.merge("RGB", (
        Image.fromarray(r),
        Image.fromarray(g),
        Image.fromarray(b),
    ))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil cyanotype bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./cyanotype-bot-output"))
    parser.add_argument("--no-post", action="store_true")
    parser.add_argument("--test-images", nargs=3, metavar="IMG", help="3 local image paths for testing (skips Slack fetch)")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token and not args.no_post and not args.test_images:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from stencil_transform import make_stencil, apply_stencil

    out_dir = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.test_images:
        logger.info("Using local test images...")
        images = [Image.open(p).convert("RGB") for p in args.test_images]
    else:
        from slack_fetcher import fetch_random_images
        source_dir = args.output_dir / "source"
        logger.info(f"Fetching 3 images from #{args.source_channel}...")
        source_paths = fetch_random_images(token, args.source_channel, 3, source_dir)
        images = [Image.open(p).convert("RGB") for p in source_paths]

    cyan_images = [to_cyanotype(img) for img in images]
    logger.info("Applied cyanotype treatment to all 3 images")

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as stencil, {a + 1} and {b + 1} as cyanotype fill...")
        mask = make_stencil(images[s])
        result = apply_stencil(mask, cyan_images[a], cyan_images[b])
        dest = out_dir / f"cyanotype_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    if not args.no_post and not args.test_images:
        from slack_poster import post_collages
        post_collages(token, args.post_channel, output_paths, bot_name="collage-stencil-cyanotype-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
