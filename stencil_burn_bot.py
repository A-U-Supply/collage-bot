"""Collage stencil burn bot.

Fetches 3 images. For each image used as a stencil, generates two variations
(images B and C swapped between white/black regions), burns each against the
other, then re-applies the original mask so the stencil boundaries remain sharp:

  white region → color_burn(var_a, var_b)   (img_a base, img_b burn)
  black region → color_burn(var_b, var_a)   (img_b base, img_a burn)

Posts 9 images: 3 pairs of (var_a, var_b, burned_result), grouped by stencil.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def color_burn(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Color burn blend mode. Both inputs: uint8 (H, W, 3). Returns uint8."""
    b = base.astype(np.float32) / 255.0
    s = blend.astype(np.float32) / 255.0
    result = 1.0 - np.clip((1.0 - b) / np.maximum(s, 1e-6), 0.0, 1.0)
    return (result * 255.0).clip(0, 255).astype(np.uint8)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil burn bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./stencil-burn-bot-output"))
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages
    from stencil_transform import apply_stencil, make_stencil

    source_dir = args.output_dir / "source"
    out_dir = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))
    images = [Image.open(p).convert("RGB") for p in source_paths]

    # Each image takes a turn as stencil; the other two fill white/black regions.
    # var_a: images[a] in white, images[b] in black
    # var_b: images[b] in white, images[a] in black  (swapped)
    # burned: color-burn(var_a, var_b)
    pairs = [
        (0, 1, 2),
        (1, 0, 2),
        (2, 0, 1),
    ]

    output_paths = []
    for pair_num, (s, a, b) in enumerate(pairs, start=1):
        logger.info(f"Pair {pair_num}: image {s + 1} as stencil...")
        mask = make_stencil(images[s])

        var_a = apply_stencil(mask, images[a], images[b])
        var_b = apply_stencil(mask, images[b], images[a])

        var_a_path = out_dir / f"burn_var_{pair_num}a.png"
        var_b_path = out_dir / f"burn_var_{pair_num}b.png"
        var_a.save(var_a_path)
        var_b.save(var_b_path)
        logger.info(f"Saved {var_a_path.name}, {var_b_path.name}")

        # Burn each variation against the other, then re-apply the original mask
        # so stencil boundaries stay sharp and each region looks distinct.
        burned_a_over_b = color_burn(np.array(var_a), np.array(var_b))
        burned_b_over_a = color_burn(np.array(var_b), np.array(var_a))
        burned = apply_stencil(mask,
                               Image.fromarray(burned_a_over_b),
                               Image.fromarray(burned_b_over_a))
        burned_path = out_dir / f"burn_result_{pair_num}.png"
        burned.save(burned_path)
        logger.info(f"Saved {burned_path.name}")

        output_paths += [var_a_path, var_b_path, burned_path]

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="collage-stencil-burn-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
