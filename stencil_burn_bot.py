"""Collage stencil burn bot.

Fetches 4 images. Images 1–3 each take a turn as stencil. Image 4 is used as
the burn/dodge layer applied to the two fill images:

  1. b2 = color_burn(base=img_p, blend=img_d)   — fill p darkened by image 4
     c2 = color_dodge(base=img_q, blend=img_d)  — fill q brightened by image 4
  2. Fill the stencil with b2 and c2 (two variations, swapped):
       result_1: b2 in white regions, c2 in black regions
       result_2: c2 in white regions, b2 in black regions

Posts 6 images: 2 results per stencil × 3 stencils.
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


def color_dodge(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Color dodge blend mode — exact opposite of color burn. Returns uint8."""
    b = base.astype(np.float32) / 255.0
    s = blend.astype(np.float32) / 255.0
    result = np.clip(b / np.maximum(1.0 - s, 1e-6), 0.0, 1.0)
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

    logger.info(f"Fetching 4 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 4, source_dir))
    images = [Image.open(p).convert("RGB") for p in source_paths]
    logger.info(f"Source sizes: {[img.size for img in images]}")

    # Each image takes a turn as stencil; the other two are the fill images.
    # (s=stencil index, p=fill image "b", q=fill image "c")
    triples = [
        (0, 1, 2),
        (1, 0, 2),
        (2, 0, 1),
    ]

    output_paths = []
    for stencil_num, (s, p, q) in enumerate(triples, start=1):
        logger.info(f"Stencil {stencil_num}: image {s + 1} as stencil...")
        mask = make_stencil(images[s])

        img_p = images[p]
        img_q = images[q]
        img_d = images[3]  # 4th image is the burn/dodge layer

        arr_p = np.array(img_p)
        arr_q = np.array(img_q)
        # Resize D to match each fill image before applying blend
        arr_d_for_p = np.array(img_d.resize(img_p.size, Image.LANCZOS))
        arr_d_for_q = np.array(img_d.resize(img_q.size, Image.LANCZOS))

        b2 = Image.fromarray(color_burn(base=arr_p, blend=arr_d_for_p))   # p darkened by D
        c2 = Image.fromarray(color_dodge(base=arr_q, blend=arr_d_for_q))  # q brightened by D
        logger.info(f"  b2=burn(img{p+1}, img4), c2=dodge(img{q+1}, img4)")

        # Two stencil variations using the burned fills
        result_1 = apply_stencil(mask, b2, c2)
        result_2 = apply_stencil(mask, c2, b2)

        path_1 = out_dir / f"burn_result_{stencil_num}a.png"
        path_2 = out_dir / f"burn_result_{stencil_num}b.png"
        result_1.save(path_1)
        result_2.save(path_2)
        logger.info(f"  Saved {path_1.name}, {path_2.name}")

        output_paths += [path_1, path_2]

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="collage-stencil-burn-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
