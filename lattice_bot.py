"""Collage lattice bot.

Fetches 2 images and weaves them into a lattice of strips. Produces 4 variations:

1. Plain weave (1×1 checkerboard) — image A on top at even cells
2. Plain weave                     — image B on top at even cells
3. Basket weave (2×2 blocks)       — image A on top at even blocks
4. Basket weave (2×2 blocks)       — image B on top at even blocks
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def make_lattice(img_a: np.ndarray, img_b: np.ndarray,
                 n_strips: int, weave: str, a_on_top: bool) -> np.ndarray:
    """Weave img_a and img_b into a lattice.

    Args:
        img_a, img_b : (H, W, 3) uint8 arrays, same size
        n_strips     : number of strips in each dimension
        weave        : 'plain' (1×1) or 'basket' (2×2)
        a_on_top     : if True, img_a occupies the 'on top' cells
    """
    h, w = img_a.shape[:2]
    cell_h = h // n_strips
    cell_w = w // n_strips
    # Crop to exact multiple
    h_crop = cell_h * n_strips
    w_crop = cell_w * n_strips
    a = img_a[:h_crop, :w_crop]
    b = img_b[:h_crop, :w_crop]

    result = np.empty_like(a)

    for row in range(n_strips):
        for col in range(n_strips):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w

            if weave == 'plain':
                on_top = (row + col) % 2 == 0
            else:  # basket 2×2
                on_top = (row // 2 + col // 2) % 2 == 0

            if on_top == a_on_top:
                result[y1:y2, x1:x2] = a[y1:y2, x1:x2]
            else:
                result[y1:y2, x1:x2] = b[y1:y2, x1:x2]

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage lattice bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./lattice-bot-output"))
    parser.add_argument("--n-strips", type=int, default=10,
                        help="Number of strips in each dimension")
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

    logger.info(f"Fetching 2 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 2, source_dir))
    images = [Image.open(p).convert("RGB") for p in source_paths]

    # Both images at the same size — use the larger one
    target_w, target_h = max((img.size for img in images), key=lambda s: s[0] * s[1])
    logger.info(f"Output resolution: {target_w}×{target_h}")

    img_a = np.array(images[0].resize((target_w, target_h), Image.LANCZOS))
    img_b = np.array(images[1].resize((target_w, target_h), Image.LANCZOS))

    variations = [
        ("plain",  True,  "plain weave, image 1 on top"),
        ("plain",  False, "plain weave, image 2 on top"),
        ("basket", True,  "basket weave, image 1 on top"),
        ("basket", False, "basket weave, image 2 on top"),
    ]

    output_paths = []
    for i, (weave, a_on_top, label) in enumerate(variations):
        logger.info(f"Version {i + 1}: {label}...")
        result = make_lattice(img_a, img_b, args.n_strips, weave, a_on_top)
        dest = out_dir / f"lattice_result_{i + 1}.png"
        Image.fromarray(result).save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="collage-lattice-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
