"""Collage lattice bot.

Fetches 3 images and weaves A and B into a lattice of strips. C shows through
the gaps between strips. Produces 5 variations — one per weave structure:

1. Plain      — 1×1 checkerboard
2. Basket     — 2×2 block checkerboard
3. Twill      — 2/2 diagonal stripes at 45°
4. Herringbone — twill that mirrors at the centre column (chevron/V pattern)
5. Satin      — mostly A, with B appearing at isolated scattered binding points
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def weave_cell(row: int, col: int, weave: str, n: int) -> bool:
    """Return True if cell (row, col) should show image A."""
    if weave == 'plain':
        return (row + col) % 2 == 0
    elif weave == 'basket':
        return (row // 2 + col // 2) % 2 == 0
    elif weave == 'twill':
        # 2/2 twill: diagonal stripes 2 cells wide
        return (row + col) % 4 >= 2
    elif weave == 'herringbone':
        # Twill that reverses at the centre column → chevron
        mid = n // 2
        if col < mid:
            return (row + col) % 4 >= 2
        else:
            return (row + (n - 1 - col)) % 4 >= 2
    elif weave == 'satin':
        # 5-end satin: B at isolated binding points, A everywhere else
        return (col - row * 2) % 5 != 0
    return True


def make_lattice(img_a: np.ndarray, img_b: np.ndarray, img_c: np.ndarray,
                 n_strips: int, gap: int, weave: str) -> np.ndarray:
    """Weave img_a and img_b into a lattice with img_c showing through the gaps.

    Args:
        img_a, img_b, img_c : (H, W, 3) uint8 arrays, same size
        n_strips : number of strips in each dimension
        gap      : width of the gap between strips in pixels
        weave    : weave structure name
    """
    h, w = img_a.shape[:2]
    cell_w = (w - (n_strips + 1) * gap) // n_strips
    cell_h = (h - (n_strips + 1) * gap) // n_strips

    if cell_w < 1 or cell_h < 1:
        raise ValueError(f"Gap too large for {n_strips} strips at {w}×{h}")

    # Canvas starts as img_c — gaps reveal it naturally
    result = img_c.copy()

    for row in range(n_strips):
        for col in range(n_strips):
            x1 = gap + col * (cell_w + gap)
            y1 = gap + row * (cell_h + gap)
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            src = img_a if weave_cell(row, col, weave, n_strips) else img_b
            result[y1:y2, x1:x2] = src[y1:y2, x1:x2]

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage lattice bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./lattice-bot-output"))
    parser.add_argument("--n-strips", type=int, default=10,
                        help="Number of strips in each dimension")
    parser.add_argument("--gap", type=int, default=6,
                        help="Gap width between strips in pixels (at 1024px short side)")
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

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))
    images = [Image.open(p).convert("RGB") for p in source_paths]

    # All images at the same size — use the largest
    target_w, target_h = max((img.size for img in images), key=lambda s: s[0] * s[1])
    logger.info(f"Output resolution: {target_w}×{target_h}")

    # Scale gap to output resolution
    res_scale = min(target_w, target_h) / 1024.0
    gap = max(2, int(args.gap * res_scale))
    logger.info(f"Gap: {gap}px")

    img_a = np.array(images[0].resize((target_w, target_h), Image.LANCZOS))
    img_b = np.array(images[1].resize((target_w, target_h), Image.LANCZOS))
    img_c = np.array(images[2].resize((target_w, target_h), Image.LANCZOS))

    weaves = ['plain', 'basket', 'twill', 'herringbone', 'satin']

    output_paths = []
    for i, weave in enumerate(weaves):
        logger.info(f"Version {i + 1}: {weave} weave...")
        result = make_lattice(img_a, img_b, img_c, args.n_strips, gap, weave)
        dest = out_dir / f"lattice_result_{i + 1}_{weave}.png"
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
