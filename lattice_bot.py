"""Collage lattice bot.

Fetches 3 images and weaves A and B into a lattice of strips with organic,
wavy boundaries. C shows through the gaps between strips.
Produces 5 variations — one per weave structure:

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


def make_organic_lattice(img_a: np.ndarray, img_b: np.ndarray, img_c: np.ndarray,
                         n_strips: int, gap_px: float, weave: str,
                         warp_strength: float = 0.4, warp_freq: float = 2.0) -> np.ndarray:
    """Weave img_a and img_b into a lattice with wavy organic strip boundaries.

    Args:
        img_a, img_b, img_c : (H, W, 3) uint8 arrays, same size
        n_strips    : number of strips in each dimension
        gap_px      : gap width in pixels (already resolution-scaled)
        weave       : weave structure name
        warp_strength : warp amplitude as fraction of cell size (0 = straight)
        warp_freq   : number of sine cycles across the full image width/height
    """
    h, w = img_a.shape[:2]
    xx = np.arange(w, dtype=np.float32)
    yy = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xx, yy)

    cell_size = min(h, w) / n_strips
    amplitude = cell_size * warp_strength

    # Warp effective coords — different phases so row and col warps look independent
    y_eff = yy + amplitude * np.sin(2 * np.pi * warp_freq * xx / w)
    x_eff = xx + amplitude * np.sin(2 * np.pi * warp_freq * yy / h + np.pi * 0.7)

    row_period = h / n_strips
    col_period = w / n_strips

    y_frac = y_eff % row_period
    x_frac = x_eff % col_period

    half_gap = gap_px / 2.0
    in_gap = (y_frac < half_gap) | (y_frac > row_period - half_gap) \
           | (x_frac < half_gap) | (x_frac > col_period - half_gap)

    row_idx = (y_eff / row_period).astype(np.int32).clip(0, n_strips - 1)
    col_idx = (x_eff / col_period).astype(np.int32).clip(0, n_strips - 1)
    n = n_strips

    if weave == 'plain':
        use_a = (row_idx + col_idx) % 2 == 0
    elif weave == 'basket':
        use_a = (row_idx // 2 + col_idx // 2) % 2 == 0
    elif weave == 'twill':
        use_a = (row_idx + col_idx) % 4 >= 2
    elif weave == 'herringbone':
        use_a = np.where(col_idx < n // 2,
                         (row_idx + col_idx) % 4 >= 2,
                         (row_idx + (n - 1 - col_idx)) % 4 >= 2)
    elif weave == 'satin':
        use_a = (col_idx - row_idx * 2) % 5 != 0
    else:
        use_a = np.ones((h, w), dtype=bool)

    result = img_c.copy()
    active = ~in_gap
    result[active] = np.where(
        use_a[active, np.newaxis],
        img_a[active],
        img_b[active]
    )

    # --- Over/under shadow ---
    cell_h = row_period - gap_px
    cell_w = col_period - gap_px

    # Normalized position within the active cell region [0, 1]
    cell_y = ((y_frac - half_gap) / cell_h).clip(0, 1)
    cell_x = ((x_frac - half_gap) / cell_w).clip(0, 1)

    # Physical pixel distance from each edge
    dist_top    = cell_y * cell_h
    dist_bottom = (1.0 - cell_y) * cell_h
    dist_left   = cell_x * cell_w
    dist_right  = (1.0 - cell_x) * cell_w

    shadow_decay = max(gap_px * 3.0, 4.0)
    shadow_strength = 0.45

    # col on top → under-strip enters from top/bottom → darken those edges
    shadow_col_on_top = (np.exp(-dist_top / shadow_decay) +
                         np.exp(-dist_bottom / shadow_decay)) * shadow_strength

    # row on top → under-strip enters from left/right → darken those edges
    shadow_row_on_top = (np.exp(-dist_left / shadow_decay) +
                         np.exp(-dist_right / shadow_decay)) * shadow_strength

    shadow_map = np.where(use_a, shadow_col_on_top, shadow_row_on_top)
    shadow_map = np.clip(shadow_map, 0.0, shadow_strength)

    result_f = result.astype(np.float32)
    result_f[active] *= (1.0 - shadow_map[active, np.newaxis])
    result = result_f.clip(0, 255).astype(np.uint8)

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage lattice bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./lattice-bot-output"))
    parser.add_argument("--n-strips", type=int, default=20,
                        help="Number of strips in each dimension")
    parser.add_argument("--gap", type=int, default=6,
                        help="Gap width between strips in pixels (at 1024px short side)")
    parser.add_argument("--warp-strength", type=float, default=0.4,
                        help="Warp amplitude as fraction of cell size (0=straight)")
    parser.add_argument("--warp-freq", type=float, default=2.0,
                        help="Sine cycles across the image")
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
    logger.info(f"Gap: {gap}px, strips: {args.n_strips}, "
                f"warp: strength={args.warp_strength} freq={args.warp_freq}")

    img_a = np.array(images[0].resize((target_w, target_h), Image.LANCZOS))
    img_b = np.array(images[1].resize((target_w, target_h), Image.LANCZOS))
    img_c = np.array(images[2].resize((target_w, target_h), Image.LANCZOS))

    weaves = ['plain', 'basket', 'twill', 'herringbone', 'satin']

    output_paths = []
    for i, weave in enumerate(weaves):
        logger.info(f"Version {i + 1}: {weave} weave...")
        result = make_organic_lattice(img_a, img_b, img_c, args.n_strips, gap, weave,
                                      warp_strength=args.warp_strength,
                                      warp_freq=args.warp_freq)
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
