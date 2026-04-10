"""Collage lattice bot.

Fetches 3 images and weaves them as strips — A, B, C each appear as warp and
weft strips in a repeating cycle. Each strip has a sinusoidal path that S-curves
as it passes over/under crossing strips (physical weave undulation). Gap is black.

Produces 5 variations — one per weave structure:

1. Plain      — 1×1 checkerboard
2. Basket     — 2×2 block checkerboard
3. Twill      — 2/2 diagonal stripes at 45°
4. Herringbone — twill that mirrors at the centre column (chevron/V pattern)
5. Satin      — mostly warp, with weft appearing at isolated scattered binding points
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
                         n_strips: int, gap_px: float, weave: str) -> np.ndarray:
    """Weave img_a, img_b, img_c as strips with physical over/under undulation.

    Args:
        img_a, img_b, img_c : (H, W, 3) uint8 arrays, same size
        n_strips  : number of strips in each dimension
        gap_px    : gap width in pixels between strips (already resolution-scaled)
        weave     : weave structure name
    """
    h, w = img_a.shape[:2]
    xx = np.arange(w, dtype=np.float32)
    yy = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xx, yy)

    col_pitch = w / n_strips
    row_pitch = h / n_strips
    strip_half_w = (col_pitch - gap_px) / 2.0
    strip_half_h = (row_pitch - gap_px) / 2.0

    if strip_half_w < 1 or strip_half_h < 1:
        raise ValueError(f"Gap too large for {n_strips} strips at {w}×{h}")

    # Physical undulation: each strip S-curves as it passes over/under crossing strips.
    # Amplitude ~25% of strip half-width; period = 2 crossings (one over, one under).
    undulate_amp_col = strip_half_w * 0.25
    undulate_amp_row = strip_half_h * 0.25
    undulate_period_col = 2.0 * row_pitch  # warp undulates at weft-crossing frequency
    undulate_period_row = 2.0 * col_pitch  # weft undulates at warp-crossing frequency

    def col_center(j_arr):
        c = (j_arr.astype(np.float32) + 0.5) * col_pitch
        phase = np.pi * j_arr
        return c + undulate_amp_col * np.sin(2 * np.pi * yy / undulate_period_col + phase)

    def row_center(i_arr):
        c = (i_arr.astype(np.float32) + 0.5) * row_pitch
        phase = np.pi * i_arr
        return c + undulate_amp_row * np.sin(2 * np.pi * xx / undulate_period_row + phase)

    # Col (warp) strip membership — check nearest strip and its neighbour
    j  = (xx / col_pitch).astype(np.int32).clip(0, n_strips - 1)
    j1 = (j + 1).clip(0, n_strips - 1)
    dist_j  = np.abs(xx - col_center(j))
    dist_j1 = np.abs(xx - col_center(j1))
    in_j  = dist_j  <= strip_half_w
    in_j1 = dist_j1 <= strip_half_w
    both_col = in_j & in_j1
    in_j [both_col] = dist_j [both_col] <= dist_j1[both_col]
    in_j1[both_col] = ~in_j[both_col]
    col_idx      = np.where(in_j1, j1, j)
    in_col_strip = in_j | in_j1
    dist_col     = np.where(in_j1, dist_j1, dist_j)  # distance from assigned strip center

    # Row (weft) strip membership — same pattern, x↔y
    i  = (yy / row_pitch).astype(np.int32).clip(0, n_strips - 1)
    i1 = (i + 1).clip(0, n_strips - 1)
    dist_i  = np.abs(yy - row_center(i))
    dist_i1 = np.abs(yy - row_center(i1))
    in_i  = dist_i  <= strip_half_h
    in_i1 = dist_i1 <= strip_half_h
    both_row = in_i & in_i1
    in_i [both_row] = dist_i [both_row] <= dist_i1[both_row]
    in_i1[both_row] = ~in_i[both_row]
    row_idx      = np.where(in_i1, i1, i)
    in_row_strip = in_i | in_i1
    dist_row     = np.where(in_i1, dist_i1, dist_i)

    # 3-image cycling assignment
    # Col strips: 0→A, 1→B, 2→C  (by col_idx % 3)
    # Row strips: 0→B, 1→C, 2→A  (offset by 1 for visual variety)
    cm = col_idx[:, :, np.newaxis] % 3
    col_img = np.where(cm == 0, img_a, np.where(cm == 1, img_b, img_c))
    rm = row_idx[:, :, np.newaxis] % 3
    row_img = np.where(rm == 0, img_b, np.where(rm == 1, img_c, img_a))

    # Weave: use_col = True means col (warp) strip is on top at this crossing
    n = n_strips
    if weave == 'plain':
        use_col = (row_idx + col_idx) % 2 == 0
    elif weave == 'basket':
        use_col = (row_idx // 2 + col_idx // 2) % 2 == 0
    elif weave == 'twill':
        use_col = (row_idx + col_idx) % 4 >= 2
    elif weave == 'herringbone':
        use_col = np.where(col_idx < n // 2,
                           (row_idx + col_idx) % 4 >= 2,
                           (row_idx + (n - 1 - col_idx)) % 4 >= 2)
    elif weave == 'satin':
        use_col = (col_idx - row_idx * 2) % 5 != 0
    else:
        use_col = np.ones((h, w), dtype=bool)

    # Composite: gap = black, col-only, row-only, crossing (weave-determined)
    at_crossing = in_col_strip & in_row_strip
    col_only    = in_col_strip & ~in_row_strip
    row_only    = in_row_strip & ~in_col_strip

    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[col_only] = col_img[col_only]
    result[row_only] = row_img[row_only]
    result[at_crossing] = np.where(
        use_col[at_crossing, np.newaxis],
        col_img[at_crossing],
        row_img[at_crossing]
    )

    # Over/under shadow: darkens the over-strip near where the under-strip enters
    shadow_decay = max(gap_px * 2.5, 4.0)
    # col on top: under-strip (row) enters at row strip edges → shadow near those edges
    shadow_col_on_top = (np.exp(-(strip_half_h - dist_row.clip(0, strip_half_h)) / shadow_decay)) * 0.45
    # row on top: under-strip (col) enters at col strip edges → shadow near those edges
    shadow_row_on_top = (np.exp(-(strip_half_w - dist_col.clip(0, strip_half_w)) / shadow_decay)) * 0.45

    shadow_map = np.where(use_col, shadow_col_on_top, shadow_row_on_top)
    shadow_map = np.clip(shadow_map, 0.0, 0.45)

    result_f = result.astype(np.float32)
    result_f[at_crossing] *= (1.0 - shadow_map[at_crossing, np.newaxis])
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
    logger.info(f"Gap: {gap}px, strips: {args.n_strips}")

    img_a = np.array(images[0].resize((target_w, target_h), Image.LANCZOS))
    img_b = np.array(images[1].resize((target_w, target_h), Image.LANCZOS))
    img_c = np.array(images[2].resize((target_w, target_h), Image.LANCZOS))

    weaves = ['plain', 'basket', 'twill', 'herringbone', 'satin']

    output_paths = []
    for i, weave in enumerate(weaves):
        logger.info(f"Version {i + 1}: {weave} weave...")
        result = make_organic_lattice(img_a, img_b, img_c, args.n_strips, gap, weave)
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
