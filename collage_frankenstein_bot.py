"""Collage Frankenstein bot.

Fetches 9 source images from Slack. Standardizes each to a square, then
slices into a 3×3 grid of 9 equal square quadrants (81 quadrants total).

Reassembles 9 output collages using a Latin-square distribution so that:
  - Each output gets exactly one quadrant from each of the 9 source images.
  - Every quadrant is used in exactly one output (no repeats, none wasted).

Before placing each quadrant the bot tries all 16 transformations (8 geometric
orientations × 2 colour states: normal and inverted) and picks whichever
minimises pixel-level discontinuity along the shared edges with already-placed
neighbours. Assembly proceeds in raster order so every new tile can be scored
against the tile above and the tile to its left.

Posts all 9 outputs as a single Slack message.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transformation catalogue: (transpose_mode_or_None, invert_colours)
# The 8 geometric modes are the full dihedral group D4 (rotations + reflections).
# Paired with colour inversion gives 16 total transformations per quadrant.
# ---------------------------------------------------------------------------
_TRANSPOSE_MODES = [
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.TRANSVERSE,
]
TRANSFORMS = [(t, inv) for t in _TRANSPOSE_MODES for inv in (False, True)]


def standardize(img: Image.Image, size: int) -> Image.Image:
    """Center-crop to square then resize to size×size."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)


def slice_quadrants(img: Image.Image) -> list[Image.Image]:
    """Cut a square image into a 3×3 grid. Returns 9 PIL Images in row-major order."""
    w, h = img.size
    assert w == h, "image must be square before slicing"
    q = w // 3
    return [
        img.crop((c * q, r * q, (c + 1) * q, (r + 1) * q))
        for r in range(3)
        for c in range(3)
    ]


def apply_transform(tile: Image.Image, transpose, invert: bool) -> Image.Image:
    if transpose is not None:
        tile = tile.transpose(transpose)
    if invert:
        tile = ImageOps.invert(tile)
    return tile


def edge_score(
    candidate: np.ndarray,
    above: np.ndarray | None,
    left: np.ndarray | None,
    depth: int = 8,
) -> float:
    """Weighted mean absolute pixel difference across a strip of pixels on each
    shared edge. Pixels closer to the boundary are weighted more heavily than
    those further in, so the score reflects gradient continuity not just the
    single border row/column.

    depth: how many pixel rows/columns from each edge to include in scoring.
    """
    score = 0.0

    # Weights: linear ramp, boundary pixel = depth, innermost pixel = 1
    weights = np.arange(depth, 0, -1, dtype=np.float32)  # [depth, depth-1, ..., 1]
    weight_sum = float(weights.sum())

    if above is not None:
        # Compare the top `depth` rows of candidate against the bottom `depth`
        # rows of the tile above. Flip above's strip so row 0 is the boundary.
        cand_strip = candidate[:depth].astype(np.float32)       # (depth, W, 3)
        above_strip = above[-depth:][::-1].astype(np.float32)   # (depth, W, 3)
        diff = np.abs(cand_strip - above_strip)                  # (depth, W, 3)
        row_means = diff.mean(axis=(1, 2))                       # (depth,)
        score += float((row_means * weights).sum() / weight_sum)

    if left is not None:
        # Compare the left `depth` columns of candidate against the right
        # `depth` columns of the tile to the left.
        cand_strip = candidate[:, :depth].astype(np.float32)       # (H, depth, 3)
        left_strip = left[:, -depth:][:, ::-1].astype(np.float32)  # (H, depth, 3)
        diff = np.abs(cand_strip - left_strip)                      # (H, depth, 3)
        col_means = diff.mean(axis=(0, 2))                          # (depth,)
        score += float((col_means * weights).sum() / weight_sum)

    return score


def make_latin_square(n: int, rng: np.random.Generator) -> list[list[int]]:
    """Return an n×n Latin square with values 0..n-1, rows and columns shuffled."""
    # Base: cyclic shift construction (classic Latin square)
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    # Shuffle rows, then columns independently
    row_order = rng.permutation(n).tolist()
    col_order = rng.permutation(n).tolist()
    return [[base[r][c] for c in col_order] for r in row_order]


def assemble_output(
    pieces: list[tuple[int, int, Image.Image]],  # (src_idx, quad_pos, tile)
    rng: np.random.Generator,
    quad_size: int,
) -> Image.Image:
    """Greedily assemble one 3×3 output from 9 pieces, maximising edge continuity.

    pieces: list of (src_idx, quad_pos, PIL Image) — one per grid slot to fill.
    Assembly order: raster scan (position 0..8).
    For each slot: try all remaining pieces × 16 transforms, pick best edge score.
    """
    canvas = Image.new("RGB", (quad_size * 3, quad_size * 3))
    placed: list[np.ndarray | None] = [None] * 9  # numpy arrays of placed tiles
    remaining = list(range(len(pieces)))           # indices into `pieces`

    # Shuffle remaining so ties are broken randomly (avoids always favouring
    # lower-indexed sources when scores are equal).
    rng.shuffle(remaining)

    for slot in range(9):
        row, col = divmod(slot, 3)
        above_arr = placed[slot - 3] if row > 0 else None
        left_arr  = placed[slot - 1] if col > 0 else None

        best_score = float("inf")
        best_piece_idx = None
        best_arr = None
        best_img = None

        for piece_list_idx in remaining:
            tile = pieces[piece_list_idx][2]
            for transpose, invert in TRANSFORMS:
                candidate = apply_transform(tile, transpose, invert)
                arr = np.array(candidate)
                score = edge_score(arr, above_arr, left_arr)
                if score < best_score:
                    best_score = score
                    best_piece_idx = piece_list_idx
                    best_arr = arr
                    best_img = candidate

        # Place the winner
        remaining.remove(best_piece_idx)
        placed[slot] = best_arr
        canvas.paste(best_img, (col * quad_size, row * quad_size))

    return canvas


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage Frankenstein bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./frankenstein-bot-output"))
    parser.add_argument("--size", type=int, default=900,
                        help="Square pixel dimension to standardize each source image to (default: 900)")
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

    rng = np.random.default_rng()
    quad_size = args.size // 3

    # ------------------------------------------------------------------
    # Step 1: Fetch & standardize
    # ------------------------------------------------------------------
    logger.info(f"Fetching 9 images from #{args.source_channel}...")
    source_paths = fetch_random_images(token, args.source_channel, 9, source_dir)
    images = [standardize(Image.open(p).convert("RGB"), args.size) for p in source_paths]
    logger.info(f"Standardized {len(images)} images to {args.size}×{args.size}")

    # ------------------------------------------------------------------
    # Step 2: Slice each image into 9 quadrants
    # quadrants[src_idx][quad_pos] = PIL Image (quad_size × quad_size)
    # ------------------------------------------------------------------
    quadrants = [slice_quadrants(img) for img in images]
    logger.info(f"Sliced into {len(quadrants) * 9} quadrants ({quad_size}×{quad_size} each)")

    # ------------------------------------------------------------------
    # Step 3: Latin square assignment
    # assignment[output_i][src_j] = quadrant position used from source j in output i
    # ------------------------------------------------------------------
    latin = make_latin_square(9, rng)

    # ------------------------------------------------------------------
    # Step 4 & 5: Assemble each output and save
    # ------------------------------------------------------------------
    output_paths = []
    for out_i in range(9):
        # Build the list of pieces for this output:
        # one (src_idx, quad_pos, tile) per source image
        pieces = [
            (src_j, latin[out_i][src_j], quadrants[src_j][latin[out_i][src_j]])
            for src_j in range(9)
        ]
        logger.info(f"Assembling output {out_i + 1}/9...")
        result = assemble_output(pieces, rng, quad_size)
        dest = out_dir / f"frankenstein_output_{out_i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    # ------------------------------------------------------------------
    # Step 6: Post
    # ------------------------------------------------------------------
    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="collage-frankenstein", threaded=False)
        logger.info(f"Posted {len(output_paths)} outputs to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
