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

After placement, image quilting (Efros & Freeman minimum-cost seam cutting) is
applied at every internal tile boundary. Adjacent tiles overlap by OVERLAP pixels;
a dynamic-programming seam finds the path of least pixel difference through each
overlap zone so the hard grid boundary is replaced by a natural edge that follows
existing image structure.

Output canvas size = 3 * quad_size - 2 * OVERLAP
(e.g. --size 900 → quad_size=300, OVERLAP=20 → 860×860 outputs).

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

# Pixels of overlap at each internal seam for image quilting.
OVERLAP = 40

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
    weights = np.arange(depth, 0, -1, dtype=np.float32)  # [depth, depth-1, ..., 1]
    weight_sum = float(weights.sum())

    if above is not None:
        cand_strip = candidate[:depth].astype(np.float32)       # (depth, W, 3)
        above_strip = above[-depth:][::-1].astype(np.float32)   # (depth, W, 3)
        diff = np.abs(cand_strip - above_strip)                  # (depth, W, 3)
        row_means = diff.mean(axis=(1, 2))                       # (depth,)
        score += float((row_means * weights).sum() / weight_sum)

    if left is not None:
        cand_strip = candidate[:, :depth].astype(np.float32)       # (H, depth, 3)
        left_strip = left[:, -depth:][:, ::-1].astype(np.float32)  # (H, depth, 3)
        diff = np.abs(cand_strip - left_strip)                      # (H, depth, 3)
        col_means = diff.mean(axis=(0, 2))                          # (depth,)
        score += float((col_means * weights).sum() / weight_sum)

    return score


# ---------------------------------------------------------------------------
# Image quilting — minimum-cost seam cutting (Efros & Freeman 2001)
# ---------------------------------------------------------------------------

def _min_cost_seam(error: np.ndarray) -> np.ndarray:
    """Dynamic-programming minimum-cost vertical seam.

    error: (H, W) float array — per-pixel cost.
    Returns a (H,) int array of column indices (0..W-1), one per row,
    tracing the lowest-cost connected path from top to bottom.
    """
    H, W = error.shape
    cost = error.copy().astype(np.float64)
    for r in range(1, H):
        prev = cost[r - 1]
        shifted_l = np.empty_like(prev)
        shifted_l[0] = np.inf
        shifted_l[1:] = prev[:-1]
        shifted_r = np.empty_like(prev)
        shifted_r[-1] = np.inf
        shifted_r[:-1] = prev[1:]
        cost[r] += np.minimum(prev, np.minimum(shifted_l, shifted_r))

    seam = np.zeros(H, dtype=int)
    seam[-1] = int(np.argmin(cost[-1]))
    for r in range(H - 2, -1, -1):
        c = seam[r + 1]
        lo, hi = max(0, c - 1), min(W - 1, c + 1)
        seam[r] = lo + int(np.argmin(cost[r, lo : hi + 1]))
    return seam


def _quilt_vertical(
    canvas: np.ndarray,
    left_arr: np.ndarray,
    right_arr: np.ndarray,
    canvas_y: int,
    x_boundary: int,
    overlap: int,
) -> None:
    """Recomposite the vertical overlap zone between two horizontally adjacent tiles.

    Because tiles are placed with spatial overlap, the right tile's pixels already
    cover the entire overlap zone on the canvas. The seam decides where to switch
    from left to right — we write the full composite, restoring left tile pixels
    left of the seam and right tile pixels right of it (hard cut, no blending).

    canvas_y: top canvas row of this tile row.
    x_boundary: canvas x where the left tile ends.
    """
    H = left_arr.shape[0]
    left_strip = left_arr[:, -overlap:].astype(np.float32)    # (H, overlap, 3)
    right_strip = right_arr[:, :overlap].astype(np.float32)   # (H, overlap, 3)
    error = np.mean((left_strip - right_strip) ** 2, axis=2)  # (H, overlap)
    seam = _min_cost_seam(error)                               # (H,) in [0, overlap)

    # Build (H, overlap) alpha: 0 = pure left tile, 1 = pure right tile.
    # Hard cut at the seam path — no blending.
    col_idx = np.arange(overlap)[np.newaxis, :]  # (1, overlap)
    seam_col = seam[:, np.newaxis]               # (H, 1)
    alpha = (col_idx >= seam_col).astype(np.float32)
    alpha = alpha[:, :, np.newaxis]  # (H, overlap, 1) for RGB broadcast

    blended = ((1.0 - alpha) * left_strip + alpha * right_strip).astype(np.uint8)
    zone_x = x_boundary - overlap
    canvas[canvas_y : canvas_y + H, zone_x : zone_x + overlap] = blended


def _quilt_horizontal(
    canvas: np.ndarray,
    top_arr: np.ndarray,
    bottom_arr: np.ndarray,
    canvas_x: int,
    y_boundary: int,
    overlap: int,
) -> None:
    """Recomposite the horizontal overlap zone between two vertically adjacent tiles.

    Because tiles are placed with spatial overlap, the bottom tile's pixels already
    cover the overlap zone. We write the full seam composite, restoring top tile
    pixels above the seam and bottom tile pixels below it (hard cut, no blending).

    canvas_x: left canvas column of this tile column.
    y_boundary: canvas y where the top tile ends.
    """
    W = top_arr.shape[1]
    top_strip = top_arr[-overlap:, :].astype(np.float32)       # (overlap, W, 3)
    bottom_strip = bottom_arr[:overlap, :].astype(np.float32)  # (overlap, W, 3)
    # Transpose so the DP seam gives a row-cut value per column.
    error = np.mean((top_strip - bottom_strip) ** 2, axis=2).T  # (W, overlap)
    seam = _min_cost_seam(error)                                 # (W,) in [0, overlap)

    row_idx = np.arange(overlap)[:, np.newaxis]  # (overlap, 1)
    seam_row = seam[np.newaxis, :]               # (1, W)
    alpha = (row_idx >= seam_row).astype(np.float32)
    alpha = alpha[:, :, np.newaxis]  # (overlap, W, 1)

    blended = ((1.0 - alpha) * top_strip + alpha * bottom_strip).astype(np.uint8)
    zone_y = y_boundary - overlap
    canvas[zone_y : zone_y + overlap, canvas_x : canvas_x + W] = blended


def make_latin_square(n: int, rng: np.random.Generator) -> list[list[int]]:
    """Return an n×n Latin square with values 0..n-1, rows and columns shuffled."""
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    row_order = rng.permutation(n).tolist()
    col_order = rng.permutation(n).tolist()
    return [[base[r][c] for c in col_order] for r in row_order]


def assemble_output(
    pieces: list[tuple[int, int, Image.Image]],
    rng: np.random.Generator,
    quad_size: int,
    overlap: int,
) -> Image.Image:
    """Greedily assemble one 3×3 output, then apply image-quilting seams.

    Phase 1 — greedy placement: raster-order selection of the best (piece,
    transform) pair for each slot, scored by weighted edge continuity.

    Phase 2 — quilting: tiles are placed with `overlap` pixels of spatial
    overlap at each internal boundary. Minimum-cost seam cutting then
    repaints each overlap zone so the boundary follows natural image edges
    rather than a hard grid line.
    """
    step = quad_size - overlap                      # pixels between tile origins
    canvas_size = quad_size + 2 * step              # = 3*quad_size - 2*overlap

    canvas = Image.new("RGB", (canvas_size, canvas_size))
    placed: list[np.ndarray | None] = [None] * 9   # for edge scoring
    grid: list[list[np.ndarray | None]] = [[None] * 3 for _ in range(3)]
    remaining = list(range(len(pieces)))
    rng.shuffle(remaining)

    # ------------------------------------------------------------------
    # Phase 1: greedy tile selection + placement
    # ------------------------------------------------------------------
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

        remaining.remove(best_piece_idx)
        placed[slot] = best_arr
        grid[row][col] = best_arr
        canvas.paste(best_img, (col * step, row * step))

    # ------------------------------------------------------------------
    # Phase 2: image quilting — recomposite overlap zones with seam cuts
    # ------------------------------------------------------------------
    canvas_arr = np.array(canvas)

    # Vertical seams (between adjacent columns within each tile row)
    for tile_row in range(3):
        canvas_y = tile_row * step
        for col_seam in range(1, 3):
            # Left tile ends at: (col_seam-1)*step + quad_size
            x_boundary = (col_seam - 1) * step + quad_size
            _quilt_vertical(
                canvas_arr,
                grid[tile_row][col_seam - 1],
                grid[tile_row][col_seam],
                canvas_y=canvas_y,
                x_boundary=x_boundary,
                overlap=overlap,
            )

    # Horizontal seams (between adjacent rows within each tile column)
    for row_seam in range(1, 3):
        # Top tile ends at: (row_seam-1)*step + quad_size
        y_boundary = (row_seam - 1) * step + quad_size
        for tile_col in range(3):
            canvas_x = tile_col * step
            _quilt_horizontal(
                canvas_arr,
                grid[row_seam - 1][tile_col],
                grid[row_seam][tile_col],
                canvas_x=canvas_x,
                y_boundary=y_boundary,
                overlap=overlap,
            )

    return Image.fromarray(canvas_arr)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage Frankenstein bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./frankenstein-bot-output"))
    parser.add_argument("--size", type=int, default=900,
                        help="Square pixel dimension to standardize each source image to (default: 900)")
    parser.add_argument("--overlap", type=int, default=OVERLAP,
                        help=f"Overlap pixels for image-quilting seams (default: {OVERLAP}). "
                             "Output canvas = 3*(size//3) - 2*overlap.")
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

    logger.info(f"Fetching 9 images from #{args.source_channel}...")
    source_paths = fetch_random_images(token, args.source_channel, 9, source_dir)
    images = [standardize(Image.open(p).convert("RGB"), args.size) for p in source_paths]
    logger.info(f"Standardized {len(images)} images to {args.size}×{args.size}")

    quadrants = [slice_quadrants(img) for img in images]
    logger.info(f"Sliced into {len(quadrants) * 9} quadrants ({quad_size}×{quad_size} each)")

    latin = make_latin_square(9, rng)

    output_paths = []
    for out_i in range(9):
        pieces = [
            (src_j, latin[out_i][src_j], quadrants[src_j][latin[out_i][src_j]])
            for src_j in range(9)
        ]
        logger.info(f"Assembling output {out_i + 1}/9...")
        result = assemble_output(pieces, rng, quad_size, args.overlap)
        dest = out_dir / f"frankenstein_output_{out_i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="collage-frankenstein", threaded=False)
        logger.info(f"Posted {len(output_paths)} outputs to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
