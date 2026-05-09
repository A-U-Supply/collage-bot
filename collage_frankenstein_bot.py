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


def slice_quadrants(img: Image.Image, pad: int = 0) -> list[Image.Image]:
    """Cut a square image into a 3×3 grid. Returns 9 PIL Images in row-major order.

    pad > 0: each returned tile is (q + 2*pad) × (q + 2*pad) — the core q×q
    quadrant surrounded by `pad` pixels of source-image context on every side,
    using edge replication at image boundaries.  The overhang pixels are not
    rendered on the output canvas; they exist solely to inform the warp shift
    computation so it can see where each quadrant's forms were heading.
    """
    w, h = img.size
    assert w == h, "image must be square before slicing"
    q = w // 3
    if pad == 0:
        return [
            img.crop((c * q, r * q, (c + 1) * q, (r + 1) * q))
            for r in range(3)
            for c in range(3)
        ]
    arr = np.pad(np.array(img), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    return [
        Image.fromarray(arr[r * q : r * q + q + 2 * pad,
                            c * q : c * q + q + 2 * pad])
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
    """Recomposite the bidirectional vertical overlap zone between horizontally adjacent tiles.

    The seam zone spans [x_boundary - overlap, x_boundary + overlap), giving the
    seam path freedom to cut into EITHER tile's territory.  For the right half of
    the zone (where the left tile has no real pixels) the left tile's edge column
    is repeated as a virtual extension so the DP error stays well-defined.

    canvas_y: top canvas row of this tile row.
    x_boundary: canvas x where the left tile nominally ends.
    """
    H = left_arr.shape[0]
    zone_width = 2 * overlap
    zone_x = x_boundary - overlap

    # Left tile: real pixels for cols [0, overlap), edge column repeated for [overlap, 2*overlap)
    left_real = left_arr[:, -overlap:].astype(np.float32)                        # (H, overlap, 3)
    left_ext  = np.repeat(left_arr[:, -1:].astype(np.float32), overlap, axis=1)  # (H, overlap, 3)
    left_zone = np.concatenate([left_real, left_ext], axis=1)                    # (H, 2*overlap, 3)

    # Right tile: real pixels covering the full zone
    right_zone = right_arr[:, :zone_width].astype(np.float32)                    # (H, 2*overlap, 3)

    error = np.mean((left_zone - right_zone) ** 2, axis=2)                       # (H, 2*overlap)
    seam  = _min_cost_seam(error)                                                 # (H,) in [0, 2*overlap)

    col_idx  = np.arange(zone_width)[np.newaxis, :]   # (1, 2*overlap)
    seam_col = seam[:, np.newaxis]                    # (H, 1)
    alpha = (col_idx >= seam_col).astype(np.float32)[:, :, np.newaxis]  # (H, 2*overlap, 1)

    blended = ((1.0 - alpha) * left_zone + alpha * right_zone).astype(np.uint8)
    canvas[canvas_y : canvas_y + H, zone_x : zone_x + zone_width] = blended


def _quilt_horizontal(
    canvas: np.ndarray,
    top_arr: np.ndarray,
    bottom_arr: np.ndarray,
    canvas_x: int,
    y_boundary: int,
    overlap: int,
) -> None:
    """Recomposite the bidirectional horizontal overlap zone between vertically adjacent tiles.

    The seam zone spans [y_boundary - overlap, y_boundary + overlap), giving the
    seam path freedom to cut into EITHER tile's territory.  For the bottom half of
    the zone (where the top tile has no real pixels) the top tile's bottom row is
    repeated as a virtual extension.

    canvas_x: left canvas column of this tile column.
    y_boundary: canvas y where the top tile nominally ends.
    """
    W = top_arr.shape[1]
    zone_height = 2 * overlap
    zone_y = y_boundary - overlap

    # Top tile: real pixels for rows [0, overlap), bottom row repeated for [overlap, 2*overlap)
    top_real = top_arr[-overlap:, :].astype(np.float32)                          # (overlap, W, 3)
    top_ext  = np.repeat(top_arr[-1:, :].astype(np.float32), overlap, axis=0)   # (overlap, W, 3)
    top_zone = np.concatenate([top_real, top_ext], axis=0)                       # (2*overlap, W, 3)

    # Bottom tile: real pixels covering the full zone
    bottom_zone = bottom_arr[:zone_height, :].astype(np.float32)                 # (2*overlap, W, 3)

    # Transpose so the DP seam gives a row-cut value per column.
    error = np.mean((top_zone - bottom_zone) ** 2, axis=2).T                    # (W, 2*overlap)
    seam  = _min_cost_seam(error)                                                # (W,) in [0, 2*overlap)

    row_idx  = np.arange(zone_height)[:, np.newaxis]  # (2*overlap, 1)
    seam_row = seam[np.newaxis, :]                    # (1, W)
    alpha = (row_idx >= seam_row).astype(np.float32)[:, :, np.newaxis]  # (2*overlap, W, 1)

    blended = ((1.0 - alpha) * top_zone + alpha * bottom_zone).astype(np.uint8)
    canvas[zone_y : zone_y + zone_height, canvas_x : canvas_x + W] = blended


# ---------------------------------------------------------------------------
# Geometric warp — per-row/column displacement field
# ---------------------------------------------------------------------------

def _gaussian_smooth_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Convolve a 1-D float array with a Gaussian kernel (pure numpy)."""
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr.astype(np.float64), kernel, mode="same")


def _bilinear_remap(src: np.ndarray, map_r: np.ndarray, map_c: np.ndarray) -> np.ndarray:
    """Bilinear interpolation remap (pure numpy).

    src:   (H, W, 3) uint8
    map_r: (H', W') float — row coordinates in src space
    map_c: (H', W') float — column coordinates in src space
    Returns (H', W', 3) uint8.
    """
    H, W = src.shape[:2]
    # Clamp coordinates first so fractional parts stay in [0, 1).
    map_r = np.clip(map_r, 0, H - 1)
    map_c = np.clip(map_c, 0, W - 1)
    r0 = np.floor(map_r).astype(int).clip(0, H - 2)
    c0 = np.floor(map_c).astype(int).clip(0, W - 2)
    r1 = r0 + 1
    c1 = c0 + 1
    fr = (map_r - r0)[..., np.newaxis].astype(np.float32)
    fc = (map_c - c0)[..., np.newaxis].astype(np.float32)
    return (
        (1 - fr) * (1 - fc) * src[r0, c0]
        + (1 - fr) * fc      * src[r0, c1]
        + fr       * (1 - fc) * src[r1, c0]
        + fr       * fc      * src[r1, c1]
    ).clip(0, 255).astype(np.uint8)


def _grad_mag(strip: np.ndarray) -> np.ndarray:
    """Gradient magnitude of an (H, W, 3) strip → (H, W) float32.

    Uses central differences on the luminance channel so that structural
    edges and forms are captured independent of colour.
    """
    gray = strip.mean(axis=2).astype(np.float32)        # (H, W)
    dy = np.gradient(gray, axis=0)
    dx = np.gradient(gray, axis=1)
    return np.sqrt(dy ** 2 + dx ** 2)


def _ncc_rows(row_a: np.ndarray, row_b: np.ndarray) -> float:
    """Normalized cross-correlation between two 1-D gradient-magnitude rows.

    Returns a value in [-1, 1]; higher = better structural alignment.
    Falls back to 0 when either row has no variance (flat region).
    """
    a = row_a - row_a.mean()
    b = row_b - row_b.mean()
    denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    if denom < 1e-6:
        return 0.0
    return float((a * b).sum() / denom)


def _compute_shifts(
    strip_a: np.ndarray,
    strip_b: np.ndarray,
    radius: int,
    scale: float = 1.0,
) -> np.ndarray:
    """Per-row vertical shifts that best align the forms in strip_a to strip_b.

    Computes gradient magnitude of each strip then, for every row r, finds
    the integer shift s in [-radius, +radius] that maximises the normalised
    cross-correlation between gradient row r of strip_a and gradient row r+s
    of strip_b.  NCC on gradients matches structural edges/forms rather than
    raw colour, so the resulting shifts align visual content across the tile
    boundary.

    strip_a, strip_b: (H, W, 3) float arrays — border strips from each tile.
    Returns smoothed float shifts multiplied by scale.
    """
    H = strip_a.shape[0]
    grad_a = _grad_mag(strip_a)  # (H, W)
    grad_b = _grad_mag(strip_b)  # (H, W)

    raw = np.zeros(H, dtype=np.float64)
    for r in range(H):
        best_s, best_score = 0, -2.0
        row_a = grad_a[r]
        for s in range(-radius, radius + 1):
            r2 = max(0, min(H - 1, r + s))
            score = _ncc_rows(row_a, grad_b[r2])
            if score > best_score:
                best_s, best_score = s, score
        raw[r] = best_s
    return _gaussian_smooth_1d(raw, sigma=8.0) * scale


def _warp_vertical(
    canvas: np.ndarray,
    left_arr: np.ndarray,
    right_arr: np.ndarray,
    canvas_y: int,
    x_boundary: int,
    overlap: int,
    warp_depth: int,
    warp_radius: int,
    warp_scale: float = 1.0,
    warp_strip: int = 20,
    warp_pad: int = 0,
) -> None:
    """Geometric warp at a left-right tile boundary.

    Both tiles bend toward each other by half the computed shift.

    When warp_pad > 0, left_arr / right_arr are padded (q+2*pad)×(q+2*pad)
    arrays.  Shift computation uses the overhang strips — what each source
    image looks like PAST the quadrant's cut edge — so the displacement field
    aligns the forms each tile was heading toward rather than just the pixels
    at the hard boundary.  The displacement is applied to the core q×q region.

    When warp_pad == 0, falls back to gradient-NCC on the tile's own boundary
    strip (warp_strip columns wide).
    """
    pad = warp_pad
    if pad > 0:
        q = left_arr.shape[0] - 2 * pad
        left_core  = left_arr [pad : pad + q, pad : pad + q]
        right_core = right_arr[pad : pad + q, pad : pad + q]
        eff = max(1, min(pad, warp_strip))
        # Left tile's right overhang: source image continuation past A's right edge
        left_overhang  = left_arr [pad : pad + q, pad + q : pad + q + eff]
        # Right tile's left overhang: source image that preceded B's left edge
        right_overhang = right_arr[pad : pad + q, pad - eff : pad          ]
        shifts = _compute_shifts(
            left_overhang.astype(np.float32),
            right_overhang.astype(np.float32),
            warp_radius, scale=warp_scale,
        )
    else:
        q = left_arr.shape[0]
        left_core  = left_arr
        right_core = right_arr
        strip_w = max(1, min(warp_strip, q))
        shifts = _compute_shifts(
            left_core[:, -strip_w:].astype(np.float32),
            right_core[:, :strip_w].astype(np.float32),
            warp_radius, scale=warp_scale,
        )

    H = q  # canvas height of this tile row
    half = shifts / 2.0

    # ---- Warp left tile's right zone ----
    # Displacement is applied to the canvas (already quilted if quilt ran first).
    # .copy() so the read and write don't alias.
    depth_l = min(warp_depth, q)
    c_idx = np.arange(depth_l, dtype=np.float32)
    fade_l = c_idx / max(depth_l - 1, 1)
    dr_l = half[:, np.newaxis] * fade_l[np.newaxis, :]          # (H, depth_l)

    left_zone_x0 = x_boundary - depth_l
    src_l = canvas[canvas_y : canvas_y + H, left_zone_x0 : x_boundary].copy()
    map_r_l = np.arange(H, dtype=np.float32)[:, np.newaxis] + dr_l
    map_c_l = c_idx[np.newaxis, :] * np.ones((H, 1))            # local cols 0..depth_l-1
    canvas[canvas_y : canvas_y + H,
           left_zone_x0 : x_boundary] = _bilinear_remap(src_l, map_r_l, map_c_l.astype(np.float32))

    # ---- Warp right tile's left zone ----
    depth_r = min(warp_depth, q)
    c_idx_r = np.arange(depth_r, dtype=np.float32)
    fade_r = (depth_r - 1 - c_idx_r) / max(depth_r - 1, 1)
    dr_r = (-half[:, np.newaxis]) * fade_r[np.newaxis, :]        # (H, depth_r)

    right_canvas_x = x_boundary - overlap
    src_r = canvas[canvas_y : canvas_y + H, right_canvas_x : right_canvas_x + depth_r].copy()
    map_r_r = np.arange(H, dtype=np.float32)[:, np.newaxis] + dr_r
    map_c_r = c_idx_r[np.newaxis, :] * np.ones((H, 1))
    canvas[canvas_y : canvas_y + H,
           right_canvas_x : right_canvas_x + depth_r] = _bilinear_remap(src_r, map_r_r, map_c_r.astype(np.float32))


def _warp_horizontal(
    canvas: np.ndarray,
    top_arr: np.ndarray,
    bottom_arr: np.ndarray,
    canvas_x: int,
    y_boundary: int,
    overlap: int,
    warp_depth: int,
    warp_radius: int,
    warp_scale: float = 1.0,
    warp_strip: int = 20,
    warp_pad: int = 0,
) -> None:
    """Geometric warp at a top-bottom tile boundary.

    Transpose of _warp_vertical: shifts are per-column horizontal offsets.
    When warp_pad > 0, uses overhang rows (source image continuation past
    each tile's cut edge) for shift computation.
    """
    pad = warp_pad
    if pad > 0:
        q = top_arr.shape[1] - 2 * pad
        top_core    = top_arr   [pad : pad + q, pad : pad + q]
        bottom_core = bottom_arr[pad : pad + q, pad : pad + q]
        eff = max(1, min(pad, warp_strip))
        # Top tile's bottom overhang: source image continuation below T's bottom edge
        top_overhang    = top_arr   [pad + q : pad + q + eff, pad : pad + q]
        # Bottom tile's top overhang: source image that preceded B's top edge
        bottom_overhang = bottom_arr[pad - eff : pad,          pad : pad + q]
        # Transpose so _compute_shifts operates per-column
        shifts = _compute_shifts(
            top_overhang.astype(np.float32).transpose(1, 0, 2),
            bottom_overhang.astype(np.float32).transpose(1, 0, 2),
            warp_radius, scale=warp_scale,
        )
    else:
        q = top_arr.shape[0]
        top_core    = top_arr
        bottom_core = bottom_arr
        strip_h = max(1, min(warp_strip, q))
        shifts = _compute_shifts(
            top_core[-strip_h:, :].astype(np.float32).transpose(1, 0, 2),
            bottom_core[:strip_h, :].astype(np.float32).transpose(1, 0, 2),
            warp_radius, scale=warp_scale,
        )

    W = q
    half = shifts / 2.0

    # ---- Warp top tile's bottom zone ----
    depth_t = min(warp_depth, q)
    r_idx = np.arange(depth_t, dtype=np.float32)
    fade_t = r_idx / max(depth_t - 1, 1)
    dc_t = half[np.newaxis, :] * fade_t[:, np.newaxis]       # (depth_t, W)

    top_zone_y0 = y_boundary - depth_t
    src_t = canvas[top_zone_y0 : y_boundary, canvas_x : canvas_x + W].copy()
    map_r_t = r_idx[:, np.newaxis] * np.ones((1, W))          # local rows 0..depth_t-1
    map_c_t = np.arange(W, dtype=np.float32)[np.newaxis, :] + dc_t
    canvas[top_zone_y0 : y_boundary,
           canvas_x : canvas_x + W] = _bilinear_remap(src_t, map_r_t.astype(np.float32), map_c_t.astype(np.float32))

    # ---- Warp bottom tile's top zone ----
    depth_b = min(warp_depth, q)
    r_idx_b = np.arange(depth_b, dtype=np.float32)
    fade_b = (depth_b - 1 - r_idx_b) / max(depth_b - 1, 1)
    dc_b = (-half[np.newaxis, :]) * fade_b[:, np.newaxis]    # (depth_b, W)

    bottom_canvas_y = y_boundary - overlap
    src_b = canvas[bottom_canvas_y : bottom_canvas_y + depth_b, canvas_x : canvas_x + W].copy()
    map_r_b = r_idx_b[:, np.newaxis] * np.ones((1, W))
    map_c_b = np.arange(W, dtype=np.float32)[np.newaxis, :] + dc_b
    canvas[bottom_canvas_y : bottom_canvas_y + depth_b,
           canvas_x : canvas_x + W] = _bilinear_remap(src_b, map_r_b.astype(np.float32), map_c_b.astype(np.float32))


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
    quilt: bool = True,
    warp: bool = False,
    warp_depth: int = 80,
    warp_radius: int = 30,
    warp_scale: float = 1.0,
    warp_strip: int = 20,
    warp_pad: int = 0,
) -> Image.Image:
    """Greedily assemble one 3×3 output, then apply boundary effects.

    Phase 1 — greedy placement: raster-order selection of the best (piece,
    transform) pair for each slot, scored by weighted edge continuity.

    Phase 2a (if quilt=True) — image quilting: minimum-cost seam cutting
    repaints each bidirectional overlap zone so the boundary follows natural
    image edges rather than a hard grid line.

    Phase 2b (if warp=True) — geometric warp: a per-row/column displacement
    field bends each tile's content toward its neighbor at the boundary.
    Both tiles warp toward each other by half the computed shift.
    """
    step = quad_size - overlap                      # pixels between tile origins
    canvas_size = quad_size + 2 * step              # = 3*quad_size - 2*overlap

    canvas = Image.new("RGB", (canvas_size, canvas_size))
    placed: list[np.ndarray | None] = [None] * 9              # core arrays, for edge scoring
    grid: list[list[np.ndarray | None]] = [[None] * 3 for _ in range(3)]         # cores, for quilt
    grid_padded: list[list[np.ndarray | None]] = [[None] * 3 for _ in range(3)]  # padded, for warp
    remaining = list(range(len(pieces)))
    rng.shuffle(remaining)

    pad = warp_pad  # overhang on each side (0 = no overhang)

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
        best_padded = None
        best_img = None

        for piece_list_idx in remaining:
            tile = pieces[piece_list_idx][2]
            for transpose, invert in TRANSFORMS:
                candidate = apply_transform(tile, transpose, invert)
                full = np.array(candidate)
                # Extract core for scoring and canvas placement
                core = full[pad : pad + quad_size, pad : pad + quad_size] if pad > 0 else full
                score = edge_score(core, above_arr, left_arr)
                if score < best_score:
                    best_score = score
                    best_piece_idx = piece_list_idx
                    best_arr = core
                    best_padded = full
                    best_img = Image.fromarray(core)

        remaining.remove(best_piece_idx)
        placed[slot] = best_arr
        grid[row][col] = best_arr
        grid_padded[row][col] = best_padded
        canvas.paste(best_img, (col * step, row * step))

    canvas_arr = np.array(canvas)

    # ------------------------------------------------------------------
    # Phase 2a: image quilting — recomposite overlap zones with seam cuts
    # ------------------------------------------------------------------
    if quilt:
        for tile_row in range(3):
            canvas_y = tile_row * step
            for col_seam in range(1, 3):
                x_boundary = (col_seam - 1) * step + quad_size
                _quilt_vertical(
                    canvas_arr,
                    grid[tile_row][col_seam - 1],
                    grid[tile_row][col_seam],
                    canvas_y=canvas_y,
                    x_boundary=x_boundary,
                    overlap=overlap,
                )
        for row_seam in range(1, 3):
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

    # ------------------------------------------------------------------
    # Phase 2b: geometric warp — bend tile content toward neighbour edges
    # ------------------------------------------------------------------
    if warp:
        for tile_row in range(3):
            canvas_y = tile_row * step
            for col_seam in range(1, 3):
                x_boundary = (col_seam - 1) * step + quad_size
                _warp_vertical(
                    canvas_arr,
                    grid_padded[tile_row][col_seam - 1],
                    grid_padded[tile_row][col_seam],
                    canvas_y=canvas_y,
                    x_boundary=x_boundary,
                    overlap=overlap,
                    warp_depth=warp_depth,
                    warp_radius=warp_radius,
                    warp_scale=warp_scale,
                    warp_strip=warp_strip,
                    warp_pad=warp_pad,
                )
        for row_seam in range(1, 3):
            y_boundary = (row_seam - 1) * step + quad_size
            for tile_col in range(3):
                canvas_x = tile_col * step
                _warp_horizontal(
                    canvas_arr,
                    grid_padded[row_seam - 1][tile_col],
                    grid_padded[row_seam][tile_col],
                    canvas_x=canvas_x,
                    y_boundary=y_boundary,
                    overlap=overlap,
                    warp_depth=warp_depth,
                    warp_radius=warp_radius,
                    warp_scale=warp_scale,
                    warp_strip=warp_strip,
                    warp_pad=warp_pad,
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
    parser.add_argument("--no-quilt", action="store_true",
                        help="Skip image-quilting seam cuts (quilting is on by default)")
    parser.add_argument("--warp", action="store_true",
                        help="Apply geometric warp at tile boundaries (off by default)")
    parser.add_argument("--warp-depth", type=int, default=80,
                        help="Pixels from boundary to warp inward (default: 80)")
    parser.add_argument("--warp-radius", type=int, default=30,
                        help="Per-row shift search radius in pixels (default: 30)")
    parser.add_argument("--warp-scale", type=float, default=1.0,
                        help="Multiply computed shifts by this factor to amplify warp (default: 1.0)")
    parser.add_argument("--warp-strip", type=int, default=20,
                        help="Width of border strip (px) used to match forms for shift computation (default: 20)")
    parser.add_argument("--warp-pad", type=int, default=20,
                        help="Overhang pixels of source context retained on each quadrant edge "
                             "to inform warp shift computation (default: 20; 0 = use tile boundary strips)")
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

    quadrants = [slice_quadrants(img, pad=args.warp_pad) for img in images]
    logger.info(f"Sliced into {len(quadrants) * 9} quadrants ({quad_size}×{quad_size} each)")

    latin = make_latin_square(9, rng)

    output_paths = []
    for out_i in range(9):
        pieces = [
            (src_j, latin[out_i][src_j], quadrants[src_j][latin[out_i][src_j]])
            for src_j in range(9)
        ]
        logger.info(f"Assembling output {out_i + 1}/9...")
        result = assemble_output(
            pieces, rng, quad_size, args.overlap,
            quilt=not args.no_quilt,
            warp=args.warp,
            warp_depth=args.warp_depth,
            warp_radius=args.warp_radius,
            warp_scale=args.warp_scale,
            warp_strip=args.warp_strip,
            warp_pad=args.warp_pad,
        )
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
