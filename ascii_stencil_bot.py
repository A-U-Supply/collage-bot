"""Collage stencil ASCII bot.

Fetches 3 images from Slack and converts each to a character art rendering
used as a binary stencil mask. Generates 6 composites plus 3 stencil masks
(9 images total). Posts to img-junkyard.

For each image cell:
  1. Tonal bracket: cells mean brightness is used to filter the character set
     to only candidates whose visual fill is in the same brightness range.
  2. Template matching: within that bracket, the character whose rendered
     pixel pattern best matches the image cell (lowest MSE) is selected.
  3. Inversion: both white-on-black and black-on-white versions of each
     character are candidates. The rendering switches per cell based on
     which version wins the MSE match.

Character set is built from CJK ideographs (sampled across U+4E00–U+9FFF),
CJK punctuation, hiragana, katakana, box drawing, block elements, geometric
shapes, math operators, and fullwidth symbols — filtered to those that
actually render with the loaded font.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Unicode ranges to include in character set
_CHAR_RANGES = [
    (0x3000, 0x303F),  # CJK Punctuation
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x2500, 0x257F),  # Box Drawing
    (0x25A0, 0x25FF),  # Geometric Shapes
    (0x2200, 0x22FF),  # Mathematical Operators
    (0x2600, 0x26FF),  # Miscellaneous Symbols
    (0xFF01, 0xFF5E),  # Fullwidth Latin / Symbols
]
_CJK_SAMPLE_STEP = 10    # sample every Nth ideograph from U+4E00–U+9FFF
_TONAL_TOLERANCE = 64    # brightness bracket half-width (out of 255)
_ATLAS_CHUNK = 128       # chars per MSE broadcast chunk (memory control)

_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # Ubuntu CI
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/System/Library/Fonts/PingFang.ttc",                        # macOS
    "/System/Library/Fonts/STHeiti Light.ttc",
]


def _load_font(font_size: int):
    for path in _FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size=font_size)
    logger.warning("No CJK font found, using PIL default")
    return ImageFont.load_default()


def _build_glyph_atlas(font, cell_w: int, cell_h: int):
    """Render all candidate characters and return those with visible pixels.

    Returns:
        atlas:      float32 array (N, cell_h, cell_w), white char on black
        char_list:  list of N characters
        fill_means: float32 array (N,) — mean pixel value per glyph (0–255)
    """
    candidates = []
    for start, end in _CHAR_RANGES:
        candidates.extend(chr(cp) for cp in range(start, end + 1))
    for cp in range(0x4E00, 0x9FFF, _CJK_SAMPLE_STEP):
        candidates.append(chr(cp))
    candidates.append(" ")

    probe = ImageDraw.Draw(Image.new("L", (cell_w * 2, cell_h * 2)))
    patches, char_list, fill_means = [], [], []

    for ch in candidates:
        patch = Image.new("L", (cell_w, cell_h), 0)
        d = ImageDraw.Draw(patch)
        try:
            bbox = probe.textbbox((0, 0), ch, font=font)
            x = (cell_w - (bbox[2] - bbox[0])) // 2 - bbox[0]
            y = (cell_h - (bbox[3] - bbox[1])) // 2 - bbox[1]
        except Exception:
            x, y = 0, 0
        d.text((x, y), ch, fill=255, font=font)
        arr = np.array(patch, dtype=np.float32)
        mean = arr.mean()
        # Skip glyphs the font can't render (tofu = near-zero pixels), keep space
        if mean < 1.0 and ch != " ":
            continue
        patches.append(arr)
        char_list.append(ch)
        fill_means.append(mean)

    atlas = np.stack(patches, axis=0)
    return atlas, char_list, np.array(fill_means, dtype=np.float32)


def make_ascii_stencil(img: Image.Image, char_height: float = 0.02) -> Image.Image:
    """Convert image to character art using tonal bracketing + template matching.

    For each cell, the character (and inversion) whose rendered pixel pattern
    best matches the image cell within its tonal bracket is selected.
    Inverted cells (black char on white background) are candidates alongside
    normal (white char on black), allowing the rendering to switch per cell.

    char_height: fraction of image height per character row (e.g. 0.03 = 3%)
    Returns an RGB image resized to original input dimensions.
    """
    font_size = max(4, round(img.height * char_height))
    font = _load_font(font_size)

    # Cell size: CJK characters are square, so cell_w ≈ font_size
    probe_draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    bbox = probe_draw.textbbox((0, 0), "一", font=font)
    cell_w = max(1, bbox[2])
    cell_h = max(1, font_size)

    w, h = img.size
    cols = max(1, w // cell_w)
    rows = max(1, h // cell_h)

    logger.info(f"Building glyph atlas (font_size={font_size}px)...")
    atlas, char_list, fill_means = _build_glyph_atlas(font, cell_w, cell_h)
    N = len(char_list)
    logger.info(f"Atlas: {N} valid characters")
    logger.info(f"Grid: {cols}×{rows} cells at {cell_w}×{cell_h}px")

    # Extend atlas with inverted versions (black char on white background)
    # Indices 0..N-1 = normal, N..2N-1 = inverted
    atlas_inv = 255.0 - atlas
    fill_means_inv = 255.0 - fill_means
    atlas_all = np.concatenate([atlas, atlas_inv], axis=0)       # (2N, H, W)
    fill_means_all = np.concatenate([fill_means, fill_means_inv]) # (2N,)

    # Grayscale source as float32, trimmed to exact grid
    gray = np.array(img.convert("L"), dtype=np.float32)
    gray_trimmed = gray[:rows * cell_h, :cols * cell_w]
    # Reshape to (rows, cols, cell_h, cell_w)
    cells_all = gray_trimmed.reshape(rows, cell_h, cols, cell_w).transpose(0, 2, 1, 3)

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    two_n = 2 * N

    for r in range(rows):
        row_cells = cells_all[r]                          # (cols, H, W)
        cell_means = row_cells.mean(axis=(1, 2))          # (cols,)

        best_mse = np.full(cols, np.inf)
        best_global_idx = np.zeros(cols, dtype=int)

        # Process atlas in chunks to keep memory bounded
        for chunk_start in range(0, two_n, _ATLAS_CHUNK):
            chunk_end = min(chunk_start + _ATLAS_CHUNK, two_n)
            chunk_atlas = atlas_all[chunk_start:chunk_end]    # (C, H, W)
            chunk_fills = fill_means_all[chunk_start:chunk_end]  # (C,)

            # Tonal mask: only consider chars whose fill is close to cell mean
            tonal_dist = np.abs(
                cell_means[:, np.newaxis] - chunk_fills[np.newaxis, :]
            )  # (cols, C)
            in_bracket = tonal_dist < _TONAL_TOLERANCE

            # MSE for all (col, chunk_char) pairs
            diffs = (
                (row_cells[:, np.newaxis, :, :] - chunk_atlas[np.newaxis, :, :, :]) ** 2
            ).mean(axis=(2, 3))  # (cols, C)

            diffs[~in_bracket] = np.inf

            chunk_min_mse = diffs.min(axis=1)               # (cols,)
            chunk_min_local = np.argmin(diffs, axis=1)      # (cols,)

            improve = chunk_min_mse < best_mse
            best_mse[improve] = chunk_min_mse[improve]
            best_global_idx[improve] = chunk_start + chunk_min_local[improve]

        # Fallback: if tonal bracket matched nothing, use global best
        no_match = np.isinf(best_mse)
        if no_match.any():
            for chunk_start in range(0, two_n, _ATLAS_CHUNK):
                chunk_end = min(chunk_start + _ATLAS_CHUNK, two_n)
                chunk_atlas = atlas_all[chunk_start:chunk_end]
                diffs = (
                    (row_cells[no_match][:, np.newaxis, :, :] -
                     chunk_atlas[np.newaxis, :, :, :]) ** 2
                ).mean(axis=(2, 3))
                chunk_min = diffs.min(axis=1)
                chunk_idx = np.argmin(diffs, axis=1)
                cols_idx = np.where(no_match)[0]
                improve = chunk_min < best_mse[cols_idx]
                upd = cols_idx[improve]
                best_mse[upd] = chunk_min[improve]
                best_global_idx[upd] = chunk_start + chunk_idx[improve]

        # Render each cell
        for c, idx in enumerate(best_global_idx):
            ch = char_list[idx % N]
            inverted = idx >= N

            if inverted:
                # White background, black character
                draw.rectangle(
                    [c * cell_w, r * cell_h, (c + 1) * cell_w - 1, (r + 1) * cell_h - 1],
                    fill=(255, 255, 255),
                )
                if ch != " ":
                    draw.text((c * cell_w, r * cell_h), ch,
                              fill=(0, 0, 0), font=font)
            elif ch != " ":
                # Black background (default), white character
                draw.text((c * cell_w, r * cell_h), ch,
                          fill=(255, 255, 255), font=font)

    return canvas.resize(img.size, Image.LANCZOS)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil ASCII bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./ascii-stencil-bot-output"))
    parser.add_argument("--char-height", type=float, default=0.02,
                        help="Character height as fraction of image height (e.g. 0.03 = 3%%)")
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

    masks = []
    mask_paths = []
    for i, img in enumerate(images):
        logger.info(f"Generating stencil {i + 1}/3 (char_height={args.char_height})...")
        ascii_img = make_ascii_stencil(img, char_height=args.char_height)
        mask = make_stencil(ascii_img)
        mask_rgb = mask.convert("RGB")
        dest = out_dir / f"ascii_mask_{i + 1}.png"
        mask_rgb.save(dest)
        logger.info(f"Saved {dest.name}")
        masks.append(mask)
        mask_paths.append(dest)

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2),
                                    (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: stencil={s + 1}, fill_a={a + 1}, fill_b={b + 1}")
        result = apply_stencil(masks[s], images[a], images[b])
        dest = out_dir / f"ascii_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    post_paths = output_paths + mask_paths

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths,
                      bot_name="collage-stencil-ascii-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
