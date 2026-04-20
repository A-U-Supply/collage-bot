"""Collage stencil ASCII bot.

Fetches 3 images from Slack and converts each to a character art rendering
(white characters on black background) used as a binary stencil mask.
Generates 6 composites (all permutations of stencil/fill pair) plus the
3 stencil masks — 9 images total. Posts to img-junkyard.

For each image cell, selects the character whose rendered pixel pattern
best matches the image content (lowest MSE). This makes character choice
contour-aware rather than purely tonal — edges and curves in the image
influence which character appears, not just brightness.

char_height controls grid detail: smaller fraction = finer grid, larger = blockier look.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Mixed character set for template matching.
# CJK ordered by stroke density + box drawing (directional) +
# block elements (tonal) + geometric shapes + space (empty cell).
_CHARS = (
    "鬱藏疆赢德意道常高重明来目日木人一"  # CJK dense → sparse
    "═║╔╗╚╝╠╣╦╩╬─│┌┐└┘├┤┬┴┼"          # box drawing (directional)
    "█▓▒░▀▄▌▐▖▗▘▝"                     # block elements (tonal)
    "●◆■▲◉◎○◇□△"                       # geometric shapes
    " "                                 # empty cell
)

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


def _build_glyph_atlas(chars: str, font, cell_w: int, cell_h: int):
    """Pre-render each character as a grayscale patch of size (cell_h, cell_w).

    Centers each glyph within the cell using its measured bounding box.

    Returns:
        atlas:     float32 numpy array, shape (N, cell_h, cell_w)
        char_list: list of N characters in matching order
    """
    probe = ImageDraw.Draw(Image.new("L", (cell_w * 2, cell_h * 2)))
    patches = []
    char_list = []
    for ch in chars:
        patch = Image.new("L", (cell_w, cell_h), 0)
        d = ImageDraw.Draw(patch)
        try:
            bbox = probe.textbbox((0, 0), ch, font=font)
            x = (cell_w - (bbox[2] - bbox[0])) // 2 - bbox[0]
            y = (cell_h - (bbox[3] - bbox[1])) // 2 - bbox[1]
        except Exception:
            x, y = 0, 0
        d.text((x, y), ch, fill=255, font=font)
        patches.append(np.array(patch, dtype=np.float32))
        char_list.append(ch)
    return np.stack(patches, axis=0), char_list


def make_ascii_stencil(img: Image.Image, char_height: float = 0.02) -> Image.Image:
    """Convert image to character art using template matching.

    For each grid cell, selects the character whose rendered pixel pattern
    has the lowest MSE against the image cell. This is contour-aware:
    horizontal edges attract box-drawing chars, dense regions attract
    complex CJK characters, etc.

    char_height: fraction of image height per character row (e.g. 0.03 = 3%)
    Returns an RGB image resized to original input dimensions.
    """
    font_size = max(4, round(img.height * char_height))
    font = _load_font(font_size)

    # Measure cell size from a representative CJK character
    probe_draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    bbox = probe_draw.textbbox((0, 0), "一", font=font)
    cell_w = max(1, bbox[2])
    cell_h = max(1, font_size)

    w, h = img.size
    cols = max(1, w // cell_w)
    rows = max(1, h // cell_h)

    logger.info(
        f"Grid: {cols}×{rows} cells at {cell_w}×{cell_h}px, "
        f"font_size={font_size}px, {len(_CHARS)} chars in atlas"
    )

    # Build glyph atlas once per call
    atlas, char_list = _build_glyph_atlas(_CHARS, font, cell_w, cell_h)
    # atlas shape: (N, cell_h, cell_w)

    # Grayscale image as float32
    gray = np.array(img.convert("L"), dtype=np.float32)
    gray_trimmed = gray[:rows * cell_h, :cols * cell_w]
    # Reshape to (rows, cols, cell_h, cell_w) for row-wise matching
    cells_all = gray_trimmed.reshape(rows, cell_h, cols, cell_w).transpose(0, 2, 1, 3)

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # Template matching row by row (memory-efficient)
    for r in range(rows):
        row_cells = cells_all[r]  # (cols, cell_h, cell_w)
        # Broadcast: (cols, N, cell_h, cell_w) → MSE → (cols, N)
        diffs = (
            (row_cells[:, np.newaxis, :, :] - atlas[np.newaxis, :, :, :]) ** 2
        ).mean(axis=(2, 3))
        best_indices = np.argmin(diffs, axis=1)  # (cols,)
        for c, idx in enumerate(best_indices):
            ch = char_list[idx]
            if ch != " ":
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

    # Build stencil masks for all 3 source images
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

    # Generate 6 permutation composites
    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2),
                                    (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: stencil={s + 1}, fill_a={a + 1}, fill_b={b + 1}")
        result = apply_stencil(masks[s], images[a], images[b])
        dest = out_dir / f"ascii_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    # 6 composites + 3 masks = 9 total
    post_paths = output_paths + mask_paths

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths,
                      bot_name="collage-stencil-ascii-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
