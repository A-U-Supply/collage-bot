"""Collage stencil ASCII bot.

Fetches 3 images from Slack and converts each to an ASCII art rendering
(white characters on black background) used as a binary stencil mask.
Generates 6 composites (all permutations of stencil/fill pair) plus the
3 ASCII stencil masks — 9 images total. Posts to img-junkyard.

Dark source pixels → dense characters (white text), bright pixels → spaces (black).
char_height controls detail level: smaller fraction = finer grid, larger = blockier look.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Character ramp ordered by visual ink density (complex → simple → space).
# Dark source pixels → complex/dense CJK character; bright pixels → space.
_RAMP = "鬱藏疆赢德意道常高重明来目日木人一 "

_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # Ubuntu CI (fonts-noto-cjk)
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/System/Library/Fonts/PingFang.ttc",                        # macOS
    "/System/Library/Fonts/STHeiti Light.ttc",
]


def _load_font(font_size: int):
    for path in _FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size=font_size)
    logger.warning("No monospace font found on disk, using PIL default")
    return ImageFont.load_default()


def make_ascii_stencil(img: Image.Image, char_height: float = 0.02) -> Image.Image:
    """Convert image to ASCII art: white characters on black background.

    Dark source pixels map to dense characters; bright pixels map to spaces.
    char_height is a fraction of the image height (e.g. 0.02 = 2%).
    Returns an RGB image resized to the original input dimensions.
    """
    font_size = max(4, round(img.height * char_height))
    font = _load_font(font_size)

    # Measure cell dimensions from the font
    probe_draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    bbox = probe_draw.textbbox((0, 0), "@", font=font)
    cell_w = max(1, bbox[2])
    cell_h = max(1, font_size)  # use font_size for consistent row height

    w, h = img.size
    cols = max(1, w // cell_w)
    rows = max(1, h // cell_h)

    # Compute per-cell average brightness via numpy reshape
    gray = np.array(img.convert("L"), dtype=np.float32)
    gray_trimmed = gray[:rows * cell_h, :cols * cell_w]
    cells = gray_trimmed.reshape(rows, cell_h, cols, cell_w).mean(axis=(1, 3))

    # Render ASCII onto black canvas
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    ramp_len = len(_RAMP) - 1

    for r in range(rows):
        for c in range(cols):
            idx = round(float(cells[r, c]) / 255.0 * ramp_len)
            char = _RAMP[idx]
            if char != " ":
                draw.text((c * cell_w, r * cell_h), char,
                          fill=(255, 255, 255), font=font)

    logger.info(f"ASCII grid: {cols}×{rows} cells at {cell_w}×{cell_h}px each")

    # Resize back to original image dimensions
    return canvas.resize(img.size, Image.LANCZOS)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil ASCII bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./ascii-stencil-bot-output"))
    parser.add_argument("--char-height", type=float, default=0.02,
                        help="Character height as fraction of image height (e.g. 0.02 = 2%%)")
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

    # Build ASCII stencil masks for all 3 source images
    masks = []
    mask_paths = []
    for i, img in enumerate(images):
        logger.info(f"Generating ASCII stencil {i + 1}/3 (char_height={args.char_height})...")
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
