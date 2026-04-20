"""Collage stencil ASCII/CJK bot.

Fetches 3 images from Slack and converts each to a CJK character art rendering
used as a binary stencil mask. Generates 6 composites plus 3 stencil masks
(9 images total). Posts to img-junkyard.

Algorithm (based on AcerolaFX ASCII shader):
  For each image tile (cell):
  1. Compute Difference of Gaussians edge map + Sobel gradient direction
  2. Vote on dominant edge direction within the tile
  3. EDGE MODE (strong edge present): pick a CJK character from the
     directional bucket matching that edge angle (vertical/horizontal/diagonal)
  4. FILL MODE (no dominant edge): pick a CJK character from a density
     ramp based on mean tile luminance
  5. Both modes: consider white-on-black vs black-on-white, pick lowest MSE

char_height controls grid detail: smaller fraction = finer grid, larger = blockier.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ── Character sets ─────────────────────────────────────────────────────────

# Edge characters: curated CJK by dominant visual stroke direction
_DIR_CHARS = {
    0: "│┃║╎╏┆┇",   # vertical   ( | )
    1: "─━═╌╍┄┅",   # horizontal ( ─ )
    2: "╱",           # diagonal  /
    3: "╲",           # diagonal  \
}

# Fill characters: CJK ordered by visual ink density (dense → sparse → space)
_FILL_CHARS = "鬱藏疆赢德意道常高重明来目日木人一 "

# ── Tunable constants ───────────────────────────────────────────────────────
_SUPERSAMPLE = 4         # atlas and canvas render scale for crisper glyphs

_DOG_SIGMA1 = 2.0
_DOG_SIGMA2 = 3.2          # sigma1 × 1.6
_DOG_THRESHOLD = 1.0       # DoG value to count a pixel as an edge
_EDGE_TILE_THRESHOLD = 6   # min edge pixels in tile to use edge mode

# ── Font paths ──────────────────────────────────────────────────────────────
_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
]


def _load_font(font_size: int):
    for path in _FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size=font_size)
    logger.warning("No CJK font found, using PIL default")
    return ImageFont.load_default()


def _build_bucket_atlas(chars: str, font_size: int, cell_w: int, cell_h: int):
    """Render a small set of characters to a float32 atlas.

    Renders at _SUPERSAMPLE× size then downsamples + binarizes for crisp patches.

    Returns (atlas, char_list):
        atlas:     (N, cell_h, cell_w) float32 — binarized (0 or 255)
        char_list: list of N chars
    """
    big_font = _load_font(font_size * _SUPERSAMPLE)
    bw = cell_w * _SUPERSAMPLE
    bh = cell_h * _SUPERSAMPLE
    probe = ImageDraw.Draw(Image.new("L", (bw * 2, bh * 2)))
    patches, char_list = [], []
    for ch in chars:
        patch = Image.new("L", (bw, bh), 0)
        d = ImageDraw.Draw(patch)
        try:
            bbox = probe.textbbox((0, 0), ch, font=big_font)
            x = (bw - (bbox[2] - bbox[0])) // 2 - bbox[0]
            y = (bh - (bbox[3] - bbox[1])) // 2 - bbox[1]
        except Exception:
            x, y = 0, 0
        d.text((x, y), ch, fill=255, font=big_font)
        small = patch.resize((cell_w, cell_h), Image.LANCZOS)
        arr = np.array(small, dtype=np.float32)
        arr = (arr > 127).astype(np.float32) * 255.0
        patches.append(arr)
        char_list.append(ch)
    return np.stack(patches, axis=0), char_list


def _best_match(cell: np.ndarray, atlas: np.ndarray, char_list: list):
    """Return (char, inverted) with lowest MSE against cell.

    Considers both normal (white-on-black) and inverted (black-on-white).
    """
    atlas_inv = 255.0 - atlas
    # MSE normal and inverted
    mse_n = ((atlas - cell[np.newaxis]) ** 2).mean(axis=(1, 2))
    mse_i = ((atlas_inv - cell[np.newaxis]) ** 2).mean(axis=(1, 2))
    best_n = int(np.argmin(mse_n))
    best_i = int(np.argmin(mse_i))
    if mse_n[best_n] <= mse_i[best_i]:
        return char_list[best_n], False
    else:
        return char_list[best_i], True


def make_ascii_stencil(img: Image.Image, char_height: float = 0.03) -> Image.Image:
    """Convert image to CJK character art using AcerolaFX-style tile algorithm.

    char_height: fraction of image height per character row (e.g. 0.03 = 3%)
    Returns an RGB image resized to original input dimensions.
    """
    font_size = max(4, round(img.height * char_height))
    font = _load_font(font_size)

    probe_draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    bbox = probe_draw.textbbox((0, 0), "一", font=font)
    cell_w = max(1, bbox[2])
    cell_h = max(1, font_size)

    w, h = img.size
    cols = max(1, w // cell_w)
    rows = max(1, h // cell_h)
    logger.info(f"Grid: {cols}×{rows} cells at {cell_w}×{cell_h}px, font_size={font_size}px")

    # Build per-bucket atlases (small, fast)
    dir_atlases = {
        d: _build_bucket_atlas(chars, font_size, cell_w, cell_h)
        for d, chars in _DIR_CHARS.items()
    }
    fill_atlas, fill_chars = _build_bucket_atlas(_FILL_CHARS, font_size, cell_w, cell_h)

    # ── Edge detection (whole image) ────────────────────────────────────────
    gray = np.array(img.convert("L"), dtype=np.float32)

    blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=_DOG_SIGMA1)
    blur2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=_DOG_SIGMA2)
    dog = blur1 - blur2
    edge_map = (dog > _DOG_THRESHOLD).astype(np.uint8)  # 1 where edge

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx)   # [-π, π]

    # Quantize angle → direction index (0–3) per pixel; -1 = no edge
    abs_norm = np.abs(angle) / np.pi   # [0, 1]
    direction_map = np.full(gray.shape, -1, dtype=np.int8)
    valid = magnitude > 1.0
    direction_map[valid & ((abs_norm < 0.05) | (abs_norm > 0.95))] = 0   # vertical
    direction_map[valid & ((abs_norm > 0.45) & (abs_norm < 0.55))] = 1   # horizontal
    direction_map[valid & (abs_norm >= 0.05) & (abs_norm <= 0.45) & (angle > 0)] = 2  # /
    direction_map[valid & (abs_norm >= 0.05) & (abs_norm <= 0.45) & (angle < 0)] = 3  # \
    direction_map[valid & (abs_norm >= 0.55) & (abs_norm <= 0.95) & (angle > 0)] = 3  # \
    direction_map[valid & (abs_norm >= 0.55) & (abs_norm <= 0.95) & (angle < 0)] = 2  # /

    # Trim to grid
    gray_trimmed = gray[:rows * cell_h, :cols * cell_w]
    edge_trimmed = edge_map[:rows * cell_h, :cols * cell_w]
    dir_trimmed = direction_map[:rows * cell_h, :cols * cell_w]

    cells_gray = gray_trimmed.reshape(rows, cell_h, cols, cell_w).transpose(0, 2, 1, 3)
    cells_edge = edge_trimmed.reshape(rows, cell_h, cols, cell_w).transpose(0, 2, 1, 3)
    cells_dir = dir_trimmed.reshape(rows, cell_h, cols, cell_w).transpose(0, 2, 1, 3)

    SS = _SUPERSAMPLE
    big_font = _load_font(font_size * SS)
    canvas = Image.new("RGB", (cols * cell_w * SS, rows * cell_h * SS), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for r in range(rows):
        for c in range(cols):
            cell = cells_gray[r, c]       # (cell_h, cell_w)
            cell_edges = cells_edge[r, c]
            cell_dirs = cells_dir[r, c]

            # Vote on dominant edge direction in this tile
            dominant_dir = -1
            if cell_edges.sum() >= _EDGE_TILE_THRESHOLD:
                counts = np.bincount(
                    cell_dirs[cell_dirs >= 0].flatten(), minlength=4
                )
                if counts.max() > 0:
                    dominant_dir = int(counts.argmax())

            if dominant_dir >= 0:
                atlas, chars = dir_atlases[dominant_dir]
            else:
                atlas, chars = fill_atlas, fill_chars

            ch, inverted = _best_match(cell, atlas, chars)

            x0, y0 = c * cell_w * SS, r * cell_h * SS
            x1, y1 = (c + 1) * cell_w * SS - 1, (r + 1) * cell_h * SS - 1
            if inverted:
                draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
                if ch != " ":
                    draw.text((x0, y0), ch, fill=(0, 0, 0), font=big_font)
            elif ch != " ":
                draw.text((x0, y0), ch, fill=(255, 255, 255), font=big_font)

    # Downsample + binarize for crisp output
    canvas_small = canvas.resize((cols * cell_w, rows * cell_h), Image.LANCZOS)
    canvas_arr = np.array(canvas_small)
    canvas_bin = ((canvas_arr > 127).astype(np.uint8) * 255)
    canvas = Image.fromarray(canvas_bin, "RGB")

    return canvas.resize(img.size, Image.LANCZOS)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil CJK bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./ascii-stencil-bot-output"))
    parser.add_argument("--char-height", type=float, default=0.03,
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
