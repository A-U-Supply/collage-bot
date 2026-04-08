"""Collage stencil line screen bot.

Like collage-stencil-halftone-bot but uses edge-following line screen
instead of AM halftone dots. Lines curve and follow image contours —
thick in shadows, thin in highlights, wrapping around edges like an
engraving or topographic map. Flat regions fall back to 45° diagonal lines.
Posts all 6 variations plus the 3 line screen stencil masks.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Steep S-curve: pushes shadows and highlights hard to reduce midtone grey
_SCURVE_IN  = [  0,  64, 128, 192, 255]
_SCURVE_OUT = [  0,  25, 128, 230, 255]
_SCURVE_LUT = np.interp(np.arange(256), _SCURVE_IN, _SCURVE_OUT).astype(np.uint8)


def preprocess_for_screen(img: Image.Image) -> np.ndarray:
    """CLAHE + S-curve contrast boost before screen generation.

    Applied first so the gradient is computed on the enhanced image —
    weak edges in flat areas are strengthened before line directions
    are derived.
    Returns grayscale uint8 array.
    """
    gray = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return _SCURVE_LUT[enhanced]


def make_linescreen_stencil(img: Image.Image, frequency: int = 30, angle: float = 0.0, preprocess: bool = True) -> Image.Image:
    """Convert image to fixed-angle line screen binary mask.

    Clean parallel lines at a fixed angle. Line width varies with tone —
    bright areas have thin lines, dark areas have thick lines.
    """
    enhanced = preprocess_for_screen(img) if preprocess else np.array(img.convert("L"))
    h, w = enhanced.shape

    angle_rad = np.radians(angle)
    y_g, x_g = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = x_g * np.cos(angle_rad) + y_g * np.sin(angle_rad)
    screen = (phase % frequency) / frequency

    smoothed = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    gray_01 = smoothed.astype(np.float32) / 255.0
    min_line = 0.1
    gray_01 = np.clip(gray_01, min_line, 1 - min_line)
    binary = (gray_01 > screen).astype(np.uint8) * 255
    return Image.fromarray(binary)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil line screen bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./linescreen-bot-output"))
    parser.add_argument("--frequency", type=int, default=30, help="Line screen frequency in pixels")
    parser.add_argument("--angle", type=float, default=0.0, help="Line screen angle in degrees")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip CLAHE + S-curve preprocessing")
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages
    from stencil_transform import apply_stencil

    source_dir = args.output_dir / "source"
    out_dir = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))
    images = [Image.open(p).convert("RGB") for p in source_paths]

    # Pre-compute and save line screen masks for all 3 stencil images
    masks = []
    mask_paths = []
    for i, img in enumerate(images):
        mask = make_linescreen_stencil(img, frequency=args.frequency, angle=args.angle, preprocess=not args.no_preprocess)
        dest = out_dir / f"linescreen_mask_{i + 1}.png"
        mask.convert("RGB").save(dest)
        logger.info(f"Saved {dest.name}")
        masks.append(mask)
        mask_paths.append(dest)

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as line screen stencil, {a + 1} and {b + 1} as fill...")
        result = apply_stencil(masks[s], images[a], images[b])
        dest = out_dir / f"linescreen_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    post_paths = output_paths + mask_paths

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths, bot_name="collage-stencil-linescreen-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
