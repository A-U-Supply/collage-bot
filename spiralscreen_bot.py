"""Collage stencil spiral screen bot.

Generates an Archimedean spiral screen centered on the image, with arms
that locally warp to follow image contours (gradient-perturbed). Line width
varies with tone — thick in shadows, thin in highlights.

Posts all 6 variations plus the 3 spiral screen stencil masks.
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

_SCURVE_IN  = [  0,  64, 128, 192, 255]
_SCURVE_OUT = [  0,  25, 128, 230, 255]
_SCURVE_LUT = np.interp(np.arange(256), _SCURVE_IN, _SCURVE_OUT).astype(np.uint8)


def preprocess_for_screen(img: Image.Image) -> np.ndarray:
    gray = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return _SCURVE_LUT[enhanced]


def make_spiralscreen_stencil(img: Image.Image, frequency: int = 80, warp_strength: float = 0.3, preprocess: bool = True) -> Image.Image:
    """Archimedean spiral screen with gradient-perturbed arms.

    1. CLAHE + S-curve preprocessing
    2. Compute edge direction field (Sobel + double-angle smoothing)
    3. Warp pixel coordinates in the edge-perpendicular direction
    4. Compute spiral phase in warped coordinates
    5. Threshold against image tone
    """
    enhanced = preprocess_for_screen(img) if preprocess else np.array(img.convert("L"))
    h, w = enhanced.shape
    cx, cy = w / 2.0, h / 2.0

    # --- Direction field (same as curvy linescreen) ---
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=5)

    line_angle = np.arctan2(gy, gx) + np.pi / 2

    cos2 = cv2.GaussianBlur(np.cos(2 * line_angle), (0, 0), sigmaX=50)
    sin2 = cv2.GaussianBlur(np.sin(2 * line_angle), (0, 0), sigmaX=50)
    smooth_angle = np.arctan2(sin2, cos2) / 2

    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_thresh = np.percentile(mag, 85)
    edge_weight = cv2.GaussianBlur(
        np.clip(mag / (mag_thresh + 1e-6), 0, 1), (0, 0), sigmaX=30
    )
    edge_weight = np.clip(edge_weight * 3, 0, 1)

    # --- Warp coordinates along edge-perpendicular direction ---
    warp_pixels = frequency * warp_strength
    y_g, x_g = np.mgrid[0:h, 0:w].astype(np.float32)

    dx = edge_weight * warp_pixels * np.cos(smooth_angle)
    dy = edge_weight * warp_pixels * np.sin(smooth_angle)

    x_w = x_g + dx
    y_w = y_g + dy

    # --- Archimedean spiral phase in warped coordinates ---
    r = np.sqrt((x_w - cx) ** 2 + (y_w - cy) ** 2)
    theta = np.arctan2(y_w - cy, x_w - cx)

    # Archimedean spiral: r = (spacing/2π)*θ  →  phase coord = r - (spacing/2π)*θ
    spiral_coord = r - (frequency / (2 * np.pi)) * theta
    screen = (spiral_coord % frequency) / frequency

    # --- Threshold against tone ---
    smoothed = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    gray_01 = smoothed.astype(np.float32) / 255.0
    min_line = 0.1
    gray_01 = np.clip(gray_01, min_line, 1 - min_line)
    binary = (gray_01 > screen).astype(np.uint8) * 255
    return Image.fromarray(binary)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil spiral screen bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./spiralscreen-bot-output"))
    parser.add_argument("--frequency", type=int, default=80, help="Spiral arm spacing in pixels")
    parser.add_argument("--warp-strength", type=float, default=0.3, help="Gradient warp amount (fraction of frequency)")
    parser.add_argument("--no-preprocess", action="store_true")
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

    masks = []
    mask_paths = []
    for i, img in enumerate(images):
        mask = make_spiralscreen_stencil(img, frequency=args.frequency, warp_strength=args.warp_strength, preprocess=not args.no_preprocess)
        dest = out_dir / f"spiralscreen_mask_{i + 1}.png"
        mask.convert("RGB").save(dest)
        logger.info(f"Saved {dest.name}")
        masks.append(mask)
        mask_paths.append(dest)

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as spiral screen stencil...")
        result = apply_stencil(masks[s], images[a], images[b])
        dest = out_dir / f"spiralscreen_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    post_paths = output_paths + mask_paths

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths, bot_name="collage-stencil-spiralscreen-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
