"""Collage stencil soapy spiral bot.

Quadratic spiral screen (arms dense at outer edges, sparse near center)
centered on the most salient region of the image via spectral residual
saliency. Arms warp to follow image contours. Line width varies with tone.

Posts all 6 variations plus the 3 stencil masks.
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


def compute_saliency_center(gray: np.ndarray) -> tuple:
    """Return (cx, cy) — centroid of salient region via spectral residual saliency.

    Implements Hou & Zhang 2007: log amplitude spectrum minus its smoothed
    average gives a residual whose inverse FFT highlights salient regions.
    """
    h, w = gray.shape
    small = cv2.resize(gray, (64, 64)).astype(np.float32)
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-6)
    avg_amp = cv2.GaussianBlur(log_amp, (0, 0), sigmaX=3)
    sr = log_amp - avg_amp
    phase = np.angle(f)
    saliency = np.abs(np.fft.ifft2(np.exp(sr + 1j * phase))) ** 2
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (0, 0), sigmaX=5)
    sal_full = cv2.resize(saliency, (w, h))
    thresh = np.percentile(sal_full, 90)
    mask = (sal_full > thresh).astype(np.float32)
    y_g, x_g = np.mgrid[0:h, 0:w]
    cx = float(np.sum(mask * x_g) / (np.sum(mask) + 1e-6))
    cy = float(np.sum(mask * y_g) / (np.sum(mask) + 1e-6))
    logger.info(f"Saliency center: ({cx:.0f}, {cy:.0f})")
    return cx, cy


def make_soapyspiral_stencil(img: Image.Image, frequency: int = 40, warp_strength: float = 0.7, preprocess: bool = True) -> Image.Image:
    """Quadratic spiral screen with saliency-based center and gradient warp.

    1. CLAHE + S-curve preprocessing
    2. Spectral residual saliency to find spiral center
    3. Compute edge direction field (Sobel + double-angle smoothing)
    4. Warp pixel coordinates along edge-perpendicular direction
    5. Compute quadratic spiral phase (r² → dense outer edges)
    6. Threshold against image tone
    """
    enhanced = preprocess_for_screen(img) if preprocess else np.array(img.convert("L"))
    h, w = enhanced.shape

    cx, cy = compute_saliency_center(enhanced)

    # --- Direction field ---
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

    # --- Quadratic spiral phase (dense outer edges) ---
    r = np.sqrt((x_w - cx) ** 2 + (y_w - cy) ** 2)
    theta = np.arctan2(y_w - cy, x_w - cx)

    # r³ phase: stronger bubble — very sparse center, dense outer edges
    r_max = float(np.sqrt(cx ** 2 + cy ** 2 + (w - cx) ** 2 + (h - cy) ** 2) / np.sqrt(2))
    spiral_coord = (r ** 3) / (frequency * r_max ** 2) - theta / (2 * np.pi)
    screen = spiral_coord % 1.0

    # --- Threshold against tone ---
    smoothed = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    gray_01 = smoothed.astype(np.float32) / 255.0
    min_line = 0.1
    gray_01 = np.clip(gray_01, min_line, 1 - min_line)
    binary = (gray_01 > screen).astype(np.uint8) * 255
    return Image.fromarray(binary)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil soapy spiral bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./soapyspiral-bot-output"))
    parser.add_argument("--frequency", type=int, default=40, help="Spiral arm spacing in pixels")
    parser.add_argument("--warp-strength", type=float, default=0.7, help="Gradient warp amount (fraction of frequency)")
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
        mask = make_soapyspiral_stencil(img, frequency=args.frequency, warp_strength=args.warp_strength, preprocess=not args.no_preprocess)
        dest = out_dir / f"soapyspiral_mask_{i + 1}.png"
        mask.convert("RGB").save(dest)
        logger.info(f"Saved {dest.name}")
        masks.append(mask)
        mask_paths.append(dest)

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as soapy spiral stencil...")
        result = apply_stencil(masks[s], images[a], images[b])
        dest = out_dir / f"soapyspiral_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    post_paths = output_paths + mask_paths

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths, bot_name="collage-stencil-soapyspiral-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
