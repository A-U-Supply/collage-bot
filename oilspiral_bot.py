"""Collage stencil oil spiral bot.

Detects bright peaks in the image and seeds one r³ spiral per peak.
Each pixel belongs to its nearest peak's Voronoi cell — where adjacent
cells meet, arms from neighboring spirals converge into organic closed-loop
boundaries that resemble iridescent oil-on-water or soap bubble patterns.

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


def find_brightness_peaks(gray: np.ndarray, n_peaks: int, min_dist_frac: float = 0.15) -> list:
    """Find up to n_peaks local brightness maxima in the image.

    Uses a heavily blurred version of the image to find broad regional maxima
    — bright areas that represent subject matter rather than specular highlights.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=80)

    for frac in (min_dist_frac, min_dist_frac * 0.6, min_dist_frac * 0.3):
        radius = max(4, int(min(h, w) * frac))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        dilated = cv2.dilate(blurred, kernel)
        local_max = (blurred >= dilated - 0.1) & (blurred > np.percentile(blurred, 70))
        ys, xs = np.where(local_max)
        if len(xs) >= 2:
            values = blurred[ys, xs]
            order = np.argsort(values)[::-1]
            peaks = [(int(xs[i]), int(ys[i])) for i in order[:n_peaks]]
            logger.info(f"Found {len(peaks)} brightness peaks (radius={radius}px)")
            return peaks

    # Fallback: evenly spaced grid
    logger.info("Peak detection fallback: using grid positions")
    return [(w // 4, h // 4), (3 * w // 4, h // 4), (w // 2, h // 2),
            (w // 4, 3 * h // 4), (3 * w // 4, 3 * h // 4)][:n_peaks]


def make_oilspiral_stencil(img: Image.Image, frequency: int = 20, warp_strength: float = 4.0,
                            n_peaks: int = 8, preprocess: bool = True) -> Image.Image:
    """Voronoi multi-spiral screen with gradient warp.

    1. CLAHE + S-curve preprocessing
    2. Detect n_peaks brightness peaks as spiral origins
    3. Compute edge direction field (Sobel + double-angle smoothing)
    4. Warp pixel coordinates along edge-perpendicular direction
    5. Assign each (warped) pixel to its nearest peak (Voronoi)
    6. Compute r³ spiral phase relative to that cell's peak
    7. Threshold against image tone
    """
    enhanced = preprocess_for_screen(img) if preprocess else np.array(img.convert("L"))
    h, w = enhanced.shape

    peaks = find_brightness_peaks(enhanced, n_peaks)

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

    # --- Warp coordinates ---
    warp_pixels = frequency * warp_strength
    y_g, x_g = np.mgrid[0:h, 0:w].astype(np.float32)
    x_w = x_g + edge_weight * warp_pixels * np.cos(smooth_angle)
    y_w = y_g + edge_weight * warp_pixels * np.sin(smooth_angle)

    # --- Voronoi: assign each pixel to nearest peak ---
    dist_maps = [np.sqrt((x_w - px) ** 2 + (y_w - py) ** 2) for px, py in peaks]
    dist_stack = np.stack(dist_maps, axis=0)   # (n_peaks, h, w)
    cell_idx = np.argmin(dist_stack, axis=0)   # (h, w)
    r_map = np.min(dist_stack, axis=0)         # distance to assigned peak

    # --- Spiral phase per cell ---
    theta_map = np.zeros((h, w), dtype=np.float32)
    r_max_map = np.zeros((h, w), dtype=np.float32)
    corners = [(0, 0), (w, 0), (0, h), (w, h)]

    for i, (px, py) in enumerate(peaks):
        mask = (cell_idx == i)
        theta_map[mask] = np.arctan2(y_w[mask] - py, x_w[mask] - px)
        r_max = max(np.sqrt((cx - px) ** 2 + (cy - py) ** 2) for cx, cy in corners)
        r_max_map[mask] = max(r_max, 1.0)

    spiral_coord = r_map ** 3 / (frequency * r_max_map ** 2 + 1e-6) - theta_map / (2 * np.pi)
    screen = spiral_coord % 1.0

    # --- Threshold against tone ---
    smoothed = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    gray_01 = smoothed.astype(np.float32) / 255.0
    gray_01 = np.clip(gray_01, 0.1, 0.9)
    binary = (gray_01 > screen).astype(np.uint8) * 255
    return Image.fromarray(binary)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil oil spiral bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./oilspiral-bot-output"))
    parser.add_argument("--frequency", type=int, default=20)
    parser.add_argument("--warp-strength", type=float, default=4.0)
    parser.add_argument("--n-peaks", type=int, default=8)
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
        mask = make_oilspiral_stencil(
            img, frequency=args.frequency, warp_strength=args.warp_strength,
            n_peaks=args.n_peaks, preprocess=not args.no_preprocess
        )
        dest = out_dir / f"oilspiral_mask_{i + 1}.png"
        mask.convert("RGB").save(dest)
        logger.info(f"Saved {dest.name}")
        masks.append(mask)
        mask_paths.append(dest)

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as oil spiral stencil...")
        result = apply_stencil(masks[s], images[a], images[b])
        dest = out_dir / f"oilspiral_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    post_paths = output_paths + mask_paths

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths, bot_name="collage-stencil-oilspiral-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
