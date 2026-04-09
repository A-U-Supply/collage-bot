"""Collage stencil oil spiral cells bot.

Detects true bright spots in the raw image (not CLAHE-enhanced) as spiral
origins. Each pixel belongs to its nearest peak's Voronoi cell. Within each
cell the spiral phase is blended with a brightness-contour phase, so lines
follow both the spiral rotation and iso-brightness contours — like a
topographic map seeded from light sources. Adjacent cells butt up against
each other forming organic closed-loop boundaries.

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


def find_brightness_peaks(raw_gray: np.ndarray, n_peaks: int, min_dist_frac: float = 0.12) -> list:
    """Find bright-spot peaks from raw (non-enhanced) grayscale.

    Uses a blurred version of the original image so origins land on genuine
    highlights and bright areas rather than high-contrast subject regions.
    """
    h, w = raw_gray.shape
    blurred = cv2.GaussianBlur(raw_gray.astype(np.float32), (0, 0), sigmaX=60)

    for frac in (min_dist_frac, min_dist_frac * 0.6, min_dist_frac * 0.3):
        radius = max(4, int(min(h, w) * frac))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        dilated = cv2.dilate(blurred, kernel)
        local_max = (blurred >= dilated - 0.1) & (blurred > np.percentile(blurred, 60))
        ys, xs = np.where(local_max)
        if len(xs) >= 2:
            values = blurred[ys, xs]
            order = np.argsort(values)[::-1]
            peaks = [(int(xs[i]), int(ys[i])) for i in order[:n_peaks]]
            logger.info(f"Found {len(peaks)} brightness peaks (radius={radius}px)")
            return peaks

    logger.info("Peak detection fallback: using grid positions")
    cols, rows = 4, 4
    return [(int(w * (c + 0.5) / cols), int(h * (r + 0.5) / rows))
            for r in range(rows) for c in range(cols)][:n_peaks]


def make_oilspiralcells_stencil(img: Image.Image, frequency: int = 20,
                                 warp_strength: float = 2.0, n_peaks: int = 16,
                                 topo_blend: float = 0.4,
                                 preprocess: bool = True) -> Image.Image:
    """Voronoi spiral-cells screen with brightness-contour blending.

    1. Detect bright peaks from raw grayscale (no CLAHE) → spiral origins
    2. CLAHE + S-curve for tone/direction processing
    3. Edge direction field (Sobel + double-angle smoothing)
    4. Warp pixel coordinates along edge-perpendicular direction
    5. Assign each warped pixel to nearest peak (Voronoi)
    6. Compute r³ spiral phase relative to cell peak
    7. Blend spiral phase with brightness contour phase (topo_blend)
    8. Threshold against image tone
    """
    raw_gray = np.array(img.convert("L"))
    enhanced = preprocess_for_screen(img) if preprocess else raw_gray.copy()
    h, w = enhanced.shape

    peaks = find_brightness_peaks(raw_gray, n_peaks)

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
    dist_stack = np.stack(dist_maps, axis=0)
    cell_idx = np.argmin(dist_stack, axis=0)
    r_map = np.min(dist_stack, axis=0)

    # --- Spiral phase per cell ---
    theta_map = np.zeros((h, w), dtype=np.float32)
    r_max_map = np.zeros((h, w), dtype=np.float32)
    corners = [(0, 0), (w, 0), (0, h), (w, h)]

    for i, (px, py) in enumerate(peaks):
        mask = (cell_idx == i)
        theta_map[mask] = np.arctan2(y_w[mask] - py, x_w[mask] - px)
        r_max = max(np.sqrt((cx - px) ** 2 + (cy - py) ** 2) for cx, cy in corners)
        r_max_map[mask] = max(r_max, 1.0)

    spiral_phase = r_map ** 3 / (frequency * r_max_map ** 2 + 1e-6) - theta_map / (2 * np.pi)

    # --- Brightness contour phase (topographic) ---
    # Smooth raw brightness → iso-brightness contours at fixed intervals
    topo_gray = cv2.GaussianBlur(raw_gray.astype(np.float32), (0, 0), sigmaX=10)
    topo_phase = topo_gray / 255.0 * (max(h, w) / frequency)

    # Blend: topo_blend=0 → pure spiral, topo_blend=1 → pure topographic
    screen = ((1 - topo_blend) * spiral_phase + topo_blend * topo_phase) % 1.0

    # --- Threshold against tone ---
    smoothed = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    gray_01 = smoothed.astype(np.float32) / 255.0
    gray_01 = np.clip(gray_01, 0.1, 0.9)
    binary = (gray_01 > screen).astype(np.uint8) * 255
    return Image.fromarray(binary)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil oil spiral cells bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./oilspiralcells-bot-output"))
    parser.add_argument("--frequency", type=int, default=20)
    parser.add_argument("--warp-strength", type=float, default=2.0)
    parser.add_argument("--n-peaks", type=int, default=16)
    parser.add_argument("--topo-blend", type=float, default=0.4,
                        help="Blend of topographic brightness contours into spiral (0=pure spiral, 1=pure topo)")
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
        mask = make_oilspiralcells_stencil(
            img, frequency=args.frequency, warp_strength=args.warp_strength,
            n_peaks=args.n_peaks, topo_blend=args.topo_blend,
            preprocess=not args.no_preprocess
        )
        dest = out_dir / f"oilspiralcells_mask_{i + 1}.png"
        mask.convert("RGB").save(dest)
        logger.info(f"Saved {dest.name}")
        masks.append(mask)
        mask_paths.append(dest)

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as oil spiral cells stencil...")
        result = apply_stencil(masks[s], images[a], images[b])
        dest = out_dir / f"oilspiralcells_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    post_paths = output_paths + mask_paths

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths, bot_name="collage-stencil-oilspiralcells-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
