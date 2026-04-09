"""Collage stencil oil spiral cells bleed bot.

Like oilspiralcells but with two additional effects:

1. Radial bleed from peaks — spiral arms are denser/wider near the peak
   center and thin out toward cell edges, creating a glow-from-center feel.

2. Fill warp toward spiral center — the fill image shown through the white
   stencil regions has its sampling coordinates pulled toward each cell's
   spiral peak, stretching/zooming the texture toward each origin.

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


def find_brightness_peaks(raw_gray: np.ndarray, n_peaks: int, min_dist_frac: float = 0.20) -> list:
    """Find bright-spot peaks from raw (non-enhanced) grayscale."""
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


def make_oilspiralcellsbleed_stencil(img: Image.Image, frequency: int = 35,
                                      warp_strength: float = 2.0, n_peaks: int = 6,
                                      topo_blend: float = 0.2,
                                      bleed_strength: float = 0.35,
                                      fill_warp_strength: float = 0.4,
                                      feather_bleed: float = 0.6,
                                      preprocess: bool = True):
    """Voronoi spiral-cells screen with radial bleed, feathered edges, and fill warp maps.

    Returns (mask, soft_alpha, x_fill, y_fill):
      mask       — hard binary PIL Image (for display)
      soft_alpha — float32 (h,w) feathered composite weight; 1.0 inside lines,
                   exponential decay outside line edges into background areas
      x_fill, y_fill — float32 cv2.remap maps pulling fill toward each spiral peak
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
    topo_gray = cv2.GaussianBlur(raw_gray.astype(np.float32), (0, 0), sigmaX=10)
    topo_phase = topo_gray / 255.0 * (max(h, w) / frequency)
    screen = ((1 - topo_blend) * spiral_phase + topo_blend * topo_phase) % 1.0

    # --- Threshold with radial bleed boost ---
    smoothed = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    gray_01 = smoothed.astype(np.float32) / 255.0
    gray_01 = np.clip(gray_01, 0.1, 0.9)

    r_norm = r_map / (r_max_map + 1e-6)                    # 0 at peak, ~1 at cell edge
    bleed_boost = bleed_strength * (1.0 - r_norm)          # max boost at center
    gray_boosted = np.clip(gray_01 + bleed_boost, 0.0, 1.0)
    binary = (gray_boosted > screen).astype(np.uint8) * 255

    # --- Fill warp maps ---
    x_fill = x_g.copy()
    y_fill = y_g.copy()

    for i, (px, py) in enumerate(peaks):
        cell_mask = (cell_idx == i)
        pull = fill_warp_strength * r_norm                  # 0 at peak, max at edge
        x_fill[cell_mask] = x_g[cell_mask] + pull[cell_mask] * (px - x_g[cell_mask])
        y_fill[cell_mask] = y_g[cell_mask] + pull[cell_mask] * (py - y_g[cell_mask])

    x_fill = np.clip(x_fill, 0, w - 1)
    y_fill = np.clip(y_fill, 0, h - 1)

    # --- Feathered bleed from line edges ---
    # Distance from white line regions, measured into the black (background) areas.
    # Exponential decay gives a smooth, organic spread — like pigment in oil film.
    feather_sigma = max(1.0, frequency * feather_bleed)
    dist_from_white = cv2.distanceTransform(
        (binary == 0).astype(np.uint8) * 255, cv2.DIST_L2, 5
    )
    feather_alpha = np.exp(-dist_from_white / feather_sigma)
    # Hard line interior is always 1.0; feathering extends beyond the edge
    soft_alpha = np.maximum(binary.astype(np.float32) / 255.0, feather_alpha)

    return Image.fromarray(binary), soft_alpha, x_fill, y_fill


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil oil spiral cells bleed bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./oilspiralcellsbleed-bot-output"))
    parser.add_argument("--frequency", type=int, default=35)
    parser.add_argument("--warp-strength", type=float, default=2.0)
    parser.add_argument("--n-peaks", type=int, default=6)
    parser.add_argument("--topo-blend", type=float, default=0.2,
                        help="Blend of topographic brightness contours into spiral (0=pure spiral, 1=pure topo)")
    parser.add_argument("--bleed-strength", type=float, default=0.35,
                        help="Radial density boost at spiral center (0=off)")
    parser.add_argument("--fill-warp-strength", type=float, default=0.4,
                        help="How far fill samples are pulled toward spiral peak (0=off, 1=fully collapsed)")
    parser.add_argument("--feather-bleed", type=float, default=0.6,
                        help="Feather spread from line edges as fraction of frequency (e.g. 0.6 = 21px at freq=35)")
    parser.add_argument("--no-preprocess", action="store_true")
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

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))
    images = [Image.open(p).convert("RGB") for p in source_paths]

    masks = []
    mask_paths = []
    soft_alphas = []
    fill_maps = []
    for i, img in enumerate(images):
        mask, soft_alpha, x_fill, y_fill = make_oilspiralcellsbleed_stencil(
            img, frequency=args.frequency, warp_strength=args.warp_strength,
            n_peaks=args.n_peaks, topo_blend=args.topo_blend,
            bleed_strength=args.bleed_strength,
            fill_warp_strength=args.fill_warp_strength,
            feather_bleed=args.feather_bleed,
            preprocess=not args.no_preprocess
        )
        dest = out_dir / f"oilspiralcellsbleed_mask_{i + 1}.png"
        mask.convert("RGB").save(dest)
        logger.info(f"Saved {dest.name}")
        masks.append(mask)
        mask_paths.append(dest)
        soft_alphas.append(soft_alpha)
        fill_maps.append((x_fill, y_fill))

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as oil spiral cells bleed stencil...")
        mw, mh = masks[s].size
        img_a_rs = images[a].convert("RGB").resize((mw, mh), Image.LANCZOS)
        img_b_rs = images[b].convert("RGB").resize((mw, mh), Image.LANCZOS)

        x_fill, y_fill = fill_maps[s]
        img_a_warped = cv2.remap(np.array(img_a_rs), x_fill, y_fill, cv2.INTER_LINEAR)

        alpha = soft_alphas[s][:, :, np.newaxis]   # (h, w, 1) float32
        composite = (alpha * img_a_warped.astype(np.float32)
                     + (1.0 - alpha) * np.array(img_b_rs).astype(np.float32))
        result = Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8))

        dest = out_dir / f"oilspiralcellsbleed_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    post_paths = output_paths + mask_paths

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths, bot_name="collage-stencil-oilspiralcells-bleed-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
