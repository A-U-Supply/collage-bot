"""Collage stencil oil spiral cells bleed bot.

Voronoi spiral-cells stencil. Fill images are composited at the largest available
resolution inside the hard stencil boundary and processed with chrome/water effects:

Fill effects applied in order (both layers):
1. Sinusoidal ripple warp  — distortion perpendicular to arm direction (rippling water /
                              curved chrome reflection)
2. Chromatic aberration    — R/B split along arm direction (prismatic edge sheen)
3. Contrast stretch        — deepen darks, punch highlights (chrome look)
4. Bulge warp              — stretch pixels near spine, compress at edges; creates the
                              appearance of raised liquid rivulets without any overlay

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
                                      warp_strength: float = 4.0, n_peaks: int = 6,
                                      topo_blend: float = 0.2,
                                      bleed_strength: float = 0.35,
                                      preprocess: bool = True):
    """Voronoi spiral-cells screen.

    Heavy computation is capped at PROC_MAX pixels on the longest dimension and
    upsampled back to original resolution at the end.

    Returns (mask, smooth_angle):
      mask         — hard binary PIL Image at original resolution
      smooth_angle — float32 (h,w) local arm direction field, used for fill effects
    """
    raw_gray = np.array(img.convert("L"))
    enhanced = preprocess_for_screen(img) if preprocess else raw_gray.copy()
    h, w = enhanced.shape

    # --- Downsample FIRST so all heavy computation runs at capped resolution ---
    # This includes peak detection (avoids huge dilation kernels on full-res images)
    PROC_MAX = 768
    scale = min(1.0, PROC_MAX / max(h, w))
    if scale < 1.0:
        ph, pw = int(h * scale), int(w * scale)
        enhanced = cv2.resize(enhanced, (pw, ph), interpolation=cv2.INTER_AREA)
        raw_gray = cv2.resize(raw_gray, (pw, ph), interpolation=cv2.INTER_AREA)
    else:
        ph, pw = h, w

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
    y_g, x_g = np.mgrid[0:ph, 0:pw].astype(np.float32)
    x_w = x_g + edge_weight * warp_pixels * np.cos(smooth_angle)
    y_w = y_g + edge_weight * warp_pixels * np.sin(smooth_angle)

    # --- Voronoi: assign each pixel to nearest peak ---
    dist_maps = [np.sqrt((x_w - px) ** 2 + (y_w - py) ** 2) for px, py in peaks]
    dist_stack = np.stack(dist_maps, axis=0)
    cell_idx = np.argmin(dist_stack, axis=0)
    r_map = np.min(dist_stack, axis=0)

    # --- Spiral phase per cell ---
    theta_map = np.zeros((ph, pw), dtype=np.float32)
    r_max_map = np.zeros((ph, pw), dtype=np.float32)
    corners = [(0, 0), (pw, 0), (0, ph), (pw, ph)]

    for i, (px, py) in enumerate(peaks):
        mask = (cell_idx == i)
        theta_map[mask] = np.arctan2(y_w[mask] - py, x_w[mask] - px)
        r_max = max(np.sqrt((cx - px) ** 2 + (cy - py) ** 2) for cx, cy in corners)
        r_max_map[mask] = max(r_max, 1.0)

    spiral_phase = r_map / frequency - theta_map / (2 * np.pi)

    # --- Brightness contour phase (topographic) ---
    topo_gray = cv2.GaussianBlur(raw_gray.astype(np.float32), (0, 0), sigmaX=10)
    topo_phase = topo_gray / 255.0 * (max(ph, pw) / frequency)
    screen = ((1 - topo_blend) * spiral_phase + topo_blend * topo_phase) % 1.0

    # --- Threshold with radial bleed boost ---
    smoothed = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    gray_01 = smoothed.astype(np.float32) / 255.0
    gray_01 = np.clip(gray_01, 0.1, 0.9)

    r_norm = r_map / (r_max_map + 1e-6)                    # 0 at peak, ~1 at cell edge
    bleed_boost = bleed_strength * (1.0 - r_norm)          # max boost at center
    gray_boosted = np.clip(gray_01 + bleed_boost, 0.0, 1.0)
    binary = (gray_boosted > screen).astype(np.uint8) * 255

    # --- Upsample results back to original resolution ---
    if scale < 1.0:
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
        smooth_angle = cv2.resize(smooth_angle, (w, h), interpolation=cv2.INTER_LINEAR)

    return Image.fromarray(binary), smooth_angle


def apply_fill_effects(img_arr: np.ndarray, smooth_angle: np.ndarray,
                       chroma_shift: int = 10,
                       ripple_amplitude: int = 20,
                       ripple_wavelength: float = 40.0) -> np.ndarray:
    """Apply chrome / rippling-water distortion to a fill layer.

    1. Sinusoidal ripple warp  — pixels displaced perp to arm direction
    2. Chromatic aberration    — R/B split along arm direction
    3. Contrast stretch        — punch darks and lights for chrome look
    """
    h, w = img_arr.shape[:2]
    y_g, x_g = np.mgrid[0:h, 0:w].astype(np.float32)

    cos_a = np.cos(smooth_angle)
    sin_a = np.sin(smooth_angle)

    # 1. Sinusoidal ripple — displace perpendicular to arm direction
    phase = x_g * cos_a + y_g * sin_a
    ripple = (ripple_amplitude * np.sin(2.0 * np.pi * phase / ripple_wavelength)).astype(np.float32)
    xs = np.clip(x_g + ripple * (-sin_a), 0, w - 1)
    ys = np.clip(y_g + ripple * cos_a, 0, h - 1)
    out = cv2.remap(img_arr, xs, ys, cv2.INTER_LINEAR)

    # 2. Chromatic aberration — R forward, B backward along arm direction
    xs_r = np.clip(x_g + chroma_shift * cos_a, 0, w - 1)
    ys_r = np.clip(y_g + chroma_shift * sin_a, 0, h - 1)
    xs_b = np.clip(x_g - chroma_shift * cos_a, 0, w - 1)
    ys_b = np.clip(y_g - chroma_shift * sin_a, 0, h - 1)
    r_ch = cv2.remap(out[:, :, 0], xs_r, ys_r, cv2.INTER_LINEAR)
    b_ch = cv2.remap(out[:, :, 2], xs_b, ys_b, cv2.INTER_LINEAR)
    out = np.stack([r_ch, out[:, :, 1], b_ch], axis=2)

    # 3. Contrast stretch — S-curve to deepen darks and punch highlights (chrome)
    lut = np.arange(256, dtype=np.float32)
    lut = 128 + 128 * np.tanh((lut - 128) / 85.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    out = lut[out]

    return out


def apply_bulge_warp(img_arr: np.ndarray, mask_L: np.ndarray,
                     bulge_strength: float = 0.6) -> np.ndarray:
    """Warp fill pixels to make stencil lines look like raised liquid rivulets.

    Uses the distance transform inside white regions as a height map. The spine
    (centre of each line) has height 1; edges have height 0.

    Source coordinates are displaced *away* from the spine proportional to
    (1 - height), so:
    - Spine pixels sample with no displacement  → texture looks stretched/magnified
    - Edge pixels sample from further outside   → texture is compressed
    This creates the appearance of a convex surface without any brightness overlay.
    """
    h, w = img_arr.shape[:2]
    y_g, x_g = np.mgrid[0:h, 0:w].astype(np.float32)

    inside = (mask_L > 127).astype(np.uint8) * 255
    dist_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5)

    # Normalise so thin and thick lines get the same 0→1 range
    nonzero = dist_in[dist_in > 0]
    scale = float(np.percentile(nonzero, 95)) if len(nonzero) else 1.0
    height = np.clip(dist_in / (scale + 1e-6), 0, 1).astype(np.float32)

    # Smooth dist_in before computing gradient so displacement direction is smooth
    dist_smooth = cv2.GaussianBlur(dist_in, (0, 0), sigmaX=scale * 0.5)
    gx_h = cv2.Sobel(dist_smooth, cv2.CV_32F, 1, 0, ksize=5)
    gy_h = cv2.Sobel(dist_smooth, cv2.CV_32F, 0, 1, ksize=5)
    grad_mag = np.sqrt(gx_h ** 2 + gy_h ** 2) + 1e-6
    nx_h = gx_h / grad_mag
    ny_h = gy_h / grad_mag

    # Displacement: max at edge (height=0), zero at spine (height=1)
    # Subtracting the toward-spine direction = pushing source coords away from spine
    displacement = (bulge_strength * scale * (1.0 - height)).astype(np.float32)
    src_x = np.clip(x_g - displacement * nx_h, 0, w - 1)
    src_y = np.clip(y_g - displacement * ny_h, 0, h - 1)

    return cv2.remap(img_arr, src_x, src_y, cv2.INTER_LINEAR)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil oil spiral cells bleed bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./oilspiralcellsbleed-bot-output"))
    parser.add_argument("--frequency", type=int, default=15)
    parser.add_argument("--warp-strength", type=float, default=4.0)
    parser.add_argument("--n-peaks", type=int, default=6)
    parser.add_argument("--topo-blend", type=float, default=0.2)
    parser.add_argument("--bleed-strength", type=float, default=0.35)
    # Fill effect params
    parser.add_argument("--chroma-shift", type=int, default=10,
                        help="Chromatic aberration shift in pixels along arm direction")
    parser.add_argument("--ripple-amplitude", type=int, default=20,
                        help="Sinusoidal ripple displacement in pixels (chrome/water warp)")
    parser.add_argument("--ripple-wavelength", type=float, default=40.0,
                        help="Ripple wavelength in pixels")
    parser.add_argument("--bulge-strength", type=float, default=0.8,
                        help="Rivulet bulge warp strength — 0 = off, 1 = full")
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
    smooth_angles = []
    for i, img in enumerate(images):
        mask, smooth_angle = make_oilspiralcellsbleed_stencil(
            img, frequency=args.frequency, warp_strength=args.warp_strength,
            n_peaks=args.n_peaks, topo_blend=args.topo_blend,
            bleed_strength=args.bleed_strength,
            preprocess=not args.no_preprocess
        )
        dest = out_dir / f"oilspiralcellsbleed_mask_{i + 1}.png"
        mask.convert("RGB").save(dest)
        logger.info(f"Saved {dest.name}")
        masks.append(mask)
        mask_paths.append(dest)
        smooth_angles.append(smooth_angle)

    # Output at the largest source image resolution
    target_w, target_h = max((img.size for img in images), key=lambda s: s[0] * s[1])
    logger.info(f"Output resolution: {target_w}×{target_h}")

    # Scale pixel-based params to output resolution (calibrated at 1024px short side)
    res_scale = min(target_w, target_h) / 1024.0
    fx_kwargs = dict(
        chroma_shift=int(args.chroma_shift * res_scale),
        ripple_amplitude=int(args.ripple_amplitude * res_scale),
        ripple_wavelength=args.ripple_wavelength * res_scale,
    )

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as oil spiral cells bleed stencil...")

        img_a_arr = np.array(images[a].convert("RGB").resize((target_w, target_h), Image.LANCZOS))
        img_b_arr = np.array(images[b].convert("RGB").resize((target_w, target_h), Image.LANCZOS))
        mask_resized = np.array(masks[s].resize((target_w, target_h), Image.NEAREST).convert("L"))
        angle_resized = cv2.resize(smooth_angles[s], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        angle_b = cv2.resize(smooth_angles[b], (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        img_a_processed = apply_fill_effects(img_a_arr, angle_resized, **fx_kwargs)
        img_b_processed = apply_fill_effects(img_b_arr, angle_b, **fx_kwargs)

        # Bulge warp — applied after fill effects so ripple/chroma are also stretched
        img_a_processed = apply_bulge_warp(img_a_processed, mask_resized, args.bulge_strength)
        img_b_processed = apply_bulge_warp(img_b_processed, mask_resized, args.bulge_strength)

        # Composite: hard binary mask — fill only inside stencil lines
        composite = np.where(
            mask_resized[:, :, np.newaxis] > 127,
            img_a_processed.astype(np.float32),
            img_b_processed.astype(np.float32)
        ).astype(np.uint8)

        # Unsharp mask — restore crispness lost in warping steps
        blurred = cv2.GaussianBlur(composite, (0, 0), sigmaX=2.0)
        sharp = cv2.addWeighted(composite, 1.8, blurred, -0.8, 0)
        result = Image.fromarray(sharp)

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
