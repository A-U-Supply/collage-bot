"""Collage stencil displacement map bot.

Uses the stencil image's gradient as a displacement map two ways:
1. Warps the stencil itself before thresholding — the seam/cut line distorts
2. Warps the fill images with high strength — content near the edge is pulled far
Both effects compound: the boundary tears AND the fills stretch across it.
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


def compute_displacement(img: Image.Image, strength: float = 200.0, blur: float = 3.0) -> tuple:
    """Compute displacement fields from stencil image gradient.

    strength: max displacement in pixels (high = dramatic)
    blur: gradient smoothing sigma (low = sharp local warps, high = smooth sweeping)
    """
    gray = np.array(img.convert("L")).astype(np.float32)
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur)

    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=5)

    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_max = np.percentile(mag, 99) + 1e-6
    gx_norm = gx / mag_max
    gy_norm = gy / mag_max

    dx = gx_norm * strength
    dy = gy_norm * strength

    return dx, dy


def compute_fill_envelope(mask_arr: np.ndarray, falloff: float = 150.0) -> np.ndarray:
    """Gaussian envelope peaking at the stencil edge, fading over falloff pixels."""
    mask_u8 = mask_arr.astype(np.uint8)
    edge = cv2.Canny(mask_u8, 50, 150)
    dist = cv2.distanceTransform((255 - edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.exp(-(dist ** 2) / (2 * falloff ** 2))


def displace_image(img: Image.Image, dx: np.ndarray, dy: np.ndarray) -> Image.Image:
    """Warp an image using per-pixel displacement vectors."""
    w, h = img.size
    img_arr = np.array(img.convert("RGB")).astype(np.float32)

    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)

    src_x = np.clip(x_grid + dx, 0, w - 1)
    src_y = np.clip(y_grid + dy, 0, h - 1)

    warped = cv2.remap(
        img_arr,
        src_x.astype(np.float32),
        src_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return Image.fromarray(warped.astype(np.uint8))


def apply_stencil_with_displacement(
    stencil: Image.Image,
    img_a: Image.Image,
    img_b: Image.Image,
    strength: float = 200.0,
    blur: float = 3.0,
    fill_falloff: float = 600.0,
) -> Image.Image:
    """Warp both the stencil boundary and the fills, then composite.

    The stencil is warped before thresholding so the cut line itself distorts.
    The fills stretch hard near the seam and taper off over fill_falloff pixels.
    """
    w, h = stencil.size
    img_a = img_a.convert("RGB").resize((w, h), Image.LANCZOS)
    img_b = img_b.convert("RGB").resize((w, h), Image.LANCZOS)

    dx, dy = compute_displacement(stencil, strength=strength, blur=blur)

    # Warp the stencil at half strength — seam distorts more subtly
    warped_stencil = displace_image(stencil, dx * 0.5, dy * 0.5)
    from stencil_transform import make_stencil
    mask = make_stencil(warped_stencil)
    mask_arr = np.array(mask)

    # Envelope: double strength at the seam, fades to zero over fill_falloff pixels
    envelope = compute_fill_envelope(mask_arr, falloff=fill_falloff)
    dx_fill = dx * 2.0 * envelope
    dy_fill = dy * 2.0 * envelope

    warped_a = displace_image(img_a, -dx_fill, -dy_fill)
    warped_b = displace_image(img_b, dx_fill, dy_fill)

    a_arr = np.array(warped_a)
    b_arr = np.array(warped_b)
    composite = np.where(mask_arr[:, :, np.newaxis] > 0, a_arr, b_arr)
    return Image.fromarray(composite.astype(np.uint8))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil displacement map bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./displacement-bot-output"))
    parser.add_argument("--strength", type=float, default=200.0, help="Max displacement in pixels")
    parser.add_argument("--blur", type=float, default=3.0, help="Gradient smoothing sigma")
    parser.add_argument("--fill-falloff", type=float, default=600.0, help="Gaussian sigma for fill stretch falloff distance")
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages
    from stencil_transform import make_stencil

    source_dir = args.output_dir / "source"
    out_dir = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = fetch_random_images(token, args.source_channel, 3, source_dir)
    images = [Image.open(p).convert("RGB") for p in source_paths]

    # Pre-compute and save stencil masks (warped versions)
    masks = []
    mask_paths = []
    for i, img in enumerate(images):
        dx, dy = compute_displacement(img, strength=args.strength, blur=args.blur)
        warped_stencil = displace_image(img, dx, dy)
        mask = make_stencil(warped_stencil)
        dest = out_dir / f"displacement_mask_{i + 1}.png"
        mask.convert("RGB").save(dest)
        logger.info(f"Saved {dest.name}")
        masks.append((mask, dx, dy))
        mask_paths.append(dest)

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as stencil, {a + 1} and {b + 1} as fill...")
        result = apply_stencil_with_displacement(
            images[s], images[a], images[b],
            strength=args.strength, blur=args.blur, fill_falloff=args.fill_falloff,
        )
        dest = out_dir / f"displacement_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    post_paths = output_paths + mask_paths

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths, bot_name="collage-stencil-displacement-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
