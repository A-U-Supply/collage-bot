"""Collage stencil halation-edge bot.

Like collage-stencil-bot but adds edge halation along stencil boundaries.
Fill images are full color originals — no additional filters applied.
Edge halation uses perlin-style noise, asymmetric white/dark bleed,
and a double hard edge at the mask boundary.
Posts 6 variations plus 4 GIFs as a single message.
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


def fractal_noise(h: int, w: int, scale: int = 60) -> np.ndarray:
    """Approximate Perlin-style noise via layered gaussian octaves (fBm)."""
    result = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    total = 0.0
    for _ in range(4):
        layer = np.random.normal(0, 1, (h, w)).astype(np.float32)
        sigma = max(scale / frequency, 1.0)
        layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=sigma)
        result += layer * amplitude
        total += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    return result / total


def apply_edge_halation(composite: np.ndarray, mask_gray: np.ndarray, width: int = 25) -> np.ndarray:
    """Add edge halation along stencil mask boundaries.

    - Perlin-style (fBm) grain texture
    - White bleed 48%, dark bleed 32% (white favored by ~20%)
    - Double edge: thin bright line on dark side, thin dark line on white side
    """
    h, w = mask_gray.shape
    grain = fractal_noise(h, w, scale=60) * 35

    # Wide soft bleed zones
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1))
    dilated = cv2.dilate(mask_gray, kernel)
    eroded = cv2.erode(mask_gray, kernel)
    white_bleed = cv2.subtract(dilated, mask_gray)
    black_bleed = cv2.subtract(mask_gray, eroded)

    sigma = width * 0.4
    white_zone = cv2.GaussianBlur(white_bleed.astype(np.float32), (0, 0), sigmaX=sigma) / 255.0
    black_zone = cv2.GaussianBlur(black_bleed.astype(np.float32), (0, 0), sigmaX=sigma) / 255.0

    # Double edge: thin hard lines right at the boundary
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_edge_zone = cv2.subtract(cv2.dilate(mask_gray, thin_kernel), mask_gray).astype(np.float32) / 255.0
    dark_edge_zone = cv2.subtract(mask_gray, cv2.erode(mask_gray, thin_kernel)).astype(np.float32) / 255.0

    result = composite.astype(np.float32)

    # Equal bleed: white and dark both at 40%
    white_grain = np.clip(210 + grain, 160, 255)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - white_zone * 0.40) + white_grain * (white_zone * 0.40)

    dark_grain = np.clip(40 + grain, 0, 90)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - black_zone * 0.40) + dark_grain * (black_zone * 0.40)

    # Double edge lines
    bright_edge = np.clip(240 + grain * 0.3, 220, 255)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - bright_edge_zone * 0.7) + bright_edge * (bright_edge_zone * 0.7)

    dark_edge = np.clip(20 + grain * 0.3, 0, 50)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - dark_edge_zone * 0.7) + dark_edge * (dark_edge_zone * 0.7)

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil halation-edge bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./halationedge-bot-output"))
    parser.add_argument("--frame-duration", type=int, default=25, help="GIF frame duration in ms")
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages
    from stencil_transform import make_stencil, apply_stencil
    from gif_bot import make_gif

    source_dir = args.output_dir / "source"
    out_dir = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = fetch_random_images(token, args.source_channel, 3, source_dir)
    images = [Image.open(p).convert("RGB") for p in source_paths]

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as stencil, {a + 1} and {b + 1} as fill...")
        mask = make_stencil(images[s])
        composite = apply_stencil(mask, images[a], images[b])
        composite_arr = apply_edge_halation(np.array(composite), np.array(mask))
        result = Image.fromarray(composite_arr)
        dest = out_dir / f"halationedge_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    gif_path = out_dir / f"halationedge_{args.frame_duration}ms.gif"
    logger.info(f"Creating GIF at {args.frame_duration}ms/frame...")
    gif_order = [0, 3, 1, 4, 2, 5]
    make_gif([output_paths[i] for i in gif_order], gif_path, frame_duration_ms=args.frame_duration)

    gif_pair_12 = out_dir / f"halationedge_pair_12_{args.frame_duration}ms.gif"
    gif_pair_34 = out_dir / f"halationedge_pair_34_{args.frame_duration}ms.gif"
    gif_pair_56 = out_dir / f"halationedge_pair_56_{args.frame_duration}ms.gif"
    make_gif([output_paths[0], output_paths[1]], gif_pair_12, frame_duration_ms=args.frame_duration)
    make_gif([output_paths[2], output_paths[3]], gif_pair_34, frame_duration_ms=args.frame_duration)
    make_gif([output_paths[4], output_paths[5]], gif_pair_56, frame_duration_ms=args.frame_duration)

    post_paths = output_paths + [gif_path, gif_pair_12, gif_pair_34, gif_pair_56]

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths, bot_name="collage-stencil-halationedge-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
