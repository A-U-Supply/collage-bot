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


def erosion_noise(h: int, w: int, scale: int = 40, iterations: int = 30) -> np.ndarray:
    """Thermal erosion simulation for organic ridge/valley edge texture.

    Material flows from steep slopes to lower neighbors, creating
    characteristic ridgelines and eroded valleys.
    Returns values in [-1, 1].
    """
    terrain = np.random.rand(h, w).astype(np.float32)
    terrain = cv2.GaussianBlur(terrain, (0, 0), sigmaX=max(scale / 3, 1.0))

    talus = 0.03  # max stable slope before material moves
    rate = 0.4
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.float32) / 4

    for _ in range(iterations):
        avg = cv2.filter2D(terrain, -1, kernel)
        diff = terrain - avg
        erode = np.maximum(diff - talus, 0)
        terrain -= erode * rate
        terrain += cv2.filter2D(erode * rate, -1, kernel)

    t_min, t_max = terrain.min(), terrain.max()
    terrain = (terrain - t_min) / (t_max - t_min + 1e-6)
    return terrain * 2 - 1  # normalize to [-1, 1]


def create_noisy_mask(mask_gray: np.ndarray, width: int = 10) -> np.ndarray:
    """Replace the hard stencil edge with perlin-style grain before compositing.

    In the edge transition zone the mask value is replaced with fractal noise,
    creating organic grainy blending where the fills meet.
    Outside the edge zone the original binary mask is preserved.
    """
    h, w = mask_gray.shape
    grain = erosion_noise(h, w, scale=60)
    noise_01 = (grain - grain.min()) / (grain.max() - grain.min() + 1e-6)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1))
    dilated = cv2.dilate(mask_gray, kernel).astype(np.float32) / 255.0
    eroded = cv2.erode(mask_gray, kernel).astype(np.float32) / 255.0
    edge_zone = dilated - eroded  # 1 in transition zone, 0 elsewhere

    mask_01 = mask_gray.astype(np.float32) / 255.0
    noisy = mask_01 * (1 - edge_zone) + noise_01 * edge_zone
    return (np.clip(noisy, 0, 1) * 255).astype(np.uint8)


def blend_with_noisy_mask(mask: Image.Image, img_a: Image.Image, img_b: Image.Image, width: int = 10) -> np.ndarray:
    """Composite two images using a soft noisy mask, then add a double hard edge."""
    w, h = mask.size
    img_a = img_a.convert("RGB").resize((w, h), Image.LANCZOS)
    img_b = img_b.convert("RGB").resize((w, h), Image.LANCZOS)

    mask_gray = np.array(mask)
    noisy_mask = create_noisy_mask(mask_gray, width=width)
    alpha = noisy_mask.astype(np.float32)[:, :, np.newaxis] / 255.0

    a_arr = np.array(img_a).astype(np.float32)
    b_arr = np.array(img_b).astype(np.float32)
    result = a_arr * alpha + b_arr * (1 - alpha)

    # Double edge: additive bright glow on dark side, subtractive shadow on white side
    # Additive/subtractive blending keeps fill image visible through the edge effect
    grain = erosion_noise(mask_gray.shape[0], mask_gray.shape[1], scale=60)
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_edge = cv2.subtract(cv2.dilate(mask_gray, thin_kernel), mask_gray).astype(np.float32) / 255.0
    dark_edge = cv2.subtract(mask_gray, cv2.erode(mask_gray, thin_kernel)).astype(np.float32) / 255.0

    edge_grain = np.clip(180 + grain * 35, 120, 230)
    bright_boost = bright_edge * edge_grain
    dark_boost = dark_edge * edge_grain

    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c] + bright_boost, 0, 255)
        result[:, :, c] = np.clip(result[:, :, c] - dark_boost, 0, 255)

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil halation-edge bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./halationedge-bot-output"))
    parser.add_argument("--frame-duration", type=int, default=50, help="GIF frame duration in ms")
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages
    from stencil_transform import make_stencil
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
        composite_arr = blend_with_noisy_mask(mask, images[a], images[b])
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
