"""Collage stencil silver bot.

Like collage-stencil-monochrome-bot but uses a silver gelatin tonal curve
plus halation (highlight bloom) for a luminous, glowing print effect.
The stencil image remains full color for mask generation.
Posts all 6 variations plus an animated GIF as a single message.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Tonal curve: crushed blacks, rich midtones, pushed highlights toward white
# Defined as (input, output) points, interpolated across 0-255
_CURVE_IN  = [  0,  30, 128, 210, 255]
_CURVE_OUT = [  0,   5, 140, 248, 255]
_LUT = np.interp(np.arange(256), _CURVE_IN, _CURVE_OUT).astype(np.uint8)


def to_silver_halation(img: Image.Image) -> Image.Image:
    """Apply tonal curve + halation + film grain + silver tone.

    1. Convert to grayscale
    2. Apply tonal curve (crush blacks, lift midtones, bloom highlights)
    3. Halation: blur highlights and screen-blend back for halo glow
    4. Film grain: add gaussian noise for organic emulsion texture
    5. Silver tone: shift whites to cool blue-grey like an aged silver print
    """
    gray = np.array(img.convert("L"))

    # Apply tonal curve via LUT
    curved = _LUT[gray]

    # Halation: lower threshold to bloom more of the image, wider blur
    highlights = np.where(curved > 170, curved.astype(np.float32), 0.0)
    halo = cv2.GaussianBlur(highlights, (0, 0), sigmaX=26)

    # Screen blend: result = 255 - ((255-a)*(255-b)/255)
    a = curved.astype(np.float32)
    result = 255.0 - (255.0 - a) * (255.0 - halo) / 255.0
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Film grain: slightly more grain for micro-contrast and texture
    grain = np.random.normal(0, 9, result.shape).astype(np.float32)
    result = np.clip(result.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    # Unsharp mask to crisp edges against the glow
    pil_sharp = Image.fromarray(result, mode="L")
    pil_sharp = pil_sharp.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=2))
    result = np.array(pil_sharp)

    # Silver tone: convert to RGB and tint whites with a cool blue-grey
    pil = Image.fromarray(result, mode="L").convert("RGB")
    r, g, b = pil.split()
    r_arr = np.array(r, dtype=np.float32)
    g_arr = np.array(g, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)

    # Shift highlights: reduce red slightly, boost blue slightly
    r_arr = np.clip(r_arr * 0.95, 0, 255)
    b_arr = np.clip(b_arr * 1.06, 0, 255)

    return Image.merge("RGB", (
        Image.fromarray(r_arr.astype(np.uint8)),
        Image.fromarray(g_arr.astype(np.uint8)),
        Image.fromarray(b_arr.astype(np.uint8)),
    ))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil silver bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./silver-bot-output"))
    parser.add_argument("--frame-duration", type=int, default=85, help="GIF frame duration in ms")
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

    # Pre-compute silver halation version of each image for use as fill
    silver_images = [to_silver_halation(img) for img in images]
    logger.info("Applied silver halation to all 3 images")

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as stencil, {a + 1} and {b + 1} as silver fill...")
        mask = make_stencil(images[s])
        result = apply_stencil(mask, silver_images[a], silver_images[b])
        dest = out_dir / f"silver_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    gif_path = out_dir / f"silver_stencil_{args.frame_duration}ms.gif"
    logger.info(f"Creating GIF at {args.frame_duration}ms/frame...")
    gif_order = [0, 3, 1, 4, 2, 5]
    make_gif([output_paths[i] for i in gif_order], gif_path, frame_duration_ms=args.frame_duration)

    gif_pair_12 = out_dir / f"silver_stencil_pair_12_{args.frame_duration}ms.gif"
    gif_pair_34 = out_dir / f"silver_stencil_pair_34_{args.frame_duration}ms.gif"
    gif_pair_56 = out_dir / f"silver_stencil_pair_56_{args.frame_duration}ms.gif"
    make_gif([output_paths[0], output_paths[1]], gif_pair_12, frame_duration_ms=args.frame_duration)
    make_gif([output_paths[2], output_paths[3]], gif_pair_34, frame_duration_ms=args.frame_duration)
    make_gif([output_paths[4], output_paths[5]], gif_pair_56, frame_duration_ms=args.frame_duration)

    post_paths = output_paths + [gif_path, gif_pair_12, gif_pair_34, gif_pair_56]

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths, bot_name="collage-stencil-silver-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
