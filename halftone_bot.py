"""Collage stencil halftone bot.

Like collage-stencil-bot but converts the stencil image to an AM halftone
dot field before thresholding to binary black and white. The fill images
are full color originals. The halftone dot structure follows the tonal
map of the stencil image — bright areas produce dense white dots (fill_a),
dark areas produce dense black dots (fill_b).
Posts all 6 variations plus the 3 halftone stencil masks as a single message.
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

# S-curve: steepens midtone contrast, pushes shadows darker and highlights
# brighter while keeping a smooth gradient through the midtone range
_SCURVE_IN  = [  0,  64, 128, 192, 255]
_SCURVE_OUT = [  0,  40, 128, 215, 255]
_SCURVE_LUT = np.interp(np.arange(256), _SCURVE_IN, _SCURVE_OUT).astype(np.uint8)


def preprocess_for_halftone(img: Image.Image) -> np.ndarray:
    """Boost contrast before halftone screening without losing gradient.

    1. CLAHE — enhances local contrast, brings out shadow and highlight detail
    2. S-curve — global push: shadows darker, highlights brighter, midtone
       gradient preserved
    Returns a grayscale float32 array in [0, 1].
    """
    gray = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    curved = _SCURVE_LUT[enhanced]
    return curved.astype(np.float32) / 255.0


def make_halftone_stencil(img: Image.Image, frequency: int = 8, angle: float = 45.0) -> Image.Image:
    """Convert image to AM halftone binary mask.

    1. Preprocess: CLAHE + S-curve for confident dots without losing gradient
    2. AM halftone screen: cosine dot grid at given frequency and angle
    3. Compare tone to screen value → binary dot field

    Bright areas produce white dots (fill_a), dark areas produce black
    dots (fill_b), midtones produce mixed dot fields.
    """
    gray = preprocess_for_halftone(img)
    h, w = gray.shape

    angle_rad = np.radians(angle)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    xr = x * np.cos(angle_rad) + y * np.sin(angle_rad)
    yr = -x * np.sin(angle_rad) + y * np.cos(angle_rad)

    screen = (np.cos(2 * np.pi * xr / frequency) *
              np.cos(2 * np.pi * yr / frequency) + 1) / 2

    binary = (gray > screen).astype(np.uint8) * 255
    return Image.fromarray(binary)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil halftone bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./halftone-bot-output"))
    parser.add_argument("--frequency", type=int, default=14, help="Halftone dot frequency in pixels")
    parser.add_argument("--angle", type=float, default=45.0, help="Halftone screen angle in degrees")
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
    source_paths = fetch_random_images(token, args.source_channel, 3, source_dir)
    images = [Image.open(p).convert("RGB") for p in source_paths]

    # Pre-compute and save halftone masks for all 3 stencil images
    masks = []
    mask_paths = []
    for i, img in enumerate(images):
        mask = make_halftone_stencil(img, frequency=args.frequency, angle=args.angle)
        mask_rgb = mask.convert("RGB")
        dest = out_dir / f"halftone_mask_{i + 1}.png"
        mask_rgb.save(dest)
        logger.info(f"Saved {dest.name}")
        masks.append(mask)
        mask_paths.append(dest)

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as halftone stencil, {a + 1} and {b + 1} as fill...")
        result = apply_stencil(masks[s], images[a], images[b])
        dest = out_dir / f"halftone_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    post_paths = output_paths + mask_paths

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths, bot_name="collage-stencil-halftone-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
