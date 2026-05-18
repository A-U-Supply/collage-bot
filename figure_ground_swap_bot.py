"""Figure/ground swap collage bot.

Inspired by Clement Greenberg's essay on collage — figure/ground ambiguity
and material juxtaposition as aesthetic statement. Fetches 3 images from a
Slack channel, isolates the foreground subject in each using binary
segmentation, then produces all 6 fg/bg permutations: every foreground
on every other background. Hard binary mask edges, no feathering — the cut
line is the aesthetic statement. Posts all 6 outputs as a single Slack message.
"""
import argparse
import logging
import os
import sys
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def segment_grabcut(img_rgb: np.ndarray) -> np.ndarray:
    """GrabCut foreground segmentation with 10%-inset initialisation rect."""
    h, w = img_rgb.shape[:2]
    margin = int(min(h, w) * 0.10)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_rgb, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    return np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)


def segment_center(img_rgb: np.ndarray) -> np.ndarray:
    """Center-weighted Gaussian blob — centre is foreground, edges are background."""
    h, w = img_rgb.shape[:2]
    Y, X = np.ogrid[:h, :w]
    gauss = np.exp(
        -(
            (X - w / 2) ** 2 / (2 * (w * 0.35) ** 2)
            + (Y - h / 2) ** 2 / (2 * (h * 0.35) ** 2)
        )
    )
    return (gauss > 0.4).astype(np.uint8)


def segment_spectral(img_rgb: np.ndarray) -> np.ndarray:
    """Spectral Residual saliency (Hou & Zhang 2007) via numpy FFT."""
    from scipy.ndimage import gaussian_filter, uniform_filter

    gray = img_rgb.mean(axis=2)
    small = cv2.resize(gray.astype(np.float32), (64, 64)) / 255.0
    F = np.fft.fft2(small)
    log_amp = np.log(np.abs(F) + 1e-8)
    residual = log_amp - uniform_filter(log_amp, size=3)
    sal_small = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(F)))) ** 2
    sal = cv2.resize(sal_small.real, (gray.shape[1], gray.shape[0]))
    sal = gaussian_filter(sal, sigma=min(gray.shape) * 0.05)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return (sal > 0.4).astype(np.uint8)


def composite_hard(
    fg_img: np.ndarray, bg_img: np.ndarray, fg_mask: np.ndarray
) -> Image.Image:
    """Composite fg_img over bg_img using a hard binary mask. No feathering."""
    h, w = fg_img.shape[:2]
    bg = cv2.resize(bg_img, (w, h))
    m = np.stack([fg_mask] * 3, axis=2)
    result = fg_img * m + bg * (1 - m)
    return Image.fromarray(result.astype(np.uint8))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Figure/ground swap collage bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./figure-ground-output"))
    parser.add_argument(
        "--seg-method",
        choices=["grabcut", "center", "spectral"],
        default="grabcut",
        help="Foreground segmentation method (default: grabcut)",
    )
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
    paths = fetch_random_images(token, args.source_channel, 3, source_dir)

    imgs = [np.array(Image.open(p).convert("RGB")) for p in paths]

    seg_fn = {
        "grabcut": segment_grabcut,
        "center": segment_center,
        "spectral": segment_spectral,
    }[args.seg_method]

    logger.info(f"Segmenting with method: {args.seg_method}")
    masks = []
    for i, img in enumerate(imgs):
        logger.info(f"  Segmenting image {i + 1}/3...")
        masks.append(seg_fn(img))

    # All permutations: every fg on every other bg → 6 outputs
    output_paths = []
    for fg_i, bg_i in permutations(range(3), 2):
        logger.info(f"  Compositing: image {fg_i + 1} fg over image {bg_i + 1} bg...")
        out_img = composite_hard(imgs[fg_i], imgs[bg_i], masks[fg_i])
        out_path = out_dir / f"figure_ground_{fg_i + 1}_on_{bg_i + 1}.jpg"
        out_img.save(out_path, quality=92)
        logger.info(f"Saved {out_path.name}")
        output_paths.append(out_path)

    if not args.no_post:
        post_collages(
            token,
            args.post_channel,
            output_paths,
            bot_name="figure-ground-swap",
            threaded=False,
        )
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
