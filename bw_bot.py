"""Collage stencil B&W bot.

Like collage-stencil-bot but converts all 3 images to high-contrast
black and white (via Otsu thresholding) before compositing. Posts all
6 variations plus an animated GIF as a single message.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def to_bw(img: Image.Image) -> Image.Image:
    """Convert image to high-contrast B&W using Otsu thresholding. Returns RGB."""
    from stencil_transform import make_stencil
    return make_stencil(img).convert("RGB")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil B&W bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./bw-bot-output"))
    parser.add_argument("--frame-duration", type=int, default=60, help="GIF frame duration in ms")
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
    bw_images = [to_bw(img) for img in images]
    logger.info("Converted all 3 images to B&W")

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as stencil...")
        mask = make_stencil(bw_images[s])
        result = apply_stencil(mask, bw_images[a], bw_images[b])
        dest = out_dir / f"bw_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    gif_path = out_dir / f"bw_stencil_{args.frame_duration}ms.gif"
    logger.info(f"Creating GIF at {args.frame_duration}ms/frame...")
    gif_order = [0, 3, 1, 4, 2, 5]
    make_gif([output_paths[i] for i in gif_order], gif_path, frame_duration_ms=args.frame_duration)

    gif_pair_12 = out_dir / f"bw_stencil_pair_12_{args.frame_duration}ms.gif"
    gif_pair_34 = out_dir / f"bw_stencil_pair_34_{args.frame_duration}ms.gif"
    gif_pair_56 = out_dir / f"bw_stencil_pair_56_{args.frame_duration}ms.gif"
    make_gif([output_paths[0], output_paths[1]], gif_pair_12, frame_duration_ms=args.frame_duration)
    make_gif([output_paths[2], output_paths[3]], gif_pair_34, frame_duration_ms=args.frame_duration)
    make_gif([output_paths[4], output_paths[5]], gif_pair_56, frame_duration_ms=args.frame_duration)

    post_paths = output_paths + [gif_path, gif_pair_12, gif_pair_34, gif_pair_56]

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths, bot_name="collage-stencil-bw-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
