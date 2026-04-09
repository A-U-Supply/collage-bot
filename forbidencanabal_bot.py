"""Collage stencil forbidden cannibal bot.

Feeds on #img-junkyard's own outputs. Same stencil process as stencil_bot.py
but sources images from img-junkyard and posts results back there. No GIFs.
Posts the 3 binary stencil masks (from source images) alongside the 6 composites.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil forbidden cannibal bot")
    parser.add_argument("--source-channel", default="img-junkyard")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./forbidencanabal-bot-output"))
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages
    from stencil_transform import make_stencil, apply_stencil
    from PIL import Image

    source_dir = args.output_dir / "source"
    out_dir = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))

    images = [Image.open(p).convert("RGB") for p in source_paths]

    # Generate and save the 3 binary stencil masks from source images
    stencil_paths = []
    stencils = []
    for i, img in enumerate(images):
        mask = make_stencil(img)
        dest = out_dir / f"forbidencanabal_stencil_{i + 1}.png"
        mask.convert("RGB").save(dest)
        logger.info(f"Saved {dest.name}")
        stencil_paths.append(dest)
        stencils.append(mask)

    # Generate 6 composite variants
    result_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as stencil...")
        result = apply_stencil(stencils[s], images[a], images[b])
        dest = out_dir / f"forbidencanabal_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        result_paths.append(dest)

    post_paths = stencil_paths + result_paths

    if not args.no_post:
        post_collages(token, args.post_channel, post_paths, bot_name="collage-stencil-forbidencanabal-bot", threaded=False)
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
