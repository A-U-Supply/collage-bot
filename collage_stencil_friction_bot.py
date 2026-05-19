"""Collage stencil friction bot.

Fetches a larger pool of images, selects the 3 with the highest pairwise
visual friction (tonal contrast + texture energy + colour distance), then
runs the standard Otsu stencil compositing on those 3 images.

"Friction" is drawn from Greenberg's essay on collage: the productive tension
between juxtaposed surfaces with different visual weights, textures, and spatial
readings. This bot maximises that tension at the selection stage rather than
leaving it to chance.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_POOL = 12


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil friction bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./collage-stencil-friction-output"))
    parser.add_argument(
        "--pool-size", type=int, default=_DEFAULT_POOL,
        help=f"Number of images to fetch before friction selection (default: {_DEFAULT_POOL})",
    )
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages
    from stencil_transform import make_stencil, apply_stencil
    from friction_selector import select_friction_triplet
    from PIL import Image

    source_dir = args.output_dir / "source"
    out_dir = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching {args.pool_size} images from #{args.source_channel}...")
    pool_paths = list(fetch_random_images(token, args.source_channel, args.pool_size, source_dir))

    logger.info("Selecting highest-friction triplet...")
    selected_paths = select_friction_triplet(pool_paths)
    for i, p in enumerate(selected_paths):
        logger.info(f"  Selected image {i + 1}: {p.name}")

    images = [Image.open(p).convert("RGB") for p in selected_paths]

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        logger.info(f"Version {i + 1}: image {s + 1} as stencil...")
        mask = make_stencil(images[s])
        result = apply_stencil(mask, images[a], images[b])
        dest = out_dir / f"friction_stencil_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    from gif_bot import make_gif
    gif_path = out_dir / "friction_stencil.gif"
    logger.info("Creating GIF...")
    gif_order = [0, 3, 1, 4, 2, 5]
    make_gif([output_paths[i] for i in gif_order], gif_path, frame_duration_ms=100)

    gif_pair_12 = out_dir / "friction_stencil_pair_12.gif"
    gif_pair_34 = out_dir / "friction_stencil_pair_34.gif"
    gif_pair_56 = out_dir / "friction_stencil_pair_56.gif"
    make_gif([output_paths[0], output_paths[1]], gif_pair_12, frame_duration_ms=100)
    make_gif([output_paths[2], output_paths[3]], gif_pair_34, frame_duration_ms=100)
    make_gif([output_paths[4], output_paths[5]], gif_pair_56, frame_duration_ms=100)

    post_paths = output_paths + [gif_path, gif_pair_12, gif_pair_34, gif_pair_56]

    if not args.no_post:
        post_collages(
            token, args.post_channel, post_paths,
            bot_name="collage-stencil-friction-bot",
            threaded=False,
        )
        logger.info(f"Posted {len(post_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
