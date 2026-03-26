"""Collage stencil bot — uses image 1 as a binary mask to composite images 2 and 3."""
import argparse
import logging
import os
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.toml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    stencil = cfg.get("stencil", {})

    parser = argparse.ArgumentParser(description="Collage stencil bot")
    parser.add_argument("--source-channel", default=stencil.get("source_channel", "image-gen"))
    parser.add_argument("--post-channel", default=stencil.get("post_channel", "collage-repository"))
    parser.add_argument("--output-dir", type=Path, default=stencil.get("output_dir", "./collage-stencil-bot-output"))
    parser.add_argument("--no-post", action="store_true")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_config()
    args = build_parser(cfg).parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages
    from stencil_transform import make_stencil, apply_stencil
    from PIL import Image

    output_dir = Path(args.output_dir)
    source_dir = output_dir / "source"
    out_dir = output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = fetch_random_images(token, args.source_channel, 3, source_dir)

    images = [Image.open(p).convert("RGB") for p in source_paths]

    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (1, 0, 2), (2, 0, 1)]):
        logger.info(f"Version {i + 1}: image {s + 1} as stencil...")
        mask = make_stencil(images[s])
        result = apply_stencil(mask, images[a], images[b])
        dest = out_dir / f"stencil_result_{i + 1}.png"
        result.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths, bot_name="collage-stencil-bot")
        logger.info(f"Posted {len(output_paths)} results to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
