"""Collage bot — fetches images from #image-gen, transforms, posts back."""
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
    slack = cfg.get("slack", {})
    collage = cfg.get("collage", {})
    transform = cfg.get("transform", {})

    parser = argparse.ArgumentParser(description="Collage bot")
    parser.add_argument("--source-channel", default=slack.get("source_channel", "image-gen"))
    parser.add_argument("--post-channel", default=slack.get("post_channel", "collage-repository"))
    parser.add_argument("--num-images", type=int, default=collage.get("num_images", 4))
    parser.add_argument("--output-dir", type=Path, default=collage.get("output_dir", "./collage-bot-output"))
    parser.add_argument("--split", type=float, default=transform.get("split", 0.25))
    parser.add_argument("--blend-width", type=int, default=transform.get("blend_width", 70))
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
    from transform import make_composites, apply_transform, blend_seams
    from PIL import Image

    output_dir = Path(args.output_dir)
    source_dir = output_dir / "source"
    out_dir = output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching {args.num_images} images from #{args.source_channel}...")
    source_paths = fetch_random_images(token, args.source_channel, args.num_images, source_dir)

    logger.info("Building composites from quadrants...")
    source_images = [Image.open(p).convert("RGB") for p in source_paths]
    composites = make_composites(source_images)

    output_paths = []
    for i, composite in enumerate(composites):
        transformed = apply_transform(composite, split=args.split)
        blended = blend_seams(transformed, strip_width=args.blend_width, split=args.split)
        dest = out_dir / f"collage_{i + 1}.png"
        blended.save(dest)
        logger.info(f"Saved {dest.name}")
        output_paths.append(dest)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths)
        logger.info(f"Posted {len(output_paths)} collages to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
