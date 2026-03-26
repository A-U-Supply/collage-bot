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
    parser.add_argument("--channel", default=stencil.get("channel", "image-gen"))
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

    logger.info(f"Fetching 3 images from #{args.channel}...")
    source_paths = fetch_random_images(token, args.channel, 3, source_dir)

    img1 = Image.open(source_paths[0]).convert("RGB")
    img2 = Image.open(source_paths[1]).convert("RGB")
    img3 = Image.open(source_paths[2]).convert("RGB")

    logger.info("Building stencil from image 1...")
    mask = make_stencil(img1)

    logger.info("Compositing images 2 and 3 using stencil...")
    result = apply_stencil(mask, img2, img3)

    dest = out_dir / "stencil_result.png"
    result.save(dest)
    logger.info(f"Saved {dest.name}")

    if not args.no_post:
        post_collages(token, args.channel, [dest], bot_name="collage-stencil-bot")
        logger.info(f"Posted result to #{args.channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
