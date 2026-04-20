"""Typographer ASCII art bot.

Fetches 3 images from Slack, converts each to ASCII art using the
ascii-typographer CLI (https://github.com/user-simon/typographer),
renders the text output to a PNG image, and posts all 3 to Slack.
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_FONT_PATHS = [
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",  # Ubuntu CI
    "/System/Library/Fonts/Menlo.ttc",                                   # macOS
    "/System/Library/Fonts/Monaco.ttf",                                  # macOS fallback
]


def _load_mono_font(font_size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size=font_size)
    logger.warning("No monospace font found, using PIL default")
    return ImageFont.load_default()


def convert_to_ascii(input_path: Path, txt_path: Path, ascii_width: int) -> None:
    """Run typographer CLI to convert image to ASCII text file."""
    typographer = shutil.which("typographer")
    if not typographer:
        # Try ~/.cargo/bin directly
        cargo_bin = Path.home() / ".cargo" / "bin" / "typographer"
        if cargo_bin.exists():
            typographer = str(cargo_bin)
        else:
            raise RuntimeError(
                "typographer binary not found. Install with: cargo install ascii-typographer"
            )

    cmd = [typographer, str(input_path), "--save", str(txt_path), "--width", str(ascii_width)]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"typographer failed:\n{result.stderr}")


def render_ascii_to_image(txt_path: Path, font_size: int = 9) -> Image.Image:
    """Render a plain-text ASCII art file to a PIL RGB image."""
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty ASCII output from {txt_path}")

    font = _load_mono_font(font_size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = probe.textbbox((0, 0), "M", font=font)
    char_w = max(1, bbox[2] - bbox[0])
    char_h = max(1, bbox[3] - bbox[1] + 1)

    cols = max(len(l) for l in lines)
    rows = len(lines)
    logger.info(f"Rendering {cols}×{rows} char grid at {char_w}×{char_h}px/cell")

    img = Image.new("RGB", (cols * char_w, rows * char_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for r, line in enumerate(lines):
        draw.text((0, r * char_h), line, fill=(255, 255, 255), font=font)
    return img


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Typographer ASCII art bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./typographer-bot-output"))
    parser.add_argument("--ascii-width", type=int, default=120,
                        help="Characters per row (wider = more detail)")
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages

    source_dir = args.output_dir / "source"
    txt_dir = args.output_dir / "text"
    out_dir = args.output_dir / "output"
    txt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))

    output_paths = []
    for i, src_path in enumerate(source_paths):
        logger.info(f"Converting image {i + 1}/3 (ascii_width={args.ascii_width})...")
        txt_path = txt_dir / f"typographer_{i + 1}.txt"
        png_path = out_dir / f"typographer_{i + 1}.png"

        convert_to_ascii(src_path, txt_path, args.ascii_width)
        img = render_ascii_to_image(txt_path)
        img.save(png_path)
        logger.info(f"Saved {png_path.name} ({img.width}×{img.height}px)")
        output_paths.append(png_path)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="typographer-ascii-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} images to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
