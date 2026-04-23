"""Docker entry point for bullethole bot. Takes one image file, writes output.png."""
import argparse
import logging
from pathlib import Path

from PIL import Image

from bullethole_bot import apply_bullet_holes

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

parser = argparse.ArgumentParser(description="Apply bullet hole effect to an image")
parser.add_argument("input", help="Path to input image")
parser.add_argument("--output", default="/work/output/output.png",
                    help="Path to write output image")
parser.add_argument("--chaos", type=float, default=0.5,
                    help="0=few large holes, 1=many small holes")
parser.add_argument("--count", type=int, default=0,
                    help="Override hole count (0=derive from chaos)")
parser.add_argument("--size", type=float, default=0.0,
                    help="Override radius as fraction of image width (0=derive from chaos)")
parser.add_argument("--placement", default="random",
                    choices=["random", "grid", "hex", "radial"])
parser.add_argument("--effect", default="rotate",
                    choices=["rotate", "invert", "blur", "grayscale",
                             "pixelate", "fill_black", "fill_white"])
parser.add_argument("--jitter", type=float, default=0.0,
                    help="Randomness for structured placements (0-1)")
parser.add_argument("--no-overlap", action="store_true",
                    help="Prevent circles from overlapping")
args = parser.parse_args()

img = Image.open(args.input).convert("RGB")
result = apply_bullet_holes(
    img,
    chaos=args.chaos,
    count=args.count,
    size_frac=args.size,
    placement=args.placement,
    effect=args.effect,
    jitter=args.jitter,
    no_overlap=args.no_overlap,
)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
result.save(args.output)
