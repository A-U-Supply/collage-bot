"""Docker entry point for bullethole bot. Takes one image file, writes output image."""
import argparse
import json
import logging
import random
from pathlib import Path

from PIL import Image

from bullethole_bot import apply_bullet_holes

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

parser = argparse.ArgumentParser(description="Apply bullet hole effect to an image")
parser.add_argument("input", help="Path to input image")
parser.add_argument("--output", default="/work/output/output.jpg",
                    help="Path to write output image (extension overridden by --format)")
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
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducible results (0=auto-generate)")
parser.add_argument("--format", choices=["jpeg", "png"], default="jpeg",
                    help="Output format")
args = parser.parse_args()

seed = args.seed if args.seed != 0 else random.randint(1, 2**31 - 1)
random.seed(seed)
logging.info("Using seed %d", seed)

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

output_path = Path(args.output).with_suffix(".jpg" if args.format == "jpeg" else ".png")
output_path.parent.mkdir(parents=True, exist_ok=True)

if args.format == "jpeg":
    result.save(output_path, format="JPEG", quality=85)
else:
    result.save(output_path, format="PNG")

manifest_path = output_path.parent / "manifest.json"
manifest_path.write_text(json.dumps({"seed": seed}))
logging.info("Wrote %s (seed=%d)", output_path, seed)
