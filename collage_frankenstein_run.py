"""Docker entry point for collage frankenstein bot.

Reads exactly 9 source images via --input flags, runs the Latin-square
frankenstein assembly, and writes 9 collage outputs to --output-dir.
No Slack dependency — pure image processing.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from collage_frankenstein_bot import (
    OVERLAP,
    apply_source_style,
    assemble_mondrian,
    assemble_output,
    slice_quadrants,
    standardize,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

parser = argparse.ArgumentParser(description="Collage Frankenstein — Docker entry point")
parser.add_argument("--input", action="append", required=True,
                    help="Path to a source image. Pass 9 times.")
parser.add_argument("--output-dir", default="/work/output",
                    help="Directory to write output collages (default: /work/output)")
parser.add_argument("--size", type=int, default=900,
                    help="Square pixel dimension to standardize each source image to (default: 900)")
parser.add_argument("--style",
                    choices=["color", "rgb", "grayscale", "stencil", "sobel", "canny",
                             "invert", "posterize", "solarize"],
                    default="color")
parser.add_argument("--seams", choices=["edge", "smooth", "off"], default="edge")
parser.add_argument("--layout", choices=["grid", "mondrian"], default="grid")
parser.add_argument("--regions", type=int, default=9,
                    help="Number of regions for Mondrian partition (1–9, default: 9)")
parser.add_argument("--warp", action="store_true")
parser.add_argument("--warp-depth", type=int, default=80)
parser.add_argument("--warp-radius", type=int, default=30)
parser.add_argument("--warp-scale", type=float, default=1.0)
parser.add_argument("--warp-strip", type=int, default=20)
parser.add_argument("--refine", action="store_true")
parser.add_argument("--refine-steps", type=int, default=500)
args = parser.parse_args()

if len(args.input) != 9:
    print(f"Error: exactly 9 --input paths required, got {len(args.input)}", file=sys.stderr)
    sys.exit(1)

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng()
quad_size = args.size // 3

# Load, standardize, apply style
images = [standardize(Image.open(p).convert("RGB"), args.size) for p in args.input]
logging.info("Loaded and standardized %d source images to %d×%d", len(images), args.size, args.size)

if args.style != "color":
    images = [apply_source_style(img, args.style) for img in images]
    logging.info("Applied style '%s'", args.style)

quadrants = [slice_quadrants(img) for img in images]
logging.info("Sliced into %d quadrants (%d×%d each)", len(quadrants) * 9, quad_size, quad_size)

# Cross-tile normalisation stats for Canny
all_arrs = np.stack([np.array(q) for qs in quadrants for q in qs]).astype(np.float32)
canny_norm = (all_arrs.mean(axis=(0, 1, 2)), all_arrs.std(axis=(0, 1, 2)))

quilt = args.seams != "off"
seam_mode = "diff" if args.seams == "smooth" else "edge"

used_per_src: dict[int, set[int]] = {src_j: set() for src_j in range(9)}
output_paths = []

for out_i in range(9):
    pieces = [
        (src_j, [(qi, q) for qi, q in enumerate(quadrants[src_j])
                 if qi not in used_per_src[src_j]])
        for src_j in range(9)
    ]
    logging.info("Assembling output %d/9...", out_i + 1)

    if args.layout == "mondrian":
        result, chosen_qis = assemble_mondrian(
            pieces, rng, args.size, max(1, min(9, args.regions))
        )
    else:
        result, chosen_qis, _ = assemble_output(
            pieces, rng, quad_size, OVERLAP,
            quilt=quilt,
            warp=args.warp,
            warp_depth=args.warp_depth,
            warp_radius=args.warp_radius,
            warp_scale=args.warp_scale,
            warp_strip=args.warp_strip,
            anneal=args.refine,
            anneal_steps=args.refine_steps,
            seam_mode=seam_mode,
            canny_norm=canny_norm,
        )

    for src_j, qi in chosen_qis.items():
        used_per_src[src_j].add(qi)

    dest = out_dir / f"frankenstein_{out_i + 1:02d}.jpg"
    result.save(dest, format="JPEG", quality=92)
    output_paths.append(str(dest))
    logging.info("Saved %s", dest.name)

manifest_path = out_dir / "manifest.json"
manifest_path.write_text(json.dumps({"outputs": output_paths}))
logging.info("Done — wrote %d collages", len(output_paths))
