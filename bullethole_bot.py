"""Collage stencil bullet hole bot.

Fetches one image from Slack and punches circular "bullet holes" into it.
A --chaos knob (0.0–1.0) controls both count and size inversely:
  0.0 = 2–3 large circles (up to w/3 diameter)
  0.5 = 5–7 medium circles
  1.0 = 13–15 small circles (down to w/12 diameter)
Each circle is cut from a random position, an effect is applied inside it,
and composited back in place. Posts the result to img-junkyard.
"""
import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def chaos_params(w: int, h: int, chaos: float) -> tuple:
    """Derive hole count and radius range from chaos value and image size.

    chaos=0.0 → 2–3 large holes (~33% of image width)
    chaos=1.0 → 85–87 small holes (~5% of image width)
    """
    # Count: 2 at chaos=0, 86 at chaos=1, with ±1 jitter
    n = max(2, round(2 + chaos * 84 + random.uniform(-1, 1)))

    # Radius as fraction of width: lerp 0.33 → 0.05
    frac = 0.33 + (0.05 - 0.33) * chaos
    max_r = max(4, round(w * frac))
    min_r = max(2, round(max_r * 0.6))

    return n, min_r, max_r


def _apply_effect(crop: Image.Image, effect: str, radius: int) -> Image.Image:
    """Apply the chosen effect to a circular crop and return the result."""
    if effect == "invert":
        from PIL import ImageChops
        return ImageChops.invert(crop.convert("RGB"))
    elif effect == "blur":
        from PIL import ImageFilter
        return crop.filter(ImageFilter.GaussianBlur(radius=max(2, radius // 6)))
    elif effect == "grayscale":
        return crop.convert("L").convert("RGB")
    elif effect == "pixelate":
        block = max(4, radius // 8)
        small = crop.resize(
            (max(1, crop.width // block), max(1, crop.height // block)),
            Image.NEAREST,
        )
        return small.resize(crop.size, Image.NEAREST)
    elif effect == "fill_black":
        return Image.new("RGB", crop.size, (0, 0, 0))
    elif effect == "fill_white":
        return Image.new("RGB", crop.size, (255, 255, 255))
    else:  # "rotate" (default)
        angle = random.uniform(30, 330)
        return crop.rotate(angle, resample=Image.BICUBIC)


def _remove_overlapping(centers: list, radius: int) -> list:
    """Filter centers so no two circles overlap."""
    kept = []
    diam_sq = (radius * 2) ** 2
    for cx, cy in centers:
        if not any((cx - ox) ** 2 + (cy - oy) ** 2 < diam_sq for ox, oy in kept):
            kept.append((cx, cy))
    return kept


def _place_centers_random(n: int, w: int, h: int, radius: int,
                           no_overlap: bool) -> list:
    """Scatter centers uniformly at random."""
    centers = []
    diam_sq = (radius * 2) ** 2
    for _ in range(n):
        if no_overlap:
            for _ in range(100):
                cx = random.randint(radius, w - radius)
                cy = random.randint(radius, h - radius)
                if not any((cx - ox) ** 2 + (cy - oy) ** 2 < diam_sq
                           for ox, oy in centers):
                    centers.append((cx, cy))
                    break
        else:
            cx = random.randint(radius, w - radius)
            cy = random.randint(radius, h - radius)
            centers.append((cx, cy))
    return centers


def _place_centers_grid(n: int, w: int, h: int, radius: int,
                        jitter: float, hex_offset: bool = False) -> list:
    """Place centers on a rectangular (or hex-offset) grid."""
    cols = max(1, round(math.sqrt(n * w / h)))
    rows = max(1, math.ceil(n / cols))

    cell_w = w / cols
    cell_h = h / rows

    centers = []
    for row in range(rows):
        for col in range(cols):
            if len(centers) >= n:
                break
            cx = cell_w * (col + 0.5)
            cy = cell_h * (row + 0.5)
            if hex_offset and row % 2 == 1:
                cx += cell_w * 0.5
            if jitter > 0:
                cx += random.uniform(-1, 1) * cell_w * jitter * 0.5
                cy += random.uniform(-1, 1) * cell_h * jitter * 0.5
            cx = max(radius, min(w - radius, int(round(cx))))
            cy = max(radius, min(h - radius, int(round(cy))))
            centers.append((cx, cy))
    return centers


def _place_centers_radial(n: int, w: int, h: int, radius: int,
                           jitter: float) -> list:
    """Place centers on concentric rings around the image center."""
    cx0, cy0 = w / 2, h / 2
    max_r_dist = min(cx0, cy0) - radius

    if max_r_dist <= 0 or n <= 1:
        return [(int(cx0), int(cy0))] * n

    num_rings = max(1, round(math.sqrt(n)))
    ring_radii = [max_r_dist * (i + 1) / num_rings for i in range(num_rings)]

    total_r = sum(ring_radii)
    counts = []
    remaining = n
    for i, r in enumerate(ring_radii):
        if i == num_rings - 1:
            counts.append(remaining)
        else:
            c = max(1, round(n * r / total_r))
            counts.append(c)
            remaining -= c

    centers = []
    for ring_r, ring_n in zip(ring_radii, counts):
        for k in range(ring_n):
            angle = 2 * math.pi * k / ring_n
            if jitter > 0:
                angle += random.uniform(-1, 1) * math.pi / ring_n * jitter
                r_jit = ring_r + random.uniform(-1, 1) * (max_r_dist / num_rings) * jitter * 0.5
                r_jit = max(radius, min(max_r_dist, r_jit))
            else:
                r_jit = ring_r
            px = int(round(cx0 + r_jit * math.cos(angle)))
            py = int(round(cy0 + r_jit * math.sin(angle)))
            px = max(radius, min(w - radius, px))
            py = max(radius, min(h - radius, py))
            centers.append((px, py))

    return centers[:n]


def apply_bullet_holes(img: Image.Image, chaos: float = 0.5,
                       count: int = 0, size_frac: float = 0.0,
                       placement: str = "random", jitter: float = 0.0,
                       effect: str = "rotate", no_overlap: bool = False) -> Image.Image:
    """Cut circular sections, apply effect, paste back.

    count:      if > 0, overrides the chaos-derived hole count
    size_frac:  if > 0, overrides the chaos-derived radius as a fraction of
                image width (e.g. 0.15 = 15% of width)
    placement:  'random' | 'grid' | 'hex' | 'radial'
    jitter:     0.0–1.0, adds random offset to structured placements
    effect:     'rotate' | 'invert' | 'blur' | 'grayscale' | 'pixelate' |
                'fill_black' | 'fill_white'
    no_overlap: if True, prevent circles from overlapping
    """
    w, h = img.size
    n, min_r, max_r = chaos_params(w, h, chaos)

    if count > 0:
        n = count
    if size_frac > 0:
        min_r = max_r = max(2, round(w * size_frac))

    logger.info(
        f"chaos={chaos:.2f} → {n} holes, radius {min_r}–{max_r}px, "
        f"placement={placement}, effect={effect}, no_overlap={no_overlap}"
    )

    result = img.copy()

    # All holes the same size — pick once per run
    radius = random.randint(min_r, max_r)
    logger.info(f"radius={radius}px for all holes")

    if placement == "grid":
        centers = _place_centers_grid(n, w, h, radius, jitter, hex_offset=False)
        if no_overlap:
            centers = _remove_overlapping(centers, radius)
    elif placement == "hex":
        centers = _place_centers_grid(n, w, h, radius, jitter, hex_offset=True)
        if no_overlap:
            centers = _remove_overlapping(centers, radius)
    elif placement == "radial":
        centers = _place_centers_radial(n, w, h, radius, jitter)
        if no_overlap:
            centers = _remove_overlapping(centers, radius)
    else:
        centers = _place_centers_random(n, w, h, radius, no_overlap)

    for cx, cy in centers:
        left = cx - radius
        top = cy - radius
        size = radius * 2

        crop = result.crop((left, top, left + size, top + size))
        processed = _apply_effect(crop.convert("RGB"), effect, radius)

        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size - 1, size - 1), fill=255)

        result.paste(processed, (left, top), mask)

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil bullet hole bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./bullethole-bot-output"))
    parser.add_argument("--chaos", type=float, default=0.5,
                        help="0.0 = few large circles, 1.0 = many small circles")
    parser.add_argument("--count", type=int, default=0,
                        help="Number of circles (0 = derive from chaos)")
    parser.add_argument("--size", type=float, default=0.0,
                        help="Circle radius as fraction of image width, e.g. 0.15 (0 = derive from chaos)")
    parser.add_argument("--placement", default="random",
                        choices=["random", "grid", "hex", "radial"],
                        help="Circle placement mode")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="Randomness added to structured placements (0.0–1.0)")
    parser.add_argument("--effect", default="rotate",
                        choices=["rotate", "invert", "blur", "grayscale",
                                 "pixelate", "fill_black", "fill_white"],
                        help="Effect applied inside each circle")
    parser.add_argument("--no-overlap", action="store_true",
                        help="Prevent circles from overlapping")
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images
    from slack_poster import post_collages
    from stencil_transform import make_stencil

    source_dir = args.output_dir / "source"
    out_dir = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching 1 image from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 1, source_dir))
    img = Image.open(source_paths[0]).convert("RGB")

    result = apply_bullet_holes(
        img,
        chaos=args.chaos,
        count=args.count,
        size_frac=args.size,
        placement=args.placement,
        jitter=args.jitter,
        effect=args.effect,
        no_overlap=args.no_overlap,
    )

    dest = out_dir / "bullethole_result.png"
    result.save(dest)
    logger.info(f"Saved {dest.name}")

    binary = make_stencil(result).convert("RGB")
    dest_binary = out_dir / "bullethole_binary.png"
    binary.save(dest_binary)
    logger.info(f"Saved {dest_binary.name}")

    if not args.no_post:
        post_collages(token, args.post_channel, [dest, dest_binary], bot_name="collage-stencil-bullethole-bot", threaded=False)
        logger.info(f"Posted to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
