"""Collage stencil quad bot.

Fetches 4 images from #image-gen. Each image takes a turn as a 3-level
stencil (black/grey/white), with all permutations of the remaining 3 images
filling those regions. Generates 24 variations, posts them in batches of 10,
then posts an animated GIF of all variations as the final thread reply.
"""
import argparse
import logging
import os
import sys
import time
from itertools import permutations
from pathlib import Path

from PIL import Image
from slack_sdk import WebClient

from slack_fetcher import fetch_random_images, find_channel_id
from quad_transform import make_3level_stencil, apply_3level_stencil
from gif_bot import make_gif

logger = logging.getLogger(__name__)

BOT_NAME = "collage-stencil-quad-bot"


def post_batch(client: WebClient, channel_id: str, paths: list[Path], thread_ts: str, comment: str | None = None) -> None:
    file_uploads = [
        {"file": str(p), "filename": p.name, "title": p.stem}
        for p in paths
    ]
    kwargs: dict = dict(channel=channel_id, file_uploads=file_uploads, thread_ts=thread_ts)
    if comment:
        kwargs["initial_comment"] = comment
    client.files_upload_v2(**kwargs)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil quad bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel", default="img-junkyard")
    parser.add_argument("--frame-duration", type=int, default=100, help="GIF frame duration in ms")
    parser.add_argument("--output-dir", type=Path, default=Path("./quad-bot-output"))
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    client = WebClient(token=token)
    channel_name = args.post_channel.lstrip("#")
    channel_id = find_channel_id(client, channel_name)
    if not channel_id:
        raise ValueError(f"Channel #{channel_name} not found")

    source_dir = args.output_dir / "source"
    out_dir = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fetch 4 source images
    logger.info(f"Fetching 4 images from #{args.source_channel}...")
    source_paths = fetch_random_images(token, args.source_channel, 4, source_dir)
    images = [Image.open(p).convert("RGB") for p in source_paths]

    # Generate all 24 variations (4 stencil choices × 3! orderings)
    # Also save the 3-level mask image for each original
    output_paths = []
    mask_paths = []
    n = 0
    for stencil_idx in range(4):
        logger.info(f"Generating variations with image {stencil_idx + 1} as stencil...")
        mask = make_3level_stencil(images[stencil_idx])
        mask_dest = out_dir / f"mask_{stencil_idx:02d}.png"
        mask.save(mask_dest)
        mask_paths.append(mask_dest)
        others = [images[i] for i in range(4) if i != stencil_idx]
        for perm in permutations(range(3)):
            ordered = [others[i] for i in perm]
            result = apply_3level_stencil(mask, ordered[0], ordered[1], ordered[2])
            dest = out_dir / f"quad_{n:02d}_s{stencil_idx}.png"
            result.save(dest)
            logger.info(f"Saved {dest.name}")
            output_paths.append(dest)
            n += 1

    # Append mask images after variations, before GIF
    all_image_paths = output_paths + mask_paths

    # Create GIF from all 24 variation outputs (not masks)
    gif_path = args.output_dir / f"quad_stencil_{args.frame_duration}ms.gif"
    logger.info(f"Creating GIF at {args.frame_duration}ms/frame...")
    make_gif(output_paths, gif_path, frame_duration_ms=args.frame_duration)

    if args.no_post:
        logger.info(f"Saved {len(all_image_paths)} images + GIF to {args.output_dir} (--no-post)")
        return

    # Post to Slack
    # Open the thread with a text message so we have a reliable ts
    msg = client.chat_postMessage(
        channel=channel_id,
        text=f":scissors: *{BOT_NAME}* — {len(output_paths)} variations",
    )
    thread_ts = msg["ts"]

    # Post all images (variations + masks) in batches of 10
    batches = [all_image_paths[i:i + 10] for i in range(0, len(all_image_paths), 10)]
    for i, batch in enumerate(batches):
        logger.info(f"Posting batch {i + 1}/{len(batches)} ({len(batch)} images)...")
        post_batch(
            client, channel_id, batch, thread_ts=thread_ts,
            comment=f"Images {i * 10 + 1}–{i * 10 + len(batch)}",
        )
        time.sleep(2)

    # Post GIF as final thread reply
    logger.info("Posting GIF...")
    client.files_upload_v2(
        channel=channel_id,
        file=str(gif_path),
        filename=gif_path.name,
        title=f"quad stencil GIF @ {args.frame_duration}ms",
        thread_ts=thread_ts,
        initial_comment=f":scissors: GIF ({args.frame_duration}ms/frame)",
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
