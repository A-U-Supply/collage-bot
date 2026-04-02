"""Collage stencil GIF bot.

Takes the 6 images from a collage-stencil-bot post and creates an animated GIF,
posted as a thread reply (broadcast to channel) in #img-junkyard.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

from PIL import Image
from slack_sdk import WebClient

from slack_fetcher import find_channel_id, _download_with_auth

logger = logging.getLogger(__name__)

IMAGE_MIME_PREFIXES = ("image/jpeg", "image/png", "image/gif", "image/webp")


def fetch_thread_images(client: WebClient, token: str, channel_id: str, thread_ts: str, download_dir: Path) -> list[Path]:
    """Fetch all images from a thread in posting order."""
    images = []
    seen_ids: set[str] = set()
    cursor = None
    while True:
        kwargs = {"channel": channel_id, "ts": thread_ts, "limit": 200}
        if cursor:
            kwargs["cursor"] = cursor
        resp = client.conversations_replies(**kwargs)
        for msg in resp.get("messages", []):
            # Check files array
            for f in msg.get("files", []):
                fid = f.get("id", "")
                mimetype = f.get("mimetype", "")
                if fid in seen_ids:
                    continue
                if any(mimetype.startswith(p) for p in IMAGE_MIME_PREFIXES):
                    url = f.get("url_private_download") or f.get("url_private")
                    if url:
                        seen_ids.add(fid)
                        images.append({"url": url, "ext": f.get("filetype", "jpg")})
            # Also check blocks — multi-file uploads put extra files here
            for block in msg.get("blocks", []):
                if block.get("type") != "file":
                    continue
                file_id = block.get("file_id") or (block.get("file") or {}).get("id")
                if not file_id or file_id in seen_ids:
                    continue
                info = client.files_info(file=file_id).get("file", {})
                mimetype = info.get("mimetype", "")
                if any(mimetype.startswith(p) for p in IMAGE_MIME_PREFIXES):
                    url = info.get("url_private_download") or info.get("url_private")
                    if url:
                        seen_ids.add(file_id)
                        images.append({"url": url, "ext": info.get("filetype", "jpg")})
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    download_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        dest = download_dir / f"frame_{i:02d}.{img['ext']}"
        data = _download_with_auth(img["url"], token)
        dest.write_bytes(data)
        logger.info(f"Downloaded {dest.name}")
        paths.append(dest)
    return paths


def make_gif(image_paths: list[Path], output_path: Path, frame_duration_ms: int = 300) -> None:
    """Create a looping animated GIF from a list of image paths."""
    frames = [Image.open(p).convert("RGBA") for p in image_paths]
    w, h = frames[0].size
    frames = [f.resize((w, h), Image.LANCZOS) for f in frames]
    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=frame_duration_ms,
        disposal=2,
    )
    logger.info(f"GIF saved to {output_path} ({len(frames)} frames)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collage stencil GIF bot")
    parser.add_argument("--channel", default="img-junkyard")
    parser.add_argument("--message-ts", required=True, help="Timestamp of the stencil bot post")
    parser.add_argument("--frame-duration", type=int, default=100, help="Frame duration in ms")
    parser.add_argument("--output-dir", type=Path, default=Path("./collage-stencil-gif-output"))
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    client = WebClient(token=token)
    channel_name = args.channel.lstrip("#")
    channel_id = find_channel_id(client, channel_name)
    if not channel_id:
        raise ValueError(f"Channel #{channel_name} not found")

    frames_dir = args.output_dir / "frames"
    gif_path = args.output_dir / "collage_stencil.gif"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching images from thread {args.message_ts} in #{channel_name}...")
    image_paths = fetch_thread_images(client, token, channel_id, args.message_ts, frames_dir)

    if not image_paths:
        print("Error: no images found in thread", file=sys.stderr)
        sys.exit(1)

    logger.info(f"Creating GIF from {len(image_paths)} frames...")
    make_gif(image_paths, gif_path, frame_duration_ms=args.frame_duration)

    if not args.no_post:
        logger.info(f"Posting GIF as thread reply to #{channel_name}...")
        # Upload to thread
        resp = client.files_upload_v2(
            channel=channel_id,
            file=str(gif_path),
            filename="collage_stencil.gif",
            title="collage stencil GIF",
            thread_ts=args.message_ts,
            initial_comment=":scissors: *collage-stencil-gif-bot*",
        )
        # Broadcast to channel — files_upload_v2 doesn't support reply_broadcast
        # so we post a separate message in the thread with broadcast enabled
        file_info = resp.get("file") or (resp.get("files") or [{}])[0]
        permalink = file_info.get("permalink", "")
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=args.message_ts,
            reply_broadcast=True,
            text=f":scissors: *collage-stencil-gif-bot* — <{permalink}|view GIF>",
        )
        logger.info("Done!")
    else:
        logger.info(f"Saved to {gif_path} (--no-post)")


if __name__ == "__main__":
    main()
