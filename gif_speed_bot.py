"""GIF speed bot — re-renders a collage GIF at a new frame duration.

Fetches the original stencil images from the thread root and re-renders
the GIF at the requested speed, posting it as a new reply in the thread.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

from PIL import Image
from slack_sdk import WebClient

from slack_fetcher import find_channel_id, _download_with_auth
from selector_fetcher import parse_message_link

logger = logging.getLogger(__name__)

SOURCE_MIME_PREFIXES = ("image/jpeg", "image/png", "image/webp")


def fetch_source_images(client: WebClient, token: str, channel_id: str, thread_ts: str, download_dir: Path) -> list[Path]:
    """Fetch the original PNG images from the stencil post (thread root)."""
    resp = client.conversations_replies(channel=channel_id, ts=thread_ts, limit=1)
    messages = resp.get("messages", [])
    if not messages:
        return []

    msg = messages[0]
    images = []
    seen_ids: set[str] = set()

    # Check files array
    for f in msg.get("files", []):
        fid = f.get("id", "")
        mimetype = f.get("mimetype", "")
        if fid in seen_ids:
            continue
        if any(mimetype.startswith(p) for p in SOURCE_MIME_PREFIXES):
            url = f.get("url_private_download") or f.get("url_private")
            if url:
                seen_ids.add(fid)
                images.append({"url": url, "ext": f.get("filetype", "png")})

    # Also check blocks — multi-file uploads put extra files here
    for block in msg.get("blocks", []):
        if block.get("type") != "file":
            continue
        file_id = block.get("file_id") or (block.get("file") or {}).get("id")
        if not file_id or file_id in seen_ids:
            continue
        info = client.files_info(file=file_id).get("file", {})
        mimetype = info.get("mimetype", "")
        if any(mimetype.startswith(p) for p in SOURCE_MIME_PREFIXES):
            url = info.get("url_private_download") or info.get("url_private")
            if url:
                seen_ids.add(file_id)
                images.append({"url": url, "ext": info.get("filetype", "png")})

    download_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        dest = download_dir / f"frame_{i:02d}.{img['ext']}"
        data = _download_with_auth(img["url"], token)
        dest.write_bytes(data)
        logger.info(f"Downloaded {dest.name}")
        paths.append(dest)
    return paths


def make_gif(image_paths: list[Path], output_path: Path, frame_duration_ms: int) -> None:
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
    logger.info(f"GIF saved to {output_path} ({len(frames)} frames @ {frame_duration_ms}ms)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="GIF speed bot")
    parser.add_argument("--channel", default="img-junkyard")
    parser.add_argument("--message-ts", help="Thread timestamp of the stencil post")
    parser.add_argument("--message-link", help="Slack message link to the stencil post")
    parser.add_argument("--frame-duration", type=int, required=True, help="Frame duration in ms")
    parser.add_argument("--output-dir", type=Path, default=Path("./gif-speed-output"))
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()

    if not args.message_ts and not args.message_link:
        print("Error: --message-ts or --message-link required", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    client = WebClient(token=token)

    if args.message_link:
        channel_id, thread_ts = parse_message_link(args.message_link)
    else:
        channel_name = args.channel.lstrip("#")
        channel_id = find_channel_id(client, channel_name)
        if not channel_id:
            raise ValueError(f"Channel #{channel_name} not found")
        thread_ts = args.message_ts

    frames_dir = args.output_dir / "frames"
    gif_path = args.output_dir / f"collage_stencil_{args.frame_duration}ms.gif"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching source images from thread {thread_ts}...")
    image_paths = fetch_source_images(client, token, channel_id, thread_ts, frames_dir)

    if not image_paths:
        print("Error: no source images found in thread", file=sys.stderr)
        sys.exit(1)

    logger.info(f"Re-rendering GIF at {args.frame_duration}ms per frame...")
    make_gif(image_paths, gif_path, frame_duration_ms=args.frame_duration)

    if not args.no_post:
        logger.info("Posting GIF to thread...")
        resp = client.files_upload_v2(
            channel=channel_id,
            file=str(gif_path),
            filename=f"collage_stencil_{args.frame_duration}ms.gif",
            title=f"collage stencil GIF @ {args.frame_duration}ms",
            thread_ts=thread_ts,
            initial_comment=f":scissors: *collage-stencil-gif-bot* — {args.frame_duration}ms/frame",
        )
        file_info = resp.get("file") or (resp.get("files") or [{}])[0]
        permalink = file_info.get("permalink", "")
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            reply_broadcast=True,
            text=f":scissors: *collage-stencil-gif-bot* — <{permalink}|view GIF> ({args.frame_duration}ms/frame)",
        )
        logger.info("Done!")
    else:
        logger.info(f"Saved to {gif_path} (--no-post)")


if __name__ == "__main__":
    main()
