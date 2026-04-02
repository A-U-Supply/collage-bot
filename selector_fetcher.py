"""Fetch images from specific Slack message links and threads."""
import logging
import re
from pathlib import Path
from typing import Optional

from slack_sdk import WebClient

from slack_fetcher import fetch_random_images, _download_with_auth

logger = logging.getLogger(__name__)

IMAGE_MIME_PREFIXES = ("image/jpeg", "image/png", "image/gif", "image/webp")


def parse_message_link(link: str) -> tuple[str, str]:
    """Parse a Slack message link into (channel_id, timestamp).

    Link format: https://workspace.slack.com/archives/CHANNEL_ID/pTIMESTAMP
    The timestamp in the URL is microseconds with no decimal point.
    """
    match = re.search(r'/archives/([A-Z0-9]+)/p(\d+)', link)
    if not match:
        raise ValueError(f"Invalid Slack message link: {link}")
    channel_id = match.group(1)
    ts_raw = match.group(2)
    ts = ts_raw[:-6] + "." + ts_raw[-6:]
    return channel_id, ts


def extract_images_from_message(msg: dict) -> list[dict]:
    """Extract image file metadata from a Slack message dict."""
    images = []
    for f in msg.get("files", []):
        mimetype = f.get("mimetype", "")
        if any(mimetype.startswith(p) for p in IMAGE_MIME_PREFIXES):
            url = f.get("url_private_download") or f.get("url_private")
            if url:
                images.append({"url": url, "ext": f.get("filetype", "jpg"), "id": f["id"]})
    return images


def fetch_message(client: WebClient, channel_id: str, ts: str) -> Optional[dict]:
    """Fetch a single message by channel and timestamp."""
    resp = client.conversations_history(
        channel=channel_id,
        latest=ts,
        oldest=ts,
        inclusive=True,
        limit=1,
    )
    messages = resp.get("messages", [])
    return messages[0] if messages else None


def fetch_thread_images(client: WebClient, channel_id: str, thread_ts: str, exclude_ts: str) -> list[dict]:
    """Fetch all images from a thread, excluding the message at exclude_ts."""
    images = []
    cursor = None
    while True:
        kwargs = {"channel": channel_id, "ts": thread_ts, "limit": 200}
        if cursor:
            kwargs["cursor"] = cursor
        resp = client.conversations_replies(**kwargs)
        for msg in resp.get("messages", []):
            if msg["ts"] == exclude_ts:
                continue
            images.extend(extract_images_from_message(msg))
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return images


def gather_images_for_reaction(
    client: WebClient,
    token: str,
    channel_id: str,
    ts: str,
    thread_ts: Optional[str],
    download_dir: Path,
) -> list[Path]:
    """Gather exactly 3 images starting from the reacted message.

    Priority:
    1. Images in the reacted message
    2. Other messages in the thread (if in a thread)
    3. Random images from #image-gen
    """
    msg = fetch_message(client, channel_id, ts)
    image_metas = extract_images_from_message(msg) if msg else []

    if len(image_metas) < 3 and thread_ts:
        thread_images = fetch_thread_images(client, channel_id, thread_ts, exclude_ts=ts)
        image_metas += thread_images

    image_metas = image_metas[:3]
    paths = download_images(token, image_metas, download_dir)

    if len(paths) < 3:
        needed = 3 - len(paths)
        extra = fetch_random_images(token, "image-gen", needed, download_dir / "extra")
        paths += extra

    return paths[:3]


def download_images(token: str, image_metas: list[dict], download_dir: Path) -> list[Path]:
    """Download image files to download_dir, return list of paths."""
    download_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(image_metas):
        dest = download_dir / f"source_{i}.{img['ext']}"
        data = _download_with_auth(img["url"], token)
        dest.write_bytes(data)
        logger.info(f"Downloaded {dest.name}")
        paths.append(dest)
    return paths
