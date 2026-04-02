"""Collage stencil selector bot.

Triggered by:
  - /collage-stencil <link1> <link2> <link3>  (slash command)
  - Any post in #image-index-gen with 3+ images
  - 🎨 reaction on a post in #image-gen, #img-junkyard, or #image-index-gen

Always posts results to #img-junkyard.

Requires:
  SLACK_BOT_TOKEN — bot OAuth token
  SLACK_APP_TOKEN — app-level token for Socket Mode (starts with xapp-)
"""
import logging
import os
import tempfile
from pathlib import Path

from PIL import Image
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

from selector_fetcher import (
    download_images,
    extract_images_from_message,
    fetch_message,
    gather_images_for_reaction,
    parse_message_link,
)
from slack_fetcher import find_channel_id
from slack_poster import post_collages
from stencil_transform import apply_stencil, make_stencil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]

POST_CHANNEL = "img-junkyard"
AUTO_TRIGGER_CHANNEL = "image-index-gen"
REACTION_CHANNELS = {"image-gen", "img-junkyard", "image-index-gen"}
TRIGGER_EMOJI = "art"  # 🎨

app = App(token=SLACK_BOT_TOKEN)
client = WebClient(token=SLACK_BOT_TOKEN)


def run_stencil(images: list[Image.Image], output_dir: Path) -> list[Path]:
    """Run 6-variation stencil transform on 3 images, return output paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for i, (s, a, b) in enumerate([(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        mask = make_stencil(images[s])
        result = apply_stencil(mask, images[a], images[b])
        dest = output_dir / f"stencil_result_{i + 1}.png"
        result.save(dest)
        output_paths.append(dest)
    return output_paths


def get_channel_name(channel_id: str) -> str:
    info = client.conversations_info(channel=channel_id)
    return info["channel"]["name"]


@app.command("/collage-stencil")
def handle_slash_command(ack, respond, command):
    ack()

    links = command.get("text", "").strip().split()
    if len(links) != 3:
        respond("Usage: `/collage-stencil <link1> <link2> <link3>`\nProvide exactly 3 Slack message links.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "source"
        out_dir = Path(tmpdir) / "output"

        image_metas = []
        for link in links:
            try:
                channel_id, ts = parse_message_link(link)
            except ValueError:
                respond(f"Invalid message link: `{link}`")
                return

            msg = fetch_message(client, channel_id, ts)
            if not msg:
                respond(f"Could not find message: `{link}`")
                return

            imgs = extract_images_from_message(msg)
            if not imgs:
                respond(f"No image found in message: `{link}`")
                return

            image_metas.append(imgs[0])

        paths = download_images(SLACK_BOT_TOKEN, image_metas, source_dir)
        images = [Image.open(p).convert("RGB") for p in paths]
        output_paths = run_stencil(images, out_dir)
        post_collages(SLACK_BOT_TOKEN, POST_CHANNEL, output_paths, bot_name="collage-stencil-selector-bot", threaded=False)
        respond(f"Done! Posted to #{POST_CHANNEL} :scissors:")


@app.event("reaction_added")
def handle_reaction(event):
    if event.get("reaction") != TRIGGER_EMOJI:
        return

    item = event.get("item", {})
    if item.get("type") != "message":
        return

    channel_id = item["channel"]
    ts = item["ts"]

    try:
        channel_name = get_channel_name(channel_id)
    except Exception:
        return

    if channel_name not in REACTION_CHANNELS:
        return

    msg = fetch_message(client, channel_id, ts)
    thread_ts = msg.get("thread_ts") if msg else None

    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "source"
        out_dir = Path(tmpdir) / "output"

        paths = gather_images_for_reaction(client, SLACK_BOT_TOKEN, channel_id, ts, thread_ts, source_dir)
        images = [Image.open(p).convert("RGB") for p in paths]
        output_paths = run_stencil(images, out_dir)
        post_collages(SLACK_BOT_TOKEN, POST_CHANNEL, output_paths, bot_name="collage-stencil-selector-bot", threaded=False)


@app.event("message")
def handle_message(event):
    # Only auto-trigger in #image-index-gen
    channel_id = event.get("channel")
    if event.get("bot_id") or event.get("subtype"):
        return

    try:
        channel_name = get_channel_name(channel_id)
    except Exception:
        return

    if channel_name != AUTO_TRIGGER_CHANNEL:
        return

    images = extract_images_from_message(event)
    if len(images) < 3:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "source"
        out_dir = Path(tmpdir) / "output"

        paths = download_images(SLACK_BOT_TOKEN, images[:3], source_dir)
        pil_images = [Image.open(p).convert("RGB") for p in paths]
        output_paths = run_stencil(pil_images, out_dir)
        post_collages(SLACK_BOT_TOKEN, POST_CHANNEL, output_paths, bot_name="collage-stencil-selector-bot", threaded=False)


if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
