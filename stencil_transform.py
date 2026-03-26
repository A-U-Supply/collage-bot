"""Stencil transform: use image 1 as a binary mask to composite images 2 and 3."""
import numpy as np
from PIL import Image


def _otsu_threshold(gray_arr: np.ndarray) -> int:
    """Compute Otsu's optimal threshold for a grayscale numpy array."""
    hist, _ = np.histogram(gray_arr.flatten(), bins=256, range=(0, 256))
    total = gray_arr.size
    total_sum = np.dot(np.arange(256), hist)

    best_thresh = 0
    best_variance = 0.0
    bg_sum = 0
    bg_count = 0

    for t in range(256):
        bg_count += hist[t]
        if bg_count == 0:
            continue
        fg_count = total - bg_count
        if fg_count == 0:
            break

        bg_sum += t * hist[t]
        bg_mean = bg_sum / bg_count
        fg_mean = (total_sum - bg_sum) / fg_count

        variance = bg_count * fg_count * (bg_mean - fg_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_thresh = t

    return best_thresh


def make_stencil(img: Image.Image) -> Image.Image:
    """Convert image to a binary mask using Otsu's thresholding.

    Returns a single-channel (L mode) image: 255 = white, 0 = black.
    """
    gray = np.array(img.convert("L"))
    thresh = _otsu_threshold(gray)
    binary = (gray > thresh).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def apply_stencil(
    mask: Image.Image,
    img2: Image.Image,
    img3: Image.Image,
) -> Image.Image:
    """Composite img2 (white regions) and img3 (black regions) using mask.

    All images are resized to mask dimensions before compositing.
    """
    w, h = mask.size
    img2 = img2.convert("RGB").resize((w, h), Image.LANCZOS)
    img3 = img3.convert("RGB").resize((w, h), Image.LANCZOS)

    mask_arr = np.array(mask)
    img2_arr = np.array(img2)
    img3_arr = np.array(img3)

    # Where mask > 0 (white), use img2; elsewhere use img3
    composite_arr = np.where(mask_arr[:, :, np.newaxis] > 0, img2_arr, img3_arr)
    return Image.fromarray(composite_arr.astype(np.uint8), mode="RGB")
