"""3-level stencil transform: use one image as a mask with black/grey/white regions."""
import numpy as np
from PIL import Image


def make_3level_stencil(img: Image.Image) -> Image.Image:
    """Convert image to a 3-level mask using percentile thresholding.

    Splits pixel brightness into three equal-population bands:
      - bottom third  -> 0   (black)
      - middle third  -> 128 (grey)
      - top third     -> 255 (white)

    Returns a single-channel (L mode) image.
    """
    gray = np.array(img.convert("L"))
    p33 = np.percentile(gray, 33)
    p67 = np.percentile(gray, 67)

    result = np.zeros_like(gray)
    result[gray >= p33] = 128
    result[gray >= p67] = 255

    return Image.fromarray(result, mode="L")


def apply_3level_stencil(
    mask: Image.Image,
    img_black: Image.Image,
    img_grey: Image.Image,
    img_white: Image.Image,
) -> Image.Image:
    """Composite 3 images using a 3-level mask.

    Regions where mask == 0   -> img_black
    Regions where mask == 128 -> img_grey
    Regions where mask == 255 -> img_white
    """
    w, h = mask.size
    img_black = img_black.convert("RGB").resize((w, h), Image.LANCZOS)
    img_grey = img_grey.convert("RGB").resize((w, h), Image.LANCZOS)
    img_white = img_white.convert("RGB").resize((w, h), Image.LANCZOS)

    mask_arr = np.array(mask)
    b_arr = np.array(img_black)
    g_arr = np.array(img_grey)
    w_arr = np.array(img_white)

    result = np.where(
        mask_arr[:, :, np.newaxis] < 64,
        b_arr,
        np.where(mask_arr[:, :, np.newaxis] < 192, g_arr, w_arr),
    )

    return Image.fromarray(result.astype(np.uint8), mode="RGB")
