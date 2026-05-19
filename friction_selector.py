"""Friction selector — choose N images from a pool that maximise visual contrast.

Friction between two images is a weighted sum of three pairwise distances:
  1. Tonal distance     — absolute difference in mean brightness
  2. Texture distance   — difference in edge energy (variance of Laplacian)
  3. Colour distance    — Bhattacharyya distance between LAB hue histograms

Triplet friction = sum of all three pairwise friction scores.
The pool is scored exhaustively (C(n,3) comparisons) and the highest-scoring
triplet is returned.
"""
from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Weights for the three friction components (must sum to 1.0 for interpretability,
# but the selection only cares about relative ordering so rescaling is fine).
_W_TONAL = 0.35
_W_TEXTURE = 0.35
_W_COLOUR = 0.30


def _features(img_rgb: np.ndarray) -> dict:
    """Compute a compact feature vector for one image."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Tonal: mean brightness (0–255)
    mean_brightness = float(gray.mean())

    # 2. Texture: variance of the Laplacian — high = lots of edges/detail
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture_energy = float(lap.var())

    # 3. Colour: normalised hue histogram in LAB a*b* space (robust to exposure)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a_hist = cv2.calcHist([lab], [1], None, [32], [0, 256]).flatten()
    b_hist = cv2.calcHist([lab], [2], None, [32], [0, 256]).flatten()
    ab_hist = np.concatenate([a_hist, b_hist])
    ab_hist = ab_hist / (ab_hist.sum() + 1e-8)

    return {
        "brightness": mean_brightness,
        "texture": texture_energy,
        "colour_hist": ab_hist,
    }


def _pair_friction(f1: dict, f2: dict) -> float:
    """Friction score in [0, 1] for one image pair."""
    tonal = abs(f1["brightness"] - f2["brightness"]) / 255.0

    # Log-scale texture distance so extreme outliers don't dominate
    t1 = np.log1p(f1["texture"])
    t2 = np.log1p(f2["texture"])
    max_t = max(t1, t2, 1.0)
    texture = abs(t1 - t2) / max_t

    # Bhattacharyya distance: 0 = identical, 1 = maximally different
    bc = float(cv2.compareHist(
        f1["colour_hist"].astype(np.float32),
        f2["colour_hist"].astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    ))
    colour = min(bc, 1.0)

    return _W_TONAL * tonal + _W_TEXTURE * texture + _W_COLOUR * colour


def select_friction_triplet(paths: list[Path], pool_size: int | None = None) -> list[Path]:
    """Return the 3 paths from *paths* whose pairwise friction is maximised.

    Args:
        paths:     List of image file paths (the pool).
        pool_size: If set, only the first *pool_size* paths are considered
                   (useful when the caller already limited the fetch count).

    Returns:
        List of 3 Paths — the highest-friction triplet.
    """
    if pool_size is not None:
        paths = paths[:pool_size]

    if len(paths) < 3:
        raise ValueError(f"Need at least 3 images, got {len(paths)}")

    if len(paths) == 3:
        return list(paths)

    logger.info(f"Computing friction scores for {len(paths)} images "
                f"({len(paths) * (len(paths)-1) * (len(paths)-2) // 6} triplets)...")

    imgs = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB"))
        # Downscale for speed — friction features don't need full resolution
        h, w = arr.shape[:2]
        scale = min(1.0, 256 / max(h, w))
        if scale < 1.0:
            arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        imgs.append(arr)

    feats = [_features(img) for img in imgs]

    # Pre-compute all pairwise friction scores
    n = len(paths)
    pair_scores = {}
    for i, j in combinations(range(n), 2):
        pair_scores[(i, j)] = _pair_friction(feats[i], feats[j])

    # Find the triplet with maximum total pairwise friction
    best_score = -1.0
    best_triplet = (0, 1, 2)
    for i, j, k in combinations(range(n), 3):
        score = pair_scores[(i, j)] + pair_scores[(i, k)] + pair_scores[(j, k)]
        if score > best_score:
            best_score = score
            best_triplet = (i, j, k)

    i, j, k = best_triplet
    selected = [paths[i], paths[j], paths[k]]
    logger.info(
        f"Selected images {i+1}, {j+1}, {k+1} "
        f"(friction score: {best_score:.3f}) "
        f"from pool of {n}"
    )
    return selected
