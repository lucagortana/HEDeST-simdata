"""Whole-slide image access through OpenSlide (pyramidal TIFF, SVS, ...)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import openslide
from PIL import Image


def open_slide(path: str | Path) -> openslide.OpenSlide:
    return openslide.OpenSlide(str(path))


def size(path: str | Path) -> tuple[int, int]:
    """Level-0 ``(width, height)``."""
    with open_slide(path) as s:
        return s.dimensions


def level_for(slide: openslide.OpenSlide, max_dim: int) -> int:
    """Finest pyramid level whose longest side is at most ``max_dim``."""
    for lv, (w, h) in enumerate(slide.level_dimensions):
        if max(w, h) <= max_dim:
            return lv
    return slide.level_count - 1


def _tiff_page(path: str | Path, level: int, shape: tuple[int, int]) -> np.ndarray | None:
    """Pyramid ``level`` read straight from the TIFF, or None if it does not line up."""
    import tifffile as tf

    try:
        with tf.TiffFile(str(path)) as t:
            if level >= len(t.pages):
                return None
            page = t.pages[level]
            if tuple(page.shape[:2]) != shape:
                return None
            img = page.asarray()
    except Exception:
        return None
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return np.ascontiguousarray(img[..., :3])


def read_level(slide: openslide.OpenSlide, level: int,
               path: str | Path | None = None) -> tuple[np.ndarray, float]:
    """A whole pyramid level as RGB, with its scale relative to level 0.

    ``path`` is the file ``slide`` was opened from. Pass it: reading a whole level in one
    OpenSlide ``read_region`` has been seen to come back with a band of tiles **unfilled**
    and no error raised -- they are transparent, ``convert("RGB")`` blackens them, and the
    result was a grey slab across one Visium HD overview figure while the stage reported
    success. The pyramids here are written by ``common.tiff``, so their pages line up with
    OpenSlide's levels and tifffile can read them with no such failure mode.

    OpenSlide stays the fallback, and it now *checks* the alpha channel rather than trusting
    it: a level read entirely inside the slide has no transparent pixel, so one is the bug
    above and must not pass silently again.
    """
    w, h = slide.level_dimensions[level]
    scale = w / slide.dimensions[0]
    if path is not None:
        img = _tiff_page(path, level, (h, w))
        if img is not None:
            return img, scale
    region = slide.read_region((0, 0), level, (w, h))
    unfilled = int((np.asarray(region.split()[3]) == 0).sum())
    if unfilled:
        raise RuntimeError(
            f"OpenSlide left {unfilled} of {w * h} pixels of level {level} unfilled; "
            "the read failed without raising -- see read_level's docstring")
    return np.asarray(region.convert("RGB")), scale


def read_window(slide: openslide.OpenSlide, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    return np.asarray(slide.read_region((int(x0), int(y0)), 0, (int(w), int(h))).convert("RGB"))


def crops(slide: openslide.OpenSlide, centers_xy: np.ndarray, size_px: int, out_px: int) -> np.ndarray:
    """Square crops centred on each ``(x, y)``, resized to ``out_px``.

    OpenSlide pads outside the slide with transparent pixels, which become white.
    """
    out = np.empty((len(centers_xy), out_px, out_px, 3), dtype=np.uint8)
    half = size_px // 2
    for i, (cx, cy) in enumerate(centers_xy):
        region = slide.read_region((int(round(cx)) - half, int(round(cy)) - half), 0, (size_px, size_px))
        bg = Image.new("RGB", region.size, (255, 255, 255))
        bg.paste(region, mask=region.split()[3])
        out[i] = np.asarray(bg.resize((out_px, out_px), Image.BILINEAR))
    return out
