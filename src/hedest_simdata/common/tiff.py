"""Writes a tiled, pyramidal TIFF that OpenSlide can read. Used by two families.

Every pipeline here hands HoVerNet a whole-slide image, and HoVerNet reads it through
OpenSlide, which needs the pyramid laid out a particular way. Getting it wrong does not
fail loudly: OpenSlide falls back to reading level 0 for everything and the run takes
hours longer, or the magnification comes out wrong and the cell windows are the wrong
physical size.

Extracted from ``VisiumHD_analysis/analysis/preprocessing/hovernet_seg.py``, where it was
written and debugged, because the Xenium family needs the same thing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: Tile size of the written pyramid.
TILE = 512

#: Level 0 plus four reductions. The tissue mask is computed at level 3, so there have to
#: be at least four.
PYRAMID_LEVELS = 5


def write_pyramid(img: np.ndarray, out_path: Path, mpp: float,
                  tile: int = TILE, levels: int = PYRAMID_LEVELS) -> dict:
    """Writes ``img`` as a tiled, multi-level pyramidal BigTIFF.

    Two details that are not optional:

    * **each level is its own top-level IFD** with the reduced-image subfiletype, all
      tiled -- not SubIFDs. OpenSlide's generic-tiff driver only recognises a pyramid in
      that form.
    * **the resolution tags carry the true mpp**, which is how HoVerNet's
      ``wsi_handler.find_slide_magnification`` recovers 40x for these slides (0.274 / 0.25
      rounds to 1). Without it the magnification is guessed and the 20 um cell windows come
      out the wrong size.
    """
    import cv2
    import tifffile as tf

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = 1.0 / (mpp * 1e-4)              # pixels per centimetre
    opts = dict(tile=(tile, tile), photometric="rgb", compression="deflate",
                resolution=(res, res), resolutionunit="CENTIMETER")

    shapes = []
    with tf.TiffWriter(str(out_path), bigtiff=True) as writer:
        writer.write(img, **opts)
        shapes.append(tuple(img.shape[:2]))
        level = img
        for _ in range(levels - 1):
            h, w = level.shape[:2]
            if min(h, w) < tile:
                break
            level = cv2.resize(level, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            writer.write(level, subfiletype=1, **opts)
            shapes.append(tuple(level.shape[:2]))
    return {"path": str(out_path), "mpp": mpp, "levels": len(shapes),
            "level_shapes": [list(s) for s in shapes],
            "bytes": out_path.stat().st_size}


def read_whole(path: Path) -> np.ndarray:
    """A whole image as RGB, with PIL's decompression-bomb guard lifted.

    These slides are tens of thousands of pixels on a side, so PIL's default cap refuses
    them; the guard exists for untrusted input, which this is not.
    """
    import PIL.Image
    import tifffile as tf

    PIL.Image.MAX_IMAGE_PIXELS = None
    img = tf.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)     # channels-first to channels-last
    return np.ascontiguousarray(img[..., :3])
