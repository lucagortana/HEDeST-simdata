"""Export the high-resolution H&E image to a tiled, pyramidal OME-TIFF."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from . import constants
from . import config, sdata_io


def export_he_tiff(root, out_path: Path, mpp: float) -> dict:
    """Write H&E (scale 0) as a pyramidal BigTIFF, reusing the zarr pyramid.

    Returns a dict with the full-resolution (height, width).
    """
    shapes = sdata_io.he_level_shapes(root)        # list of (C, Y, X)
    n_levels = len(shapes)
    n_sub = min(constants.TIFF_N_PYRAMID_LEVELS, n_levels - 1)

    base = sdata_io.read_he_level(root, 0)         # (C, Y, X) uint8
    base = np.moveaxis(base, 0, -1)                # (Y, X, C)
    height, width = base.shape[0], base.shape[1]

    opts = dict(
        photometric="rgb",
        tile=constants.TIFF_TILE,
        compression=constants.TIFF_COMPRESSION,
    )
    with tifffile.TiffWriter(str(out_path), bigtiff=True, ome=True) as tif:
        tif.write(
            base,
            subifds=n_sub,
            metadata={
                "axes": "YXC",
                "PhysicalSizeX": mpp,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": mpp,
                "PhysicalSizeYUnit": "µm",
            },
            **opts,
        )
        del base
        for level in range(1, n_sub + 1):
            sub = sdata_io.read_he_level(root, level)
            sub = np.moveaxis(sub, 0, -1)
            tif.write(sub, subfiletype=1, **opts)
            del sub

    return {"height": int(height), "width": int(width), "n_pyramid_levels": int(n_sub + 1)}
