"""HoVerNet-format nucleus segmentations.

The format is ``{"mag": int, "mpp": float, "nuc": {id: entry}}`` where each entry
holds ``centroid`` ``[x, y]``, ``contour`` ``[[x, y], ...]``, ``bbox``,
``type`` and ``type_prob``. Coordinates are level-0 pixels of the image the
segmentation ran on, ``x`` being the column.

HoVerNet's own ``bbox`` is expressed in another frame than ``centroid`` and
``contour`` (it does not enclose the contour), so :func:`select` recomputes it
from the contour as ``[[row_min, col_min], [row_max, col_max]]``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# PanNuke classes predicted by the HoVerNet "fast" type model, with the colours of
# HoVerNet's own type_info.json.
CLASS_NAMES = {0: "nolabe", 1: "neopla", 2: "inflam", 3: "connec", 4: "necros", 5: "no-neo"}
CLASS_COLORS = {
    0: (0.0, 0.0, 0.0),
    1: (1.0, 0.0, 0.0),
    2: (0.0, 1.0, 0.0),
    3: (0.0, 0.0, 1.0),
    4: (1.0, 1.0, 0.0),
    5: (1.0, 165 / 255, 0.0),
}


def load(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def contour_xy(entry: dict) -> np.ndarray:
    """``(K, 2)`` integer ``[x, y]`` contour vertices."""
    return np.asarray(entry["contour"], dtype=np.int64).reshape(-1, 2)


def bbox_from_contour(contour: np.ndarray) -> list[list[int]]:
    xs, ys = contour[:, 0], contour[:, 1]
    return [[int(ys.min()), int(xs.min())], [int(ys.max()), int(xs.max())]]


def centroids(nuc: dict, keys) -> np.ndarray:
    """``(N, 2)`` float ``[x, y]`` centroids for ``keys``, in that order."""
    return np.asarray([nuc[k]["centroid"] for k in keys], dtype=np.float64)


def select(nuc: dict, new_to_old: dict[str, str]) -> dict:
    """Keep only the nuclei in ``new_to_old`` and key them by the new ids.

    The order of ``new_to_old`` is the order of the output. Each entry keeps its
    HoVerNet fields, gets a bbox recomputed from its contour, and records the key
    it had in the source file as ``hovernet_key``.
    """
    out = {}
    for new_id, old_key in new_to_old.items():
        src = nuc[old_key]
        contour = contour_xy(src)
        out[str(new_id)] = {
            "bbox": bbox_from_contour(contour),
            "centroid": [float(v) for v in src["centroid"]],
            "contour": contour.tolist(),
            "type": int(src.get("type", 0)),
            "type_prob": None if src.get("type_prob") is None else float(src["type_prob"]),
            "hovernet_key": str(old_key),
        }
    return out


def write(path: str | Path, nuc: dict, mag, mpp: float) -> None:
    """Write atomically (temporary file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump({"mag": mag, "mpp": float(mpp), "nuc": nuc}, fh, separators=(",", ":"))
    os.replace(tmp, path)
