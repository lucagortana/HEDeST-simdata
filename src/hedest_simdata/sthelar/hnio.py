"""Shared IO for HoVerNet-schema segmentation JSONs.

A HoVerNet JSON is ``{"mag": .., "mpp": .., "nuc": {id: entry}}`` where each
``entry`` has ``bbox`` (``[[ymin, xmin], [ymax, xmax]]``), ``centroid``
(``[x, y]``), ``contour`` (``[[x, y], ...]``), ``type`` (int) and often
``type_prob``. All coordinates are in the H&E ``he.tiff`` (level-0) pixel frame,
which is shared by the DAPI ``label2`` segmentation and the HoVerNet output, so
polygon overlap between the two is meaningful.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# HoVerNet PanNuke class id -> short name (external/hovernet/type_info.json).
HOVERNET_TYPE_NAMES = {
    0: "nolabe", 1: "neopla", 2: "inflam",
    3: "connec", 4: "necros", 5: "no-neo",
}


def load_nuc(path: str | Path) -> tuple[dict, dict]:
    """Return ``(full_data, nuc_dict)`` from a HoVerNet-schema JSON."""
    with open(path) as f:
        data = json.load(f)
    return data, data["nuc"]


def centroids(nuc: dict) -> tuple[list[str], np.ndarray]:
    """Ordered ``(ids, (N, 2) [x, y] float array)`` of nucleus centroids."""
    ids = list(nuc.keys())
    arr = np.array([nuc[i]["centroid"] for i in ids], dtype=np.float64)
    return ids, arr


def contour_xy(entry: dict) -> np.ndarray:
    """``(K, 2)`` integer ``[x, y]`` contour vertices."""
    return np.asarray(entry["contour"], dtype=np.int64)


def write_nuc(path: str | Path, nuc: dict, mag, mpp) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"mag": mag, "mpp": mpp, "nuc": nuc}, f)


def load_dapi_label2(mapping_csv: str | Path) -> dict[str, str]:
    """seg_id (str) -> label2 name, from a dapi_seg ``mapping_no_other.csv``.

    ``seg_id`` equals the corresponding nucleus key in the DAPI ``label2`` JSON.
    """
    out: dict[str, str] = {}
    with open(mapping_csv) as f:
        for row in csv.DictReader(f):
            out[str(row["seg_id"])] = row["cell_type_label2"]
    return out
