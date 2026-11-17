"""Build per-nucleus records and DAPI-segmentation JSON files (in H&E pixels).

The JSONs use the **HoVerNet output schema** (so they stay drop-in compatible
with HoVerNet-consuming tooling), but no HoVerNet inference is run: the nuclei
are STHELAR's Xenium/DAPI nucleus segmentation, only re-projected into H&E pixels
and re-keyed here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from .transforms import MicronToHE


@dataclass
class CellRecords:
    cell_ids: np.ndarray            # (n,) original zarr cell_id
    types: np.ndarray               # (n,) cell-type name (str)
    centroids_he: np.ndarray        # (n, 2) H&E pixel (x, y)
    centroids_um: np.ndarray        # (n, 2) micron (x, y)  [for spot assignment]
    contours_he: list               # list of (k, 2) int arrays, H&E pixel (x, y)
    bboxes_he: np.ndarray           # (n, 4) [ymin, xmin, ymax, xmax] int


def build_cell_records(gdf, cell_types: pd.Series, tf: MicronToHE) -> CellRecords:
    """Transform nucleus polygons (micron) to H&E pixels and gather attributes."""
    cell_ids = gdf.index.astype(str).to_numpy()
    geoms = gdf.geometry.values

    # Vectorised coordinate transform for every vertex of every polygon.
    coords_um = shapely.get_coordinates(geoms)             # (T, 2)
    counts = shapely.get_num_coordinates(geoms)            # (n,)
    coords_he = tf(coords_um)
    splits = np.cumsum(counts)[:-1]
    contours_all = np.split(coords_he, splits)

    contours_he, bboxes = [], np.empty((len(geoms), 4), dtype=np.int64)
    for i, c in enumerate(contours_all):
        c = np.rint(c).astype(np.int64)
        if len(c) > 1 and np.array_equal(c[0], c[-1]):
            c = c[:-1]                                     # drop closing vertex
        contours_he.append(c)
        xs, ys = c[:, 0], c[:, 1]
        bboxes[i] = [ys.min(), xs.min(), ys.max(), xs.max()]

    centroids_um = shapely.get_coordinates(shapely.centroid(geoms))
    centroids_he = tf(centroids_um)

    types = cell_types.reindex(cell_ids).to_numpy().astype(str)
    return CellRecords(
        cell_ids=cell_ids,
        types=types,
        centroids_he=centroids_he,
        centroids_um=centroids_um,
        contours_he=contours_he,
        bboxes_he=bboxes,
    )


def _mag_from_mpp(mpp: float) -> int:
    # 40x scanners are ~0.25 um/px, 20x ~0.5 um/px.
    return int(round(10.0 / mpp))


def write_dapi_seg_json(
    rec: CellRecords,
    keep_mask: np.ndarray,
    mpp: float,
    json_path: Path,
) -> dict:
    """Write one DAPI-segmentation JSON (HoVerNet schema) for ``keep_mask``.

    Nuclei are re-keyed contiguously "0".."m-1". ``type`` is the alphabetical
    index of the cell type within the kept set. Returns a small summary dict
    (type map + counts) for the metadata file. The seg_id -> cell_id mapping
    is written separately (see ``write_dapi_seg_mapping``) so a single mapping
    file can carry the cell type from several annotations at once.
    """
    idx = np.where(keep_mask)[0]
    kept_types = rec.types[idx]
    type_names = sorted(np.unique(kept_types).tolist())
    type_to_int = {name: i for i, name in enumerate(type_names)}

    nuc = {}
    for new_id, i in enumerate(idx):
        key = str(new_id)
        contour = rec.contours_he[i]
        nuc[key] = {
            "bbox": [
                [int(rec.bboxes_he[i, 0]), int(rec.bboxes_he[i, 1])],
                [int(rec.bboxes_he[i, 2]), int(rec.bboxes_he[i, 3])],
            ],
            "centroid": [float(rec.centroids_he[i, 0]), float(rec.centroids_he[i, 1])],
            "contour": contour.tolist(),
            "type_prob": 1.0,
            "type": int(type_to_int[rec.types[i]]),
        }

    payload = {"mag": _mag_from_mpp(mpp), "mpp": round(mpp, 6), "nuc": nuc}
    with open(json_path, "w") as fh:
        json.dump(payload, fh)

    return {
        "n_cells": int(len(idx)),
        "cell_types": type_names,
        "type_map": type_to_int,
        "json": json_path.name,
    }


def write_dapi_seg_mapping(recs_by_col: dict, keep_mask: np.ndarray, mapping_path: Path):
    """Write ``seg_id -> cell_id`` plus one cell-type column per annotation.

    ``recs_by_col`` maps an output column name (e.g. ``"cell_type_original"``) to
    the :class:`CellRecords` carrying that annotation's ``types``. All records
    must share the same geometry / cell ordering (they do: only ``types``
    differs), and ``keep_mask`` must be identical across annotations, so the
    re-keyed ``seg_id`` refers to the same nucleus in every column.
    """
    idx = np.where(keep_mask)[0]
    any_rec = next(iter(recs_by_col.values()))
    data = {
        "seg_id": [str(k) for k in range(len(idx))],
        "cell_id": any_rec.cell_ids[idx],
    }
    for col, rec in recs_by_col.items():
        data[col] = rec.types[idx]
    pd.DataFrame(data).to_csv(mapping_path, index=False)
