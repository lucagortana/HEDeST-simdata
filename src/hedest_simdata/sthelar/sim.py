"""Per-level ground truth: ``sim/level<k>/{proportions.csv, dapi_seg.json}``.

These are the DAPI-derived reference labels; they do **not** depend on HoVerNet.

* ``dapi_seg.json`` -- the source DAPI ``label2_no_other`` nuclei (no ``Other``),
  reindexed 0..n-1 *once* and shared by every level, with ``type`` recoded to the
  level's category index and ``type_name`` set to the category (``label2`` kept
  for reference). Same cells at every level, only the labels coarsen.
* ``proportions.csv`` -- per (non-empty DAPI) spot, the fraction of each category,
  columns in the level's canonical order. Derived from the source
  ``proportions_label2.csv`` by summing the columns of each category's leaves
  (fractions add, so rows still sum to 1). The ``ORIGINAL_LEVEL`` reproduces the
  source ``proportions_original.csv`` and the last level reproduces
  ``proportions_label2.csv`` -- both are asserted, value for value.

``levels.json`` records, per level, the category list and which level maps to the
existing ``original`` / ``label2`` annotations.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Layout
from . import hnio
from . import bench_hier, paths


def _dapi_cells(ds: paths.Dataset):
    """Return ``(mag, mpp, ordered [(new_id, label2, entry), ...])``.

    Cells are the source ``label2_no_other`` nuclei, reindexed 0..n-1 in the
    JSON's own key order; every cell's ``label2`` comes from the mapping CSV.
    """
    data, nuc = hnio.load_nuc(ds.dapi_json)
    seg2label2 = hnio.load_dapi_label2(ds.dapi_mapping)
    json_ids = set(nuc)
    map_ids = set(seg2label2)
    assert json_ids == map_ids, (
        f"{ds.name}: dapi json vs mapping seg_id mismatch "
        f"(json-only {len(json_ids - map_ids)}, map-only {len(map_ids - json_ids)})")
    leaf_set = set(bench_hier.leaves(ds.name))
    cells = []
    for new_id, seg_id in enumerate(nuc):          # JSON insertion order
        label2 = seg2label2[seg_id]
        assert label2 in leaf_set, f"{ds.name}: unexpected label2 {label2!r}"
        cells.append((new_id, label2, nuc[seg_id]))
    return data.get("mag"), data.get("mpp"), cells


def build_dapi_seg_levels(ds: paths.Dataset, cells, mag, mpp) -> None:
    for lv in range(bench_hier.n_levels(ds.name)):
        cats = bench_hier.categories(ds.name, lv)
        cat_idx = {c: i for i, c in enumerate(cats)}
        leaf2cat = bench_hier.leaf_to_category(ds.name, lv)
        out: dict[str, dict] = {}
        for new_id, label2, entry in cells:
            cat = leaf2cat[label2]
            e = dict(entry)
            e["type"] = int(cat_idx[cat])
            e["type_name"] = cat
            e["label2"] = label2
            out[str(new_id)] = e
        ldir = ds.level_dir(lv)
        ldir.mkdir(parents=True, exist_ok=True)
        hnio.write_nuc(ldir / "dapi_seg.json", out, mag, mpp)


def _group_columns(prop_label2: pd.DataFrame, name: str, level: int) -> pd.DataFrame:
    """Sum label2 columns into ``level``'s categories, in canonical order."""
    cats = bench_hier.categories(name, level)
    leaf2cat = bench_hier.leaf_to_category(name, level)
    out = pd.DataFrame(0.0, index=prop_label2.index, columns=cats)
    for leaf in prop_label2.columns:
        out[leaf2cat[leaf]] += prop_label2[leaf].to_numpy()
    return out


def build_proportions_levels(ds: paths.Dataset) -> dict:
    name = ds.name
    leaves = bench_hier.leaves(name)
    src_l2 = pd.read_csv(ds.prop_label2, index_col=0)
    assert set(src_l2.columns) == set(leaves), (
        f"{name}: proportions_label2 cols {set(src_l2.columns) ^ set(leaves)}")
    src_l2 = src_l2[leaves]                          # canonical leaf order

    checks: dict[str, str] = {}
    for lv in range(bench_hier.n_levels(name)):
        props = _group_columns(src_l2, name, lv)
        props.index.name = "spot_id"
        props.to_csv(ds.level_dir(lv) / "proportions.csv")

    # --- validation against the source annotations -------------------------
    lastlv = bench_hier.label2_level(name)
    last = _group_columns(src_l2, name, lastlv)
    assert np.allclose(last.to_numpy(), src_l2.to_numpy()), \
        f"{name}: last level != proportions_label2"
    checks["label2"] = "ok (== proportions_label2)"

    orig = _group_columns(src_l2, name, bench_hier.ORIGINAL_LEVEL)
    src_orig = pd.read_csv(ds.prop_original, index_col=0)
    assert set(orig.columns) == set(src_orig.columns), (
        f"{name}: original cols {set(orig.columns) ^ set(src_orig.columns)}")
    assert list(orig.index) == list(src_orig.index), \
        f"{name}: original spot index differs from source"
    aligned = src_orig[orig.columns].to_numpy()
    max_abs = float(np.max(np.abs(orig.to_numpy() - aligned)))
    assert max_abs < 1e-9, f"{name}: original proportions differ (max |Δ|={max_abs})"
    checks["original"] = f"ok (== proportions_original, max|Δ|={max_abs:.1e})"
    return checks


def write_levels_json(ds: paths.Dataset) -> None:
    name = ds.name
    lastlv = bench_hier.label2_level(name)
    meta = {
        "dataset": name,
        "n_levels": bench_hier.n_levels(name),
        "original_level": bench_hier.ORIGINAL_LEVEL,
        "label2_level": lastlv,
        "levels": [
            {
                "level": lv,
                "n_types": len(bench_hier.categories(name, lv)),
                "cell_types": bench_hier.categories(name, lv),
                "is_original": lv == bench_hier.ORIGINAL_LEVEL,
                "is_label2": lv == lastlv,
            }
            for lv in range(bench_hier.n_levels(name))
        ],
    }
    ds.sim_dir.mkdir(parents=True, exist_ok=True)
    (ds.sim_dir / "levels.json").write_text(json.dumps(meta, indent=2))


def build_sim(layout: Layout, name: str) -> dict:
    ds = paths.dataset(layout, name)
    ds.sim_dir.mkdir(parents=True, exist_ok=True)
    mag, mpp, cells = _dapi_cells(ds)
    build_dapi_seg_levels(ds, cells, mag, mpp)
    checks = build_proportions_levels(ds)
    write_levels_json(ds)
    return {"dataset": name, "n_dapi_cells": len(cells),
            "n_levels": bench_hier.n_levels(name), "validation": checks}
