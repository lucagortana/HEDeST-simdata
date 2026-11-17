"""Spots: one grid, nuclei -> spots, bins -> counts, and the per-version files.

* A nucleus belongs to the spot whose centre is within 27.5 um of its centroid.
  Proportions at a level are the fractions of the spot's nuclei.
* A spot's counts are the sum of bin2cell's filtered, destriped 2 um bins whose
  centre falls in the circle -- whichever cell they belong to, if any.
* Only spots holding at least one nucleus are kept.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..common import lattice
from . import slide

from . import inputs
from .config import Sample, log, update_metadata
from .labels import Hierarchy

HIRES_MAX_DIM = 6500


def hierarchy_from_json(d: dict) -> Hierarchy:
    return Hierarchy(d["columns"], d["levels"], [dict(p) for p in d["parent"]],
                     {(int(k), n) for k, n in d["unassigned"]})


def hierarchy_to_json(h: Hierarchy) -> dict:
    return {"columns": h.columns, "levels": h.levels, "parent": h.parent,
            "unassigned": sorted([k, n] for k, n in h.unassigned)}


def read_cell_table(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(out_dir) / "sim" / "cell_types.csv", dtype={"cell_id": str, "hovernet_key": str},
                       keep_default_na=False)


def _write_csv(df: pd.DataFrame, path: Path, **kw) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, **kw)
    os.replace(tmp, path)


def _log_norm(X: sp.csr_matrix) -> sp.csr_matrix:
    tot = np.asarray(X.sum(axis=1)).ravel()
    target = float(np.median(tot[tot > 0])) if (tot > 0).any() else 1.0
    scale = np.divide(target, tot, out=np.zeros_like(tot), where=tot > 0)
    out = sp.diags(scale) @ X
    out = out.tocsr()
    out.data = np.log1p(out.data)
    return out.astype(np.float32)


def run(s: Sample, seg: dict) -> dict:
    mpp = seg["mpp"]["used"]
    offset = np.asarray(seg["crop_offset_xy"], dtype=np.float64)
    h = hierarchy_from_json(seg["hierarchy"])
    cells = read_cell_table(s.out_dir)
    xy = cells[["x", "y"]].to_numpy(dtype=np.float64)

    grid = lattice.honeycomb(xy * mpp, s.spot_spacing_um, s.spot_diameter_um,
                              margin_rings=lattice.MARGIN_RINGS["visiumhd"])
    spot = lattice.assign(xy * mpp, grid)
    keep_spots = np.unique(spot[spot >= 0])
    member = spot >= 0
    log(f"grid {len(grid):,} spots; {len(keep_spots):,} hold a nucleus; "
        f"{member.sum():,} of {len(xy):,} nuclei inside a spot")

    row_of = pd.Series(np.arange(len(keep_spots)), index=keep_spots)
    m_row = row_of.loc[spot[member]].to_numpy()
    n_cells = np.bincount(m_row, minlength=len(keep_spots))
    sids = np.array([f"spot_{i:06d}" for i in keep_spots])

    # ---- counts from the bins ----
    bins = inputs.load_bins(s, seg["crop_key"])
    b_spot = lattice.assign(bins.xy_crop * mpp, grid)
    b_idx = np.flatnonzero(np.isin(b_spot, keep_spots))
    X, var = inputs.load_bin_matrix(s)
    log(f"bins: {len(b_spot):,} filtered+destriped, {len(b_idx):,} inside a kept spot; matrix {X.shape}")
    A = sp.csr_matrix((np.ones(len(b_idx)), (row_of.loc[b_spot[b_idx]].to_numpy(), np.arange(len(b_idx)))),
                      shape=(len(keep_spots), len(b_idx)))
    counts = (A @ X[b_idx]).tocsr().astype(np.float32)
    n_bins = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    del X, A

    # ---- proportions ----
    props = {}
    for k, col in enumerate(h.columns):
        cats = h.levels[k]
        code = pd.Categorical(cells.loc[member, f"lv{k}"], categories=cats).codes
        if (code < 0).any():
            raise RuntimeError(f"lv{k} labels outside the hierarchy")
        cnt = np.zeros((len(keep_spots), len(cats)))
        np.add.at(cnt, (m_row, code), 1)
        props[k] = pd.DataFrame(cnt / n_cells[:, None], index=pd.Index(sids, name="spot_id"), columns=cats)

    sd: dict[str, list[str]] = {sid: [] for sid in sids}
    for cid, r in zip(cells.loc[member, "cell_id"], m_row):
        sd[sids[r]].append(cid)
    cells["spot_id"] = ""
    cells.loc[member, "spot_id"] = sids[m_row]

    with slide.open_slide(s.pyr_tif) as sl:
        lv = slide.level_for(sl, HIRES_MAX_DIM)
        hires, hires_scale = slide.read_level(sl, lv, s.pyr_tif)

    centers_um = grid.centers[keep_spots]
    centers_px = centers_um / mpp
    tot = np.asarray(counts.sum(axis=1)).ravel()
    obs = pd.DataFrame({
        "array_row": grid.array_row[keep_spots],
        "array_col": grid.array_col[keep_spots],
        "pxl_col_in_fullres": centers_px[:, 0],
        "pxl_row_in_fullres": centers_px[:, 1],
        "pxl_col_in_btf": centers_px[:, 0] + offset[0],
        "pxl_row_in_btf": centers_px[:, 1] + offset[1],
        "x_um": centers_um[:, 0],
        "y_um": centers_um[:, 1],
        "n_cells": n_cells,
        "n_bins": n_bins,
        "total_counts": tot,
        "n_genes_by_counts": np.diff(counts.indptr),
        "in_tissue": 1,
    }, index=pd.Index(sids, name="spot_id"))
    for k in props:
        obs[f"dominant_lv{k}"] = pd.Categorical(props[k].idxmax(axis=1), categories=h.levels[k])
    adata = ad.AnnData(X=counts, obs=obs, var=var.copy())
    adata.layers["log_norm"] = _log_norm(counts)
    adata.obsm["spatial"] = centers_px.astype(np.float64)
    adata.obsm["spatial_btf"] = (centers_px + offset).astype(np.float64)
    for k in props:
        adata.obsm[f"proportions_lv{k}"] = props[k].to_numpy(dtype=np.float32)
    adata.uns["spatial"] = {s.name: {
        "images": {"hires": hires},
        "scalefactors": {"tissue_hires_scalef": float(hires_scale),
                         "spot_diameter_fullres": float(s.spot_diameter_um / mpp),
                         "microns_per_pixel": float(mpp)},
        "metadata": {"image": str(s.pyr_tif), "hires_level": int(lv)}}}
    adata.uns["pseudovisium"] = {
        "sample": s.name,
        "X": "sum of bin2cell's filtered, destriped 2 um bin counts whose centre lies in the spot",
        "layers": {"log_norm": "normalised to the median spot total, then log1p"},
        "coordinates": {
            "spatial / pxl_*_in_fullres": "level-0 px of the capture image (pyr_tif), the frame of hovernet.json",
            "spatial_btf / pxl_*_in_btf": "level-0 px of the full .btf image",
            "crop_offset_xy": offset.tolist()},
        "mpp": float(mpp),
        "spot_diameter_um": float(s.spot_diameter_um),
        "spot_spacing_um": float(s.spot_spacing_um),
        "levels": {f"lv{k}": col for k, col in enumerate(h.columns)},
        "proportion_columns": {f"lv{k}": list(h.levels[k]) for k in props},
        "n_nuclei": int(len(cells)),
        "n_nuclei_in_spots": int(member.sum()),
        "unassigned": "cells unassigned at any level were removed from every level",
    }
    tmp = s.out_dir / "pseudovisium.h5ad.tmp"
    adata.write_h5ad(tmp, compression="gzip")
    os.replace(tmp, s.out_dir / "pseudovisium.h5ad")

    sim = s.out_dir / "sim"
    for k in props:
        _write_csv(props[k], sim / f"proportions_lv{k}.csv")
    (sim / "levels.json").write_text(json.dumps(
        {"sample": s.name, "n_levels": h.n_levels,
         "files": {f"proportions_lv{k}.csv": col for k, col in enumerate(h.columns)},
         "unassigned_pattern": s.unassigned_pattern,
         "unassigned_types_removed": seg["unassigned"]["types_removed"],
         "levels": h.to_json()}, indent=2))
    tmp = s.out_dir / "spot_dict.json.tmp"
    tmp.write_text(json.dumps(sd))
    os.replace(tmp, s.out_dir / "spot_dict.json")
    _write_csv(cells, sim / "cell_types.csv", index=False)

    info = {"grid": {"spacing_um": s.spot_spacing_um, "diameter_um": s.spot_diameter_um,
                     "n_grid_spots": len(grid), "row_pitch_um": s.spot_spacing_um * np.sqrt(3) / 2},
            "spots": int(len(keep_spots)), "nuclei": int(len(cells)),
            "nuclei_in_spots": int(member.sum()),
            "bins_total": int(len(b_spot)), "bins_in_spots": int(len(b_idx)),
            "genes": int(var.shape[0]),
            "median_nuclei_per_spot": float(np.median(n_cells)),
            "median_bins_per_spot": float(np.median(n_bins)),
            "median_counts_per_spot": float(np.median(tot)),
            "types_per_level": [len(l) for l in h.levels]}
    update_metadata(s.out_dir, "spots", info)
    log(f"{info}")
    return info
