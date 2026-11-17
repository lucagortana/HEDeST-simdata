"""Nucleus <-> cell <-> counts <-> cell type: the matching checks.

The match itself is by id. HoVerNet names nuclei ``"0".."n-1"``; painting them
into a label image had to start at 1 (bin2cell reads 0 as "no nucleus"), and
bin2cell names each cell after its label. So a cell id is a label, and the table
written at painting time gives its HoVerNet key (key + 1 = label). QC only
removes cells, it never renumbers them -- but ids are therefore not row
positions, and nothing here ever matches by position.

The checks, each against a deliberately wrong control:

C1  JSON keys are "0".."n-1" in order; the label table matches the JSON exactly
C2  cell ids are unique, equal ``object_id`` and are all HoVerNet labels
C3  for EACH cell, the label under its nucleus centre (or anywhere inside its
    outline, for concave nuclei) is its id                          -> 100 %
C4  among nuclei whose centre 2 um bin exists, that bin belongs to the same cell
                                                                      -> >= 99 %
C5  for EACH cell, the annotated total count equals bin2cell's total for the
    same id                                                           -> 100 %
C6  (reported) each cell type scores highest on its own marker genes
Controls (C3-C5 with id + 1) must stay <= 1 %.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.draw import polygon as sk_polygon

from . import hovernet

from .inputs import Bins, Cells, check

C4_MIN = 0.99
CONTROL_MAX = 0.01

# Marker genes for the colon types seen so far (C6 only scores types listed here).
COLON_MARKERS = {
    "Tumor Epithelial": ["EPCAM", "CEACAM5", "KRT8"],
    "Tumor Epithelial Inflamed": ["LCN2", "DMBT1", "PI3"],
    "Epithelial": ["EPCAM", "KRT8", "KRT20", "CDH17"],
    "Goblet": ["MUC2", "CLCA1", "FCGBP", "ZG16"],
    "Paneth cell": ["DEFA5", "DEFA6", "REG3A", "LYZ"],
    "Fibroblast": ["COL1A1", "COL3A1", "DCN", "LUM"],
    "Smooth Muscle": ["DES", "MYH11", "ACTG2", "CNN1"],
    "Vascular": ["PECAM1", "VWF", "PLVAP", "CD34"],
    "Endothelial Vascular": ["PECAM1", "VWF", "PLVAP"],
    "Lymphatic Vascular": ["PROX1", "LYVE1", "CCL21"],
    "Macrophage": ["CD68", "C1QA", "C1QB", "CD163"],
    "Neutrophil": ["S100A8", "S100A9", "FCGR3B", "CSF3R"],
    "Mast cell": ["TPSAB1", "CPA3", "KIT"],
    "Plasma cell": ["JCHAIN", "IGKC", "MZB1", "IGHA1"],
    "B cell": ["MS4A1", "CD79A", "CD19", "BANK1"],
    "T cell / NK cell": ["CD3D", "CD3E", "NKG7", "GNLY", "CD2"],
    "T cell": ["CD3D", "CD3E", "TRAC", "CD2"],
}


class MatchError(RuntimeError):
    pass


def _labels_under(mask, xy: np.ndarray) -> np.ndarray:
    r = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, mask.shape[0] - 1)
    c = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, mask.shape[1] - 1)
    return np.asarray(mask[r, c]).ravel()


def _px_to_array(bins: Bins) -> np.ndarray:
    A = np.c_[bins.xy_crop, np.ones(len(bins.xy_crop))]
    M, *_ = np.linalg.lstsq(A, np.c_[bins.array_row, bins.array_col], rcond=None)
    return M


def run(cells: Cells, nuc: dict, table: pd.DataFrame, table_source: str, mask, bins: Bins,
        totals: pd.Series, slide_wh, markers: pd.DataFrame | None, level3_col: str) -> tuple[list[dict], np.ndarray, dict]:
    """Returns the check list, the HoVerNet key of every cell, and C6 scores."""
    out: list[dict] = []
    ids = cells.object_id

    # C1 ---------------------------------------------------------------------
    keys = list(nuc)
    in_order = keys == [str(i) for i in range(len(keys))]
    t_ok = table["key"].isin(nuc.keys()).all() and table.index.is_unique and table["key"].is_unique
    xy_json = hovernet.centroids(nuc, table["key"].tolist()) if t_ok else None
    same_xy = bool(t_ok and np.allclose(xy_json, table[["x", "y"]].to_numpy()))
    offsets = (table.index.to_numpy() - table["key"].astype(np.int64).to_numpy()) if t_ok else np.array([])
    out.append(check("C1", "HoVerNet keys in order; label table matches the JSON",
                     in_order and t_ok and same_xy and len(table) == len(nuc),
                     value={"nuclei": len(nuc), "table_rows": len(table), "keys_in_order": in_order,
                            "centroids_identical": same_xy, "table_source": table_source,
                            "label_minus_key": sorted(set(offsets.tolist()))}, rule="exact"))

    # C2 ---------------------------------------------------------------------
    uniq = pd.Index(ids).is_unique
    names_ok = bool((cells.ids == ids.astype(str)).all())
    known = np.isin(ids, table.index.to_numpy())
    out.append(check("C2", "cell ids unique, equal to object_id, all HoVerNet labels",
                     uniq and names_ok and known.all(),
                     value={"cells": len(ids), "unique": uniq, "obs_names_equal_object_id": names_ok,
                            "ids_not_in_hovernet": int((~known).sum())}, rule="exact"))
    if not (uniq and names_ok and known.all() and t_ok):
        raise MatchError("ids cannot be matched; stopping before the geometric checks")
    cell_keys = table.loc[ids, "key"].to_numpy().astype(str)
    xy = table.loc[ids, ["x", "y"]].to_numpy()

    W, H = slide_wh
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
    out.append(check("C2b", "every nucleus centre lies inside the capture image", inside.all(),
                     value={"outside": int((~inside).sum())}, rule="exact"))

    # C3 ---------------------------------------------------------------------
    hit = _labels_under(mask, xy) == ids
    rescued = 0
    for i in np.flatnonzero(~hit):
        c = hovernet.contour_xy(nuc[cell_keys[i]])
        rr, cc = sk_polygon(c[:, 1], c[:, 0], mask.shape)
        if rr.size and (np.asarray(mask[rr, cc]).ravel() == ids[i]).any():
            hit[i] = True
            rescued += 1
    shifted = np.isin(ids + 1, table.index.to_numpy())
    ctrl = float((_labels_under(mask, table.loc[ids[shifted] + 1, ["x", "y"]].to_numpy()) == ids[shifted]).mean())
    out.append(check("C3", "label under each nucleus is its cell id", hit.all() and ctrl <= CONTROL_MAX,
                     value=float(hit.mean()), rule="== 1 (per cell)", control=ctrl, control_rule=f"<= {CONTROL_MAX}",
                     details={"at_centre": int(hit.sum() - rescued), "inside_outline_only": rescued,
                              "failing_ids": ids[~hit][:20].tolist()}))

    # C4 ---------------------------------------------------------------------
    M = _px_to_array(bins)
    grid = pd.Series(bins.labels_joint, index=pd.MultiIndex.from_arrays([bins.array_row, bins.array_col]))

    def bin_owner(points):
        rc = np.rint(np.c_[points, np.ones(len(points))] @ M).astype(np.int64)
        return grid.reindex(pd.MultiIndex.from_arrays([rc[:, 0], rc[:, 1]])).to_numpy(dtype=float)

    own = bin_owner(xy)
    has_bin = ~np.isnan(own)
    frac = float((own[has_bin] == ids[has_bin]).mean())
    own_s = bin_owner(table.loc[ids[shifted] + 1, ["x", "y"]].to_numpy())
    hb = ~np.isnan(own_s)
    ctrl4 = float((own_s[hb] == ids[shifted][hb]).mean())
    out.append(check("C4", "nucleus sits inside its own cell (centre bin, where it exists)",
                     frac >= C4_MIN and ctrl4 <= CONTROL_MAX, value=frac, rule=f">= {C4_MIN}",
                     control=ctrl4, control_rule=f"<= {CONTROL_MAX}",
                     details={"no_bin_at_centre": int((~has_bin).sum()),
                              "bin_of_another_cell": int((own[has_bin] != ids[has_bin]).sum())}))

    # C5 ---------------------------------------------------------------------
    known5 = np.isin(ids, totals.index.to_numpy())
    same = np.zeros(len(ids), dtype=bool)
    same[known5] = np.isclose(cells.total_counts[known5], totals.loc[ids[known5]].to_numpy(), rtol=1e-6, atol=1e-6)
    nxt = np.isin(ids + 1, totals.index.to_numpy())
    ctrl5 = float(np.isclose(cells.total_counts[nxt], totals.loc[ids[nxt] + 1].to_numpy(), rtol=1e-6, atol=1e-6).mean())
    out.append(check("C5", "each cell's counts are bin2cell's counts for that id",
                     same.all() and ctrl5 <= CONTROL_MAX, value=float(same.mean()), rule="== 1 (per cell)",
                     control=ctrl5, control_rule=f"<= {CONTROL_MAX}",
                     details={"missing_in_bin2cell": int((~known5).sum()),
                              "failing_ids": ids[~same][:20].tolist()}))

    # C6 (reported) ------------------------------------------------------------
    scores = {}
    if markers is not None and not markers.empty:
        types = cells.labels[level3_col].astype(str).to_numpy()
        sets = {t: [g for g in COLON_MARKERS[t] if g in markers.columns]
                for t in pd.unique(types) if t in COLON_MARKERS}
        sets = {t: g for t, g in sets.items() if g}

        def score(labels):
            mu = markers.assign(_t=labels).groupby("_t").mean()
            S = pd.DataFrame({t: mu[g].mean(axis=1) for t, g in sets.items()})
            S = (S - S.mean()) / S.std(ddof=0)
            return S.loc[list(sets)], [t for t in sets if S.loc[t].idxmax() == t]

        S, top = score(types)
        _, top_ctrl = score(np.random.default_rng(0).permutation(types))
        scores = {"types": list(sets), "matrix": S.to_numpy().round(3).tolist(), "top": top}
        out.append(check("C6", "each cell type scores highest on its own marker genes",
                         len(top) == len(sets), hard=False, value=f"{len(top)}/{len(sets)}",
                         control=f"{len(top_ctrl)}/{len(sets)} (labels shuffled)",
                         details={"misses": [t for t in sets if t not in top],
                                  "types_without_markers": [t for t in pd.unique(types) if t not in sets]}))
    return out, cell_keys, scores


def failures(checks: list[dict]) -> list[dict]:
    return [c for c in checks if c["hard"] and not c["passed"]]
