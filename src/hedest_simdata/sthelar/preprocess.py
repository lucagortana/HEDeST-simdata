"""End-to-end per-sample preprocessing: STHELAR Xenium -> pseudo-Visium v2.

Two cell-type annotation levels are produced for every sample and kept side by
side (never one replacing the other):

* ``original`` -- the coarse harmonized taxonomy ``cells_final_label_group``,
* ``label2``   -- the finer *curated* taxonomy ``cells_label2``.

``label2`` nests strictly inside ``original`` (every subtype maps to exactly one
coarse group) and both agree exactly on which cells are ``Other`` (the coarse
``Other`` set defines the dropped cells for both). Because the dropped set is
identical, the pseudo-spot geometry, ``n_cells`` and the raw-count transcriptome
are annotation-independent -- only the per-spot proportions differ.
"""
from __future__ import annotations

import dataclasses
import json
import time
import types as _types

import numpy as np
import pandas as pd

from . import constants
from ..config import Layout
from . import he_export, nuclei, paths, pseudovisium, sdata_io
from .transforms import MicronToHE


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_records(sl: paths.Slide, tf: MicronToHE):
    """CellRecords for both annotations, sharing geometry and the ``Other`` set.

    Returns ``(rec_original, rec_label2)`` where ``rec_label2`` differs only in
    ``types``. The coarse ``Other`` (plus any nucleus missing from the metadata)
    defines the dropped set for *both*, so their Other masks are identical.
    """
    labels = sdata_io.read_cell_labels(
        sl, [constants.CELLTYPE_COL, constants.CELLTYPE_COL_FINE])
    gdf = sdata_io.read_nucleus_boundaries(sl)
    rec = nuclei.build_cell_records(gdf, labels[constants.CELLTYPE_COL], tf)  # types=coarse
    label2 = labels[constants.CELLTYPE_COL_FINE].reindex(rec.cell_ids).to_numpy().astype(str)

    other = (rec.types == constants.OTHER_LABEL) | (rec.types == "nan")
    coarse = np.where(other, constants.OTHER_LABEL, rec.types)
    fine = np.where(other, constants.OTHER_LABEL, label2)
    rec = dataclasses.replace(rec, types=coarse)
    rec_fine = dataclasses.replace(rec, types=fine)
    subtype_to_group = _label2_group_map(fine[~other], coarse[~other])
    return rec, rec_fine, subtype_to_group


def _label2_group_map(fine: np.ndarray, coarse: np.ndarray) -> dict:
    """label2 -> coarse-group map; raises if any subtype spans two groups."""
    df = pd.DataFrame({"fine": fine, "coarse": coarse})
    agg = df.groupby("fine")["coarse"].agg(["nunique", "first"])
    bad = agg[agg["nunique"] > 1]
    if len(bad):
        raise AssertionError(
            f"{constants.CELLTYPE_COL_FINE} does not nest in {constants.CELLTYPE_COL}: "
            f"{list(bad.index)}")
    return {k: v for k, v in agg["first"].items()}


def _type_lists(rec):
    all_types = sorted(np.unique(rec.types).tolist())
    no_other = [t for t in all_types if t != constants.OTHER_LABEL]
    return all_types, no_other



def preprocess_sample(layout: Layout, sample: str, skip_tiff: bool = False) -> dict:
    t_start = time.time()
    sl = paths.slide(layout, sample)
    out = sl.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _log(f"=== {sample}: opening zarr ===")
    store, root = sdata_io.open_zarr(sl)
    try:
        scale = sdata_io.nucleus_scale(sl, root)
        L, b, mpp = sdata_io.he_affine_and_mpp(root, scale)
        tf = MicronToHE(L, b, scale, mpp)
        _log(f"{sample}: mpp={mpp:.5f} um/px, nucleus_scale={scale:.5f}")

        # --- H&E TIFF -------------------------------------------------------
        if not skip_tiff:
            _log(f"{sample}: exporting H&E tiff ...")
            he_info = he_export.export_he_tiff(root, out / "he.tiff", mpp)
            _log(f"{sample}: H&E {he_info['width']}x{he_info['height']} -> he.tiff")
        else:
            shp = sdata_io.he_level_shapes(root)[0]
            he_info = {"height": int(shp[1]), "width": int(shp[2])}

        # --- nuclei / cell records (both annotations) ----------------------
        _log(f"{sample}: reading nucleus boundaries + cell types (2 levels) ...")
        rec, rec_fine, subtype_to_group = _build_records(sl, tf)
        keep_no_other = rec.types != constants.OTHER_LABEL
        assert np.array_equal(keep_no_other, rec_fine.types != constants.OTHER_LABEL), \
            "Other sets differ between annotations"
        group_all, group_no = _type_lists(rec)
        fine_all, fine_no = _type_lists(rec_fine)
        _log(f"{sample}: {len(rec.cell_ids)} nuclei; "
             f"original={len(group_no)} types, label2={len(fine_no)} types (non-Other)")

        # --- DAPI-segmentation JSONs (HoVerNet schema; 2 annotations x with/without Other) ---
        _log(f"{sample}: writing DAPI-segmentation JSONs + mappings ...")
        seg_dir = sl.dapi_seg_dir
        seg_dir.mkdir(parents=True, exist_ok=True)
        all_mask = np.ones(len(rec.cell_ids), bool)
        seg = {
            "original_with_other": nuclei.write_dapi_seg_json(
                rec, all_mask, mpp, seg_dir / "original_with_other.json"),
            "original_no_other": nuclei.write_dapi_seg_json(
                rec, keep_no_other, mpp, seg_dir / "original_no_other.json"),
            "label2_with_other": nuclei.write_dapi_seg_json(
                rec_fine, all_mask, mpp, seg_dir / "label2_with_other.json"),
            "label2_no_other": nuclei.write_dapi_seg_json(
                rec_fine, keep_no_other, mpp, seg_dir / "label2_no_other.json"),
        }
        recs_by_col = {"cell_type_original": rec, "cell_type_label2": rec_fine}
        nuclei.write_dapi_seg_mapping(recs_by_col, all_mask, seg_dir / "mapping_with_other.csv")
        nuclei.write_dapi_seg_mapping(recs_by_col, keep_no_other, seg_dir / "mapping_no_other.csv")

        # --- pseudo-Visium grid + aggregation (annotation-independent) -----
        _log(f"{sample}: building pseudo-Visium grid + aggregating raw counts ...")
        centroids_no = rec.centroids_um[keep_no_other]
        cell_ids_no = rec.cell_ids[keep_no_other]
        grid = pseudovisium.build_grid(centroids_no, tf)
        expr = sdata_io.read_cell_expression(root)               # raw counts (count layer)
        agg = pseudovisium.aggregate(
            rec_centroids_um=centroids_no,
            rec_cell_ids=cell_ids_no,
            expr_cell_ids=expr.cell_ids,
            expr_X=expr.X,
            genes=expr.genes,
            grid=grid,
            tf=tf,
            sample=sample,
        )
        adata = agg.adata
        prop_original = pseudovisium.compute_proportions(
            agg, rec.types[keep_no_other], group_no)
        prop_label2 = pseudovisium.compute_proportions(
            agg, rec_fine.types[keep_no_other], fine_no)
        _log(f"{sample}: {adata.n_obs} spots x {adata.n_vars} genes "
             f"({int(agg.n_cells_per_spot.sum())} cells aggregated)")

        adata.write_h5ad(out / "pseudovisium.h5ad")
        prop_dir = sl.proportions_dir
        prop_dir.mkdir(parents=True, exist_ok=True)
        prop_original.to_csv(prop_dir / "proportions_original.csv", index_label="spot_id")
        prop_label2.to_csv(prop_dir / "proportions_label2.csv", index_label="spot_id")

        # --- metadata (both annotation levels) -----------------------------
        def _annot_meta(prefix, column, proportions_csv, extra=None):
            wo, no = seg[f"{prefix}_with_other"], seg[f"{prefix}_no_other"]
            d = {
                "column": column,
                "n_cells_with_other": wo["n_cells"],
                "n_cells_no_other": no["n_cells"],
                "cell_types_with_other": wo["cell_types"],
                "type_map_with_other": wo["type_map"],
                "cell_types_no_other": no["cell_types"],
                "type_map_no_other": no["type_map"],
                "dapi_seg_json_with_other": f"dapi_seg/{prefix}_with_other.json",
                "dapi_seg_json_no_other": f"dapi_seg/{prefix}_no_other.json",
                "proportions_csv": f"proportions/{proportions_csv}",
            }
            if extra:
                d.update(extra)
            return d

        meta = {
            "sample": sample,
            "mpp": mpp,
            "he": he_info,
            "n_nuclei_total": int(len(rec.cell_ids)),
            "n_nuclei_no_other": int(keep_no_other.sum()),
            "n_spots": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "expression": {
                "X": "raw counts (sum of member cells' raw Xenium counts, "
                     "table_cells layers['count'])",
                "layers": {constants.LOGNORM_LAYER: "normalize_total(median) + log1p of X"},
            },
            "spot_geometry": {
                "diameter_um": constants.SPOT_DIAMETER_UM,
                "spacing_um": constants.SPOT_SPACING_UM,
                "radius_um": grid.radius_um,
                "radius_he_px": grid.radius_he,
                "diameter_he_px": constants.SPOT_DIAMETER_UM * tf.px_per_um,
                "arrangement": "staggered_honeycomb",
                "cell_in_spot": "nucleus centroid inside the spot circle",
            },
            "transform_micron_to_he": {
                "L_he_xy_to_global_xy": L.tolist(),
                "b_he_xy_to_global_xy": b.tolist(),
                "nucleus_scale": scale,
                "note": ("global_xy = L @ he_xy + b ; global_xy = nucleus_scale * micron_xy ; "
                         "he_xy = L^-1 @ (nucleus_scale * micron_xy - b)"),
            },
            "annotations": {
                constants.ANNOT_ORIGINAL: _annot_meta(
                    "original", constants.CELLTYPE_COL, "proportions_original.csv"),
                constants.ANNOT_FINE: _annot_meta(
                    "label2", constants.CELLTYPE_COL_FINE, "proportions_label2.csv",
                    extra={"subtype_to_group": subtype_to_group}),
            },
            "dapi_seg_mapping": {
                "with_other": "dapi_seg/mapping_with_other.csv",
                "no_other": "dapi_seg/mapping_no_other.csv",
                "columns": ["seg_id", "cell_id",
                            "cell_type_original", "cell_type_label2"],
            },
            "outputs": {
                "he_tiff": "he.tiff",
                "anndata": "pseudovisium.h5ad",
                "proportions_dir": "proportions/",
                "dapi_seg_dir": "dapi_seg/",
            },
        }

        with open(out / "metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        _log(f"=== {sample}: done in {time.time() - t_start:.0f}s ===")
        return meta
    finally:
        store.close()
