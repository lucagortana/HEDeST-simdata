"""Materialize higher-quality cropped sub-slides as self-contained datasets.

The crop analysis (see CONVERSATION_BACKGROUND.md sec.5) proposed rectangular
H&E windows that concentrate the well-segmented tissue; this module actually
cuts them. For every crop we write, under ``DATA_ROOT/cropped_data/<name>/``, the
same layout as a full sample:

* ``he.tiff``           -- cropped pyramidal H&E, extended by a small safety
                           margin (``HE_MARGIN_UM``) on each side (clamped to the
                           slide bounds) so cells at the crop edge keep full image
                           context; crop-local pixel frame,
* ``pseudovisium.h5ad`` -- only spots whose full 55um capture circle lies inside
                           the crop (no half spots); ``X`` (raw counts) is
                           untouched, the ``log_norm`` layer is recomputed over
                           the retained spots so the crop is self-contained,
* ``dapi_seg/``         -- only nuclei whose centroid lies inside the crop, with
                           coordinates re-based to the crop and the cell-type
                           integers re-indexed over the types actually present,
* ``proportions/``      -- per-spot proportions, columns harmonized to the types
                           present in the crop,
* ``metadata.json``     -- crop-specific metadata (dims, updated transform, type
                           maps, source bbox, ...),
* ``plots/``            -- the same verification figures as a full sample.

Coordinate convention: every pixel coordinate in a crop is the **crop-local**
scale-0 H&E frame, i.e. ``source_scale0 - (ex0, ey0)`` where ``(ex0, ey0)`` is the
top-left of the (margin-extended) H&E image. Cells and spots are still *selected*
on the crop bbox proper (``bbox_source_px``); the margin only enlarges the raster
and shifts the origin, it never adds cells or spots. Micron coordinates are
unchanged; the micron->H&E transform absorbs the origin shift
(``b_crop = L @ [ex0, ey0] + b``).
"""
from __future__ import annotations

import dataclasses
import json
import time
import types as _types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import tifffile
from scipy.sparse import csr_matrix

from . import constants
from .. import config
from ..config import Layout
from . import nuclei, paths, preprocess, pseudovisium, sdata_io
from .transforms import MicronToHE


# Per-crop label2 relabeling, applied AFTER cropping (source samples untouched).
# Collapses fragmented / too-rare fine subtypes into a coarser label *for that crop
# only*. The merged label must nest in a single coarse `original` group (enforced by
# the label2->group recomputation); merging two subtypes of the same group is safe.
# cervix_s0_1: its two epithelial subtypes are merged into one "Epithelial" because
# Secretory_epithelial_cell has only 12 nuclei / 3 spots in this crop.
LABEL2_MERGE: dict[str, dict[str, str]] = {
    "cervix_s0_1": {
        "Secretory_epithelial_cell": "Epithelial",
        "Squamous_epithelial_cell": "Epithelial",
    },
    # lymph_node_s0: collapse over-split immune subtypes. T + T/NK -> T_NK ; the two
    # mono/macro(/DC) subtypes -> Monocyte/Macrophage ; DC + pDC -> DC. Each merged
    # label stays within one coarse `original` group (T_NK, Myeloid, Myeloid). This
    # takes label2 from 11 -> 8 types.
    "lymph_node_s0": {
        "T": "T_NK",
        "T/NK": "T_NK",
        "Monocyte/Macrophage/DC": "Monocyte/Macrophage",
        "pDC": "DC",
    },
}

# H&E safety margin: the cropped he.tiff is the crop bbox extended by this many
# microns on every side (clamped to the true slide bounds -- no extension past a
# real slide edge). It gives full cell context for edge cells during downstream
# image extraction and does NOT change which cells/spots are kept (selection stays
# on the crop bbox). Set to 0.0 to disable.
HE_MARGIN_UM: float = 12.0


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# H&E tiff crop
# ---------------------------------------------------------------------------

def export_crop_tiff(source_root, x0, y0, x1, y1, out_path: Path, mpp: float) -> dict:
    """Write the crop as a tiled, pyramidal BigTIFF, reusing the source pyramid.

    Each pyramid level of the crop is read as the corresponding sub-region of the
    slide's own pyramid level (already properly downsampled), so we never have to
    downsample the multi-GB base ourselves.
    """
    shapes = sdata_io.he_level_shapes(source_root)
    n_levels = len(shapes)
    n_sub = min(constants.TIFF_N_PYRAMID_LEVELS, n_levels - 1)
    Y0, X0 = shapes[0][1], shapes[0][2]

    def _crop_level(lvl):
        _c, Yl, Xl = shapes[lvl]
        fx, fy = X0 / Xl, Y0 / Yl
        lx0 = int(round(x0 / fx)); lx1 = min(int(round(x1 / fx)), Xl)
        ly0 = int(round(y0 / fy)); ly1 = min(int(round(y1 / fy)), Yl)
        arr = source_root[f"images/he/{lvl}"][:, ly0:ly1, lx0:lx1]   # (C, ch, cw)
        return np.moveaxis(arr, 0, -1)                               # (ch, cw, C)

    base = _crop_level(0)
    height, width = base.shape[0], base.shape[1]
    opts = dict(photometric="rgb", tile=constants.TIFF_TILE, compression=constants.TIFF_COMPRESSION)
    with tifffile.TiffWriter(str(out_path), bigtiff=True, ome=True) as tif:
        tif.write(
            base,
            subifds=n_sub,
            metadata={
                "axes": "YXC",
                "PhysicalSizeX": mpp, "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": mpp, "PhysicalSizeYUnit": "µm",
            },
            **opts,
        )
        del base
        for lvl in range(1, n_sub + 1):
            sub = _crop_level(lvl)
            tif.write(sub, subfiletype=1, **opts)
            del sub
    return {"height": int(height), "width": int(width), "n_pyramid_levels": int(n_sub + 1)}


# ---------------------------------------------------------------------------
# cell records / dapi_seg
# ---------------------------------------------------------------------------

def _translate_and_restrict(rec, x0, y0, in_crop) -> nuclei.CellRecords:
    """Re-base a CellRecords to the crop origin and keep only ``in_crop`` cells.

    Micron centroids are unchanged; H&E centroids/contours/bboxes shift by
    ``-(x0, y0)``. ``rec`` and its fine-annotation twin share geometry, so
    ``in_crop`` (computed once) applies to both.
    """
    idx = np.where(in_crop)[0]
    off = np.array([x0, y0], dtype=np.int64)
    contours = [rec.contours_he[i] - off for i in idx]
    bboxes = rec.bboxes_he[idx].copy()          # [ymin, xmin, ymax, xmax]
    bboxes[:, [0, 2]] -= y0
    bboxes[:, [1, 3]] -= x0
    return nuclei.CellRecords(
        cell_ids=rec.cell_ids[idx],
        types=rec.types[idx],
        centroids_he=rec.centroids_he[idx] - np.array([x0, y0], dtype=float),
        centroids_um=rec.centroids_um[idx],
        contours_he=contours,
        bboxes_he=bboxes,
    )


# ---------------------------------------------------------------------------
# spots (AnnData) / proportions
# ---------------------------------------------------------------------------

def crop_adata(adata, sel, origin, radius_he):
    """Keep spots whose full capture circle lies inside the crop; re-base coords.

    ``sel = (sx0, sy0, sx1, sy1)`` is the crop bbox used to *select* spots (a spot
    is kept only if its whole capture circle lies inside it). ``origin = (ox, oy)``
    is the top-left of the (margin-extended) H&E image that all kept coordinates are
    re-based to -- so pixel coords become ``source - origin``, which may differ from
    ``sel`` when a safety margin is applied.

    Returns ``(cropped_adata, keep_mask)``. ``X`` (raw counts) is unchanged; the
    log-normalized layer is recomputed over the retained spots so the crop's
    normalization target reflects its own library sizes.
    """
    sx0, sy0, sx1, sy1 = sel
    ox, oy = origin
    cx = adata.obs["pxl_col_in_fullres"].to_numpy(dtype=float)
    cy = adata.obs["pxl_row_in_fullres"].to_numpy(dtype=float)
    keep = (
        (cx - radius_he >= sx0) & (cx + radius_he <= sx1)
        & (cy - radius_he >= sy0) & (cy + radius_he <= sy1)
    )
    sub = adata[keep].copy()
    sub.obs["pxl_col_in_fullres"] = sub.obs["pxl_col_in_fullres"].to_numpy(dtype=float) - ox
    sub.obs["pxl_row_in_fullres"] = sub.obs["pxl_row_in_fullres"].to_numpy(dtype=float) - oy
    spatial = np.asarray(sub.obsm["spatial"], dtype=np.float64).copy()
    spatial[:, 0] -= ox
    spatial[:, 1] -= oy
    sub.obsm["spatial"] = spatial
    raw = np.asarray(sub.X.todense(), dtype=np.float64)
    sub.layers[constants.LOGNORM_LAYER] = csr_matrix(
        pseudovisium._log_normalize(raw).astype(np.float32))
    return sub, keep


def crop_proportions(prop_df, keep_spot_ids, present_types, merge=None) -> pd.DataFrame:
    """Restrict a proportions table to the kept spots and to present cell types.

    If ``merge`` is given (``{old_type: new_type}``), columns are relabeled and the
    per-spot fractions of merged types are summed first (so row sums are preserved).
    Columns are then reindexed to ``present_types`` (the non-Other types in the crop,
    alphabetical). Dropped columns are all-zero for the kept spots (their cells are,
    by construction, inside the crop), so per-spot fractions stay valid.
    """
    ids = [s for s in keep_spot_ids if s in prop_df.index]
    sub = prop_df.loc[ids]
    if merge:
        sub = sub.T.groupby([merge.get(c, c) for c in sub.columns]).sum().T
    dropped = [c for c in sub.columns if c not in set(present_types)]
    if dropped and len(sub):
        lost = sub[dropped].to_numpy(dtype=float).sum()
        assert lost < 1e-6, (
            f"proportions harmonization would drop {lost:.4g} of mass in columns "
            f"{dropped} -- a spot references a cell type absent from the crop cells")
    return sub.reindex(columns=list(present_types), fill_value=0.0)


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def _annot_meta(prefix, column, seg, proportions_csv, extra=None):
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


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def crop_dataset(layout: Layout, crop_name: str, skip_tiff: bool = False) -> dict:
    if crop_name not in config.STHELAR_CROPS:
        raise KeyError(f"unknown crop {crop_name!r}; known: {sorted(config.STHELAR_CROPS)}")
    source_sample, bbox_req = config.STHELAR_CROPS[crop_name]
    t_start = time.time()
    ds = paths.dataset(layout, crop_name)
    src = paths.slide(layout, source_sample)
    out = ds.source_dir
    out.mkdir(parents=True, exist_ok=True)
    _log(f"=== {crop_name} (from {source_sample}) : opening zarr ===")
    store, source_root = sdata_io.open_zarr(src)
    try:
        scale = sdata_io.nucleus_scale(src, source_root)
        L, b, mpp = sdata_io.he_affine_and_mpp(source_root, scale)
        tf = MicronToHE(L, b, scale, mpp)

        shapes = sdata_io.he_level_shapes(source_root)
        H, W = int(shapes[0][1]), int(shapes[0][2])
        rx0, ry0, rx1, ry1 = bbox_req
        # selection bbox = the crop proper (decides which cells / spots are kept)
        sx0, sy0 = max(int(rx0), 0), max(int(ry0), 0)
        sx1, sy1 = min(int(rx1), W), min(int(ry1), H)
        sel_w, sel_h = sx1 - sx0, sy1 - sy0
        # image bbox = selection bbox extended by the H&E safety margin, clamped to
        # the true slide bounds (never extends past a real slide edge). All pixel
        # coordinates are re-based to the image origin (ex0, ey0).
        pad = int(round(HE_MARGIN_UM * tf.px_per_um))
        ex0, ey0 = max(sx0 - pad, 0), max(sy0 - pad, 0)
        ex1, ey1 = min(sx1 + pad, W), min(sy1 + pad, H)
        img_w, img_h = ex1 - ex0, ey1 - ey0
        margin_applied = {"left": sx0 - ex0, "top": sy0 - ey0,
                          "right": ex1 - sx1, "bottom": ey1 - sy1}
        radius_he = constants.SPOT_DIAMETER_UM / 2.0 * tf.px_per_um
        b_crop = (L @ np.array([ex0, ey0], dtype=np.float64) + b)
        _log(f"{crop_name}: bbox_req={bbox_req} -> crop x[{sx0},{sx1}) y[{sy0},{sy1}) "
             f"= {sel_w}x{sel_h}; +{pad}px he margin -> image x[{ex0},{ex1}) "
             f"y[{ey0},{ey1}) = {img_w}x{img_h} (applied L{margin_applied['left']} "
             f"T{margin_applied['top']} R{margin_applied['right']} B{margin_applied['bottom']})")

        # --- H&E tiff (margin-extended image) ------------------------------
        if not skip_tiff:
            _log(f"{crop_name}: exporting cropped H&E tiff ...")
            he_info = export_crop_tiff(source_root, ex0, ey0, ex1, ey1, out / "he.tiff", mpp)
            _log(f"{crop_name}: H&E {he_info['width']}x{he_info['height']} -> he.tiff")
        else:
            he_info = {"height": int(img_h), "width": int(img_w)}

        # --- cell records (both annotations) -------------------------------
        _log(f"{crop_name}: reading nucleus boundaries + cell types (2 levels) ...")
        rec, rec_fine, subtype_to_group = preprocess._build_records(src, tf)
        cx, cy = rec.centroids_he[:, 0], rec.centroids_he[:, 1]
        # select on the crop bbox; re-base to the (margin-extended) image origin
        in_crop = (cx >= sx0) & (cx < sx1) & (cy >= sy0) & (cy < sy1)
        rec_c = _translate_and_restrict(rec, ex0, ey0, in_crop)
        rec_fine_c = _translate_and_restrict(rec_fine, ex0, ey0, in_crop)
        keep_no_other = rec_c.types != constants.OTHER_LABEL
        assert np.array_equal(keep_no_other, rec_fine_c.types != constants.OTHER_LABEL), \
            "Other sets differ between annotations"

        # --- per-crop label2 merge (applied after cropping only) -----------
        merge = LABEL2_MERGE.get(crop_name, {})
        if merge:
            merged = np.array([merge.get(t, t) for t in rec_fine_c.types]).astype(str)
            rec_fine_c = dataclasses.replace(rec_fine_c, types=merged)
            _log(f"{crop_name}: label2 merge {merge}")

        all_mask = np.ones(len(rec_c.cell_ids), bool)
        group_all_c, group_no_c = preprocess._type_lists(rec_c)
        fine_all_c, fine_no_c = preprocess._type_lists(rec_fine_c)
        _log(f"{crop_name}: kept {int(in_crop.sum())}/{len(rec.cell_ids)} nuclei "
             f"({int(keep_no_other.sum())} non-Other); "
             f"original={len(group_no_c)} types, label2={len(fine_no_c)} types")

        # --- DAPI-segmentation JSONs + mappings ----------------------------
        _log(f"{crop_name}: writing DAPI-segmentation JSONs + mappings ...")
        seg_dir = out / "dapi_seg"
        seg_dir.mkdir(parents=True, exist_ok=True)
        seg = {
            "original_with_other": nuclei.write_dapi_seg_json(
                rec_c, all_mask, mpp, seg_dir / "original_with_other.json"),
            "original_no_other": nuclei.write_dapi_seg_json(
                rec_c, keep_no_other, mpp, seg_dir / "original_no_other.json"),
            "label2_with_other": nuclei.write_dapi_seg_json(
                rec_fine_c, all_mask, mpp, seg_dir / "label2_with_other.json"),
            "label2_no_other": nuclei.write_dapi_seg_json(
                rec_fine_c, keep_no_other, mpp, seg_dir / "label2_no_other.json"),
        }
        recs_by_col = {"cell_type_original": rec_c, "cell_type_label2": rec_fine_c}
        nuclei.write_dapi_seg_mapping(recs_by_col, all_mask, seg_dir / "mapping_with_other.csv")
        nuclei.write_dapi_seg_mapping(recs_by_col, keep_no_other, seg_dir / "mapping_no_other.csv")

        # --- spots (AnnData) -----------------------------------------------
        _log(f"{crop_name}: cropping pseudo-Visium spots ...")
        adata = ad.read_h5ad(src.adata_path)
        cropped, keep_spots = crop_adata(adata, (sx0, sy0, sx1, sy1), (ex0, ey0), radius_he)
        cropped.uns["spatial"] = {crop_name: adata.uns["spatial"][source_sample]}
        cropped.uns["pseudovisium"] = dict(adata.uns["pseudovisium"])
        cropped.uns["pseudovisium"]["crop"] = {
            "source_sample": source_sample,
            "bbox_source_px": [sx0, sy0, sx1, sy1],
            "image_bbox_source_px": [ex0, ey0, ex1, ey1],
            "origin_offset_px": [ex0, ey0],
            "he_margin_um": HE_MARGIN_UM,
            "he_margin_applied_px": margin_applied,
            "log_norm": "recomputed over the retained spots",
            # stored as [from, to] pairs, not a dict: h5ad/h5py treat "/" in a dict
            # key as a group-path separator, which breaks slash-bearing subtype names
            # (e.g. "T/NK", "Monocyte/Macrophage/DC").
            **({"label2_merge": [[k, v] for k, v in merge.items()]} if merge else {}),
        }
        cropped.write_h5ad(out / "pseudovisium.h5ad")
        _log(f"{crop_name}: kept {cropped.n_obs}/{adata.n_obs} spots")

        # --- proportions ---------------------------------------------------
        prop_dir = out / "proportions"
        prop_dir.mkdir(parents=True, exist_ok=True)
        src_prop_dir = src.proportions_dir
        prop_original = pd.read_csv(src_prop_dir / "proportions_original.csv", index_col=0)
        prop_label2 = pd.read_csv(src_prop_dir / "proportions_label2.csv", index_col=0)
        prop_original_c = crop_proportions(prop_original, cropped.obs_names, group_no_c)
        prop_label2_c = crop_proportions(prop_label2, cropped.obs_names, fine_no_c, merge=merge)
        prop_original_c.to_csv(prop_dir / "proportions_original.csv", index_label="spot_id")
        prop_label2_c.to_csv(prop_dir / "proportions_label2.csv", index_label="spot_id")

        # --- metadata ------------------------------------------------------
        # Recompute label2 -> coarse-group from the crop's (post-merge) cells; this
        # restricts to present subtypes and re-verifies that every (merged) label2
        # type still nests in exactly one coarse group.
        subtype_to_group_c = preprocess._label2_group_map(
            rec_fine_c.types[keep_no_other], rec_c.types[keep_no_other])
        meta = {
            "sample": crop_name,
            "crop": {
                "source_sample": source_sample,
                "bbox_requested_px": list(bbox_req),
                "bbox_source_px": [sx0, sy0, sx1, sy1],
                "image_bbox_source_px": [ex0, ey0, ex1, ey1],
                "origin_offset_px": [ex0, ey0],
                "he_margin_um": HE_MARGIN_UM,
                "he_margin_px": pad,
                "he_margin_applied_px": margin_applied,
                "note": ("crop-local scale-0 H&E frame = source scale-0 frame - "
                         "(ex0, ey0), the top-left of the margin-extended H&E image "
                         "(image_bbox_source_px). The he.tiff is the crop bbox "
                         "(bbox_source_px) grown by he_margin_um on each side, "
                         "clamped to the slide, to give edge cells full context. "
                         "Cells/spots are selected on bbox_source_px only -- the "
                         "margin adds none: spots kept iff their full capture circle "
                         "lies inside bbox_source_px, nuclei kept iff their centroid "
                         "lies inside bbox_source_px. Micron coords unchanged."),
                **({"label2_merge": merge} if merge else {}),
            },
            "mpp": mpp,
            "he": he_info,
            "n_nuclei_total": int(in_crop.sum()),
            "n_nuclei_no_other": int(keep_no_other.sum()),
            "n_spots": int(cropped.n_obs),
            "n_genes": int(cropped.n_vars),
            "expression": {
                "X": "raw counts (sum of member cells' raw Xenium counts, "
                     "table_cells layers['count']); unchanged by the crop",
                "layers": {constants.LOGNORM_LAYER:
                           "normalize_total(median) + log1p of X, recomputed over "
                           "the retained spots"},
            },
            "spot_geometry": {
                "diameter_um": constants.SPOT_DIAMETER_UM,
                "spacing_um": constants.SPOT_SPACING_UM,
                "radius_um": constants.SPOT_DIAMETER_UM / 2.0,
                "radius_he_px": radius_he,
                "diameter_he_px": constants.SPOT_DIAMETER_UM * tf.px_per_um,
                "arrangement": "staggered_honeycomb",
                "cell_in_spot": "nucleus centroid inside the spot circle",
            },
            "transform_micron_to_he": {
                "L_he_xy_to_global_xy": L.tolist(),
                "b_he_xy_to_global_xy": b_crop.tolist(),
                "nucleus_scale": scale,
                "note": ("crop-local he_xy (relative to the extended-image origin "
                         "ex0, ey0); global_xy = L @ he_xy + b_crop ; "
                         "global_xy = nucleus_scale * micron_xy ; "
                         "b_crop = L @ [ex0, ey0] + b_source"),
            },
            "annotations": {
                constants.ANNOT_ORIGINAL: _annot_meta(
                    "original", constants.CELLTYPE_COL, seg, "proportions_original.csv"),
                constants.ANNOT_FINE: _annot_meta(
                    "label2", constants.CELLTYPE_COL_FINE, seg, "proportions_label2.csv",
                    extra={"subtype_to_group": subtype_to_group_c,
                           **({"merged_subtypes": merge} if merge else {})}),
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
        _log(f"=== {crop_name}: done in {time.time() - t_start:.0f}s ===")
        return meta
    finally:
        store.close()
