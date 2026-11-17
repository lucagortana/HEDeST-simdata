"""Builds one Xenium slide's six deliverables from the segmentation and the annotation.

This is the part of the port that had to be reverse-engineered rather than translated. The
two source notebooks (157 and 117 cells) explored several variants side by side and were
last saved in a configuration that is **not** the one that shipped: their output paths
point at a ``sim_3types/`` folder and their last cells collapse everything onto three
PanNuke names, while the delivered datasets have seven cell types in ``sim/`` and
``sim/LuCA/``. So every choice here is pinned from the delivered files and checked against
them, and the places where the two slides differ are explicit instead of implied:

===============  =========================  =============================================
                 lung                       breast
===============  =========================  =============================================
annotation       committed CSV, re-keyed    the dataset page's spreadsheet,
                 to this release's ids by   **positionally** (this Xenium release has
                 centroid                   no barcodes at all)
lattice          283 / 440                  350 / 500
catch-all        ``low_counts``             ``Unlabeled``
===============  =========================  =============================================

Both notebooks set a variable called ``threshold`` twice -- 10 for a transcript-count
filter and then 40 for the matching distance -- and the matching cell reads whichever came
last. That is the kind of thing that does not survive a translation, so the thresholds are
named fields on :class:`~hedest_simdata.config.XeniumSlide` now.

How close the rebuild comes
--------------------------
Measured by rebuilding both slides from a clean tree -- nothing reused, every intermediate
re-derived -- and comparing every deliverable against the delivered one. An earlier round
of this section reported both slides as near-exact; it had been measured in a work
directory that still held the delivered segmentation, so the stages that matter had never
run. These numbers are from the clean build.

**Lung agrees on everything downstream of the cells.**

* ``spot_dict_real.json`` / ``prop_real.csv`` / ``pseudo_adata_real.h5ad`` -- 1,958 spots
  against 1,957, **every one of the delivered 1,957 present**, the extra one holding a
  single nucleus; and 34,716 nuclei in spots against 34,716, equal to the unit.
* ``sim_<slide>_gt.csv`` -- 107,433 rows against 107,429.
* ``pannuke_fast_mask_lvl3_annotated.json`` -- 141,257 nuclei in both; codes 0-6 (the seven
  cell types) identical bar 4 Epithelial cells.

This section used to report 1,813 spots against 1,957 and blame the bounding box the
lattice is built over. That was wrong, and wrong in an instructive way: the cause was
``min_nuclei_per_spot``, shared at 4 by both slides, where lung's delivered
``spot_dict_real.json`` has a minimum of **1** -- 54 spots holding one annotated nucleus,
50 holding two, 40 holding three, which is exactly the 144 that went missing, and exactly
the 274 nuclei that made up the "275 extra" the bounding-box story was invented to explain.
Breast's minimum really is 4. A threshold shared by two datasets looked right for as long
as only one of them was compared, and the fix is :attr:`XeniumSlide.delivered_spots`: a
count taken from the delivered data, which the code cannot satisfy by agreeing with itself.

The one residual is **cosmetic**: code 7 (matched a cell whose type was dropped) is 13,126
against 10,576 and code 8 (nothing matched) 20,698 against 23,252, so the rebuild matches
2,554 more nuclei and every one of them to a dropped type -- which is why the ground truth,
the proportions and the spots are unaffected, all three being taken after the dropped types
leave. In HEDeST these two codes are read only by ``analysis/postseg.py``, to build a
fallback palette for an overlay; no published number depends on which of them a
non-callable nucleus carries.

It sits exactly where the dataset's re-release broke the chain, which is the best available
account of it. The delivered archive's ``cell_boundaries`` is keyed by the **retired**
sequential ids (``aaaaaaaa-1``, ``aaaaaaab-1``, ...) while its ``cell_circles`` and
``nucleus_boundaries`` carry the published ones; a fresh download has all three on the
published ids. The counts are identical in both (162,254 / 162,254 / 156,628), so the
geometry is the same -- but the reference's nucleus -> cell -> type hop therefore ran
entirely in the retired id space, and this one runs in the published space through the
re-key. The two agree on 107,429 of 107,433 callable nuclei and differ on which of the two
non-callable buckets 2,554 others fall into.

**Breast is no longer exact, and the reason is the H&E pyramid.** The delivered dataset was
built on a 20511 x 27587 crop of the H&E; this repo builds the full 24241 x 30786 image
from the same OME-TIFF, so HoVerNet sees more tissue and finds 170,888 nuclei against
175,204, with ids that do not correspond. See :mod:`hedest_simdata.xenium.fetch` for how
that crop was found -- it is recorded nowhere except as a re-fit of the published affine --
and why it is not reproduced. What the rebuild gives, against the delivered dataset:

* **2,656 spots against 2,651** (+0.19%), and 43,362 nuclei in them.
* 123,358 of 167,780 Xenium nuclei matched, 73.5% against the delivered 75.8% -- the gap is
  the 4,316 nuclei the crop cost, not the alignment, which lands its matches at a 6.85 px
  median where the committed affine managed 62.38.
* ground truth 116,752 rows against 120,336, and per-type nucleus counts 1-4% below the
  delivered ones (46,215 / 9,143 / 30,415 / 7,013 / 13,101 / 6,536 / 4,329 against
  48,365 / 9,287 / 30,937 / 7,353 / 13,285 / 6,722 / 4,387) -- a shortfall that tracks the
  2.5% fewer nuclei rather than any change of rule.

"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from .. import config
from ..common import lattice
from . import annotation
from . import spatial_tools as st
from .paths import Slide


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class BuildError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------


def _affine(matrix) -> tuple:
    """A 3x3 spatialdata matrix as shapely's ``(a, b, d, e, xoff, yoff)``."""
    return (matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1],
            matrix[0, 2], matrix[1, 2])


def _scale_affine(scale) -> tuple:
    sx, sy = (float(s) for s in scale)
    return (sx, 0.0, 0.0, sy, 0.0, 0.0)


def load_sdata(sl: Slide):
    """The Xenium archive with the aligned H&E added as ``he_image``."""
    import spatialdata as sd
    from spatialdata_io.readers.xenium import xenium_aligned_image

    if not sl.sdata_zarr.exists():
        raise BuildError(f"{sl.sdata_zarr} is missing; the convert stage writes it")
    sdata = sd.read_zarr(str(sl.sdata_zarr))
    if not sl.he_alignment.exists():
        raise BuildError(f"{sl.he_alignment} is missing; it is one of the downloads")
    sdata.images["he_image"] = xenium_aligned_image(str(sl.pyr_tif),
                                                    str(sl.he_alignment))
    return sdata


def hovernet_shapes(sl: Slide, sdata):
    """HoVerNet contours as a GeoDataFrame in the H&E image's coordinate system."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    from spatialdata.models import ShapesModel
    from spatialdata.transformations import get_transformation

    raw = json.loads(sl.seg_json.read_text())
    ids, polys = [], []
    for cell_id, cell in raw["nuc"].items():
        contour = cell.get("contour", [])
        if len(contour) >= 3:
            ids.append(cell_id)
            polys.append(Polygon(contour))
    gdf = gpd.GeoDataFrame({"geometry": polys}, index=pd.Index(ids, name=None))
    gdf = ShapesModel.parse(gdf,
                            transformations={"global": get_transformation(sdata["he_image"])})
    _log(f"{sl.key}: {len(raw['nuc']):,} HoVerNet nuclei, {len(gdf):,} with a usable contour")
    return raw, gdf


# --------------------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------------------


def _in_global(sdata, name: str):
    """One of the Xenium shape layers, in the global (micron) frame."""
    from spatialdata.transformations import get_transformation

    layer = sdata.shapes[name]
    out = layer.copy()
    out.geometry = st.apply_transformation_to_gdf(
        out, _scale_affine(get_transformation(layer).__dict__["scale"]))
    return out


def nucleus_to_cell(sl: Slide, sdata) -> pd.Series | None:
    """``Xenium nucleus id -> Xenium cell id``, or None when the nuclei *are* the cells.

    Only lung needs this. Its ``nucleus_boundaries`` is a separate id space from its cell
    table -- 156,628 nuclei whose ids barely intersect the 162,254 cell ids -- so a matched
    nucleus carries no cell type until it is joined to the cell containing it. Breast's
    nuclei are one-per-cell and share the cell's id, so the hop is a no-op there and is
    skipped rather than faked.
    """
    if sl.spec.nucleus_to_cell_overlap is None:
        return None
    nuclei = _in_global(sdata, "nucleus_boundaries")
    cells = _in_global(sdata, "cell_boundaries")
    inter = st.match_nuclei_and_cells(nuclei, cells)
    kept = st.filter_matches(inter, overlap_threshold=sl.spec.nucleus_to_cell_overlap)
    _log(f"{sl.key}: {len(kept):,} of {len(nuclei):,} Xenium nuclei sit >= "
         f"{sl.spec.nucleus_to_cell_overlap:.0%} inside one cell")
    return kept.set_index(kept["nucleus_id"].astype(str))["cell_id"].astype(str)


def match(sl: Slide, sdata, gdf_hn, types: pd.Series) -> pd.DataFrame:
    """``nucleus_id -> cell_id, cell_type`` for every matched HoVerNet nucleus.

    The catch-all type is kept here on purpose: it is what tells "matched but not callable"
    (code 7) apart from "not matched at all" (code 8) when the segmentation is written.
    """
    from spatialdata.transformations import get_transformation

    hn = gdf_hn.copy()
    hn.geometry = st.apply_transformation_to_gdf(
        hn, _affine(get_transformation(gdf_hn).matrix))

    if sl.spec.match == "intersect":
        inter = st.match_nuclei_and_cells(hn, _in_global(sdata, "cell_boundaries"))
        out = st.filter_matches(inter, overlap_threshold=sl.spec.match_threshold)
        out = out[["nucleus_id", "cell_id"]].copy()
        _log(f"{sl.key}: {len(inter):,} nucleus/cell intersections, "
             f"{len(out):,} with overlap >= {sl.spec.match_threshold}")
    elif sl.spec.match == "closest":
        closest = st.find_closest_nucleus(_in_global(sdata, "nucleus_boundaries"), hn)
        closest["nucleus_hn_id"] = closest.index.astype(str)
        closest = closest[closest["distance"] < sl.spec.match_threshold]
        # one HoVerNet nucleus per Xenium nucleus: the nearest
        closest = closest.loc[closest.groupby("nucleus_id")["distance"].idxmin()]
        out = closest.rename(columns={"nucleus_hn_id": "nucleus_id",
                                      "nucleus_id": "xenium_nucleus_id"})
        out = out[["nucleus_id", "xenium_nucleus_id", "distance"]].copy()
        _log(f"{sl.key}: {len(out):,} nuclei matched within "
             f"{sl.spec.match_threshold} px of a Xenium nucleus")

        n2c = nucleus_to_cell(sl, sdata)
        if n2c is None:
            out["cell_id"] = out["xenium_nucleus_id"]
        else:
            out["cell_id"] = out["xenium_nucleus_id"].map(n2c)
            before = len(out)
            out = out[out["cell_id"].notna()]
            _log(f"{sl.key}: {len(out):,} of {before:,} kept after joining the Xenium "
                 f"nucleus to its cell")
    else:  # pragma: no cover
        raise BuildError(f"unknown match route {sl.spec.match!r}")

    out["cell_id"] = out["cell_id"].astype(str)
    out["nucleus_id"] = out["nucleus_id"].astype(str)
    out["cell_type"] = out["cell_id"].map(types).astype(str)
    unknown = out["cell_type"].isin(["nan", "None"])
    if unknown.any():
        raise BuildError(
            f"{sl.key}: {int(unknown.sum()):,} of {len(out):,} matched nuclei have no cell "
            f"type (e.g. cell_id {list(out.loc[unknown, 'cell_id'][:3])}). The annotation "
            "is keyed differently from this slide's cell boundaries.")
    order = {t: i for i, t in enumerate(sl.spec.cell_types)}
    catchall = len(sl.spec.cell_types) + config.XENIUM_TYPE_CATCHALL_OFFSET
    out["cell_type_id"] = out["cell_type"].map(order).fillna(catchall).astype(int)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# spots
# --------------------------------------------------------------------------------------


def build_spots(sl: Slide, sdata, gdf_hn, matched: pd.DataFrame):
    """The brick lattice, the nuclei inside each spot, and the kept spots."""
    import geopandas as gpd
    from shapely.geometry import Point
    from spatialdata import get_extent

    extent = get_extent(sdata)
    x0, x1 = (float(v) for v in extent["x"])
    y0, y1 = (float(v) for v in extent["y"])
    grid = lattice.brick(x0, x1, y0, y1,
                         diameter=sl.spec.diameter, spacing=sl.spec.spacing)
    in_row, staggered = grid.neighbour_distances()
    _log(f"{sl.key}: {len(grid):,} spots on a {grid.kind} lattice "
         f"(d={sl.spec.diameter:g}, pitch={sl.spec.spacing:g}, neighbours "
         f"{in_row:.0f}/{staggered:.0f})")

    radius = grid.radius
    gdf_spots = gpd.GeoDataFrame(
        {"geometry": [Point(x, y).buffer(radius) for x, y in grid.centers]},
        index=pd.Index([str(i) for i in range(len(grid))]))

    nuclei_in_spots = st.match_nuclei_in_spots(gdf_spots, gdf_hn)
    pseudo = matched[matched["nucleus_id"].isin(nuclei_in_spots["nucleus_id"])]
    pseudo = pseudo.merge(nuclei_in_spots[["nucleus_id", "spot_id"]],
                          on="nucleus_id", how="left")

    # the catch-all is not a cell type, so it leaves before the proportions are taken
    pseudo = pseudo[annotation.keep_mask(sl, pseudo["cell_type"])]

    # a spot with one or two nuclei gives a proportion vector that is mostly an accident
    counts = pseudo["spot_id"].value_counts()
    keep = counts[counts >= sl.spec.min_nuclei_per_spot].index
    dropped = len(counts) - len(keep)
    pseudo = pseudo[pseudo["spot_id"].isin(keep)]
    _log(f"{sl.key}: {len(pseudo):,} nuclei in {len(keep):,} spots "
         f"({dropped:,} spots dropped for holding < {sl.spec.min_nuclei_per_spot})")
    return gdf_spots, pseudo.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------


def write_adata(sl: Slide, sdata, gdf_spots, pseudo: pd.DataFrame) -> int:
    """``pseudo_adata_real.h5ad``: the kept spots, in H&E pixel coordinates.

    The lattice was built in the global (micron) frame, so the spots are mapped back
    through the inverse of the H&E transform -- the frame ``pannuke_fast_mask_lvl3_
    annotated.json`` and the embeddings live in.
    """
    from anndata import AnnData
    from spatialdata.transformations import get_transformation

    inv = get_transformation(sdata["he_image"]).inverse()
    local = st.apply_transformation_to_gdf(gdf_spots, _affine(inv.matrix))
    spots_df = st.spot_gdf_to_df(local)
    spots_df = spots_df[spots_df.index.isin(pseudo["spot_id"])]

    adata = AnnData()
    adata.obs.index = spots_df.index
    adata.obs["spot_id"] = spots_df.index
    adata.obsm["spatial"] = spots_df[["x_center", "y_center"]].values.astype(int)
    adata.uns["spatial"] = {sl.spec.name: {"scalefactors": {
        "spot_diameter_fullres": float(spots_df["diameter"].values[0])}}}
    sl.adata.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(sl.adata)
    return len(spots_df)


def write_proportions(sl: Slide, pseudo: pd.DataFrame) -> int:
    counts = pseudo.groupby(["spot_id", "cell_type"]).size().reset_index(name="count")
    total = counts.groupby("spot_id")["count"].transform("sum")
    counts["proportion"] = counts["count"] / total
    props = counts.pivot(index="spot_id", columns="cell_type",
                         values="proportion").fillna(0.0)
    props = props.reindex(columns=list(sl.spec.cell_types), fill_value=0.0)
    props.index.name = None
    props.columns.name = None
    props.to_csv(sl.proportions, index=True)
    return len(props)


def write_ground_truth(sl: Slide, matched: pd.DataFrame,
                       conf: pd.DataFrame | None) -> pd.Index:
    """``sim_<slide>_gt.csv``: one row per matched, callable nucleus.

    Two shapes, because the two slides' annotations are different in kind. Lung's came from
    a label transfer and is **soft**, so each row is that cell's confidence over the seven
    types, left unnormalised where some confidence went to a dropped type. Breast's came
    from a spreadsheet of hard calls, so each row is one-hot. The gridsearch reads either
    with an argmax, but writing a one-hot where the reference has a distribution would be a
    different file that looks just as reasonable -- see
    :func:`hedest_simdata.xenium.annotation.confidence`.
    """
    callable_rows = matched[annotation.keep_mask(sl, matched["cell_type"])]
    if conf is not None:
        gt = conf.reindex(callable_rows["cell_id"].to_numpy())
        gt.index = callable_rows["nucleus_id"].to_numpy()
        gt = gt.fillna(0.0)
    else:
        gt = callable_rows.pivot_table(index="nucleus_id", columns="cell_type",
                                       values="cell_id", aggfunc=lambda x: 1, fill_value=0)
    gt = gt.reindex(columns=list(sl.spec.cell_types), fill_value=0)
    gt.columns.name = None
    gt.index = gt.index.astype(int)
    gt = gt.sort_index()
    gt.index = gt.index.astype(str)
    gt.index.name = "nucleus_id"
    gt.to_csv(sl.ground_truth, index=True)
    return gt.index


def write_annotated_segmentation(sl: Slide, raw: dict, matched: pd.DataFrame) -> dict:
    """``pannuke_fast_mask_lvl3_annotated.json``: HoVerNet nuclei carrying the Xenium type.

    Keys are renumbered 0..n-1 in file order. HoVerNet already writes them that way, so
    this is a safeguard rather than a change, and it is what makes the ids line up with the
    embeddings, which are also numbered by position in this file.
    """
    ids = sorted(raw["nuc"], key=int)
    remap = {old: str(new) for new, old in enumerate(ids)}
    nuc = {remap[k]: dict(v) for k, v in raw["nuc"].items()}

    by_nucleus = matched.set_index("nucleus_id")["cell_type_id"].to_dict()
    unmatched_code = len(sl.spec.cell_types) + config.XENIUM_TYPE_UNMATCHED_OFFSET
    for key, entry in nuc.items():
        entry["type"] = int(by_nucleus.get(key, unmatched_code))

    out = {k: v for k, v in raw.items() if k != "nuc"}
    out["nuc"] = nuc
    sl.annotated_seg.write_text(json.dumps(out))

    codes = pd.Series([e["type"] for e in nuc.values()]).value_counts().sort_index()
    return {int(k): int(v) for k, v in codes.items()}


def write_spot_dict(sl: Slide, pseudo: pd.DataFrame) -> int:
    spot_dict: dict[str, list[str]] = defaultdict(list)
    for spot_id, group in pseudo.groupby("spot_id"):
        spot_dict[str(spot_id)] = [str(n) for n in group["nucleus_id"].tolist()]
    sl.spot_dict.write_text(json.dumps(dict(spot_dict)))
    return len(spot_dict)


# --------------------------------------------------------------------------------------


def _archive_centroids(sdata) -> pd.DataFrame | None:
    """``cell_id -> (x_centroid, y_centroid)`` from the slide's own archive, or None.

    ``cell_circles`` is the vendor's reported centroid per cell, indexed by ``cell_id``;
    that is the quantity the committed annotation stores, so the two are directly
    comparable. Returns None rather than raising when the layer is absent, because the
    check it feeds is an extra guarantee and not a requirement.
    """
    cc = sdata.shapes.get("cell_circles") if hasattr(sdata, "shapes") else None
    if cc is None or not len(cc):
        return None
    return pd.DataFrame({"x_centroid": cc.geometry.x.values,
                         "y_centroid": cc.geometry.y.values},
                        index=pd.Index(cc.index.astype(str), name="cell_id"))


def run(sl: Slide) -> dict:
    """Everything but the embeddings, which need a GPU and come after."""
    sl.out_dir.mkdir(parents=True, exist_ok=True)
    sdata = load_sdata(sl)
    raw, gdf_hn = hovernet_shapes(sl, sdata)

    table = sdata.tables["table"]
    # keyed by obs["cell_id"], NOT obs_names: on breast those differ by one (obs_names are
    # positions '0','1','2'... while cell_id is 1,2,3...), and the cell/nucleus boundary
    # layers are keyed by cell_id. Using obs_names shifts every type by one cell.
    # the centroids come along so the join can be *checked* rather than only performed: an
    # id join reports nothing when ids are reused for different cells, which is what a
    # re-release of the dataset did to the lung annotation
    types = annotation.load(sl, cell_ids=pd.Index(table.obs["cell_id"].astype(str)),
                            centroids=_archive_centroids(sdata))
    kept = annotation.keep_mask(sl, types)
    _log(f"{sl.key}: {len(types):,} annotated cells, {int(kept.sum()):,} of a kept type "
         f"({len(sl.spec.cell_types)} types)")

    matched = match(sl, sdata, gdf_hn, types)
    gdf_spots, pseudo = build_spots(sl, sdata, gdf_hn, matched)

    n_spots = write_adata(sl, sdata, gdf_spots, pseudo)
    n_prop = write_proportions(sl, pseudo)
    gt_index = write_ground_truth(sl, matched, annotation.confidence(
        sl, pd.Index(table.obs['cell_id'].astype(str))))
    codes = write_annotated_segmentation(sl, raw, matched)
    n_dict = write_spot_dict(sl, pseudo)

    if n_spots != n_prop or n_prop != n_dict:
        raise BuildError(
            f"{sl.key}: the spot sets disagree -- adata {n_spots}, proportions {n_prop}, "
            f"spot_dict {n_dict}. They must be the same spots.")

    n_types = len(sl.spec.cell_types)
    expected_gt = sum(v for k, v in codes.items() if k < n_types)
    if len(gt_index) != expected_gt:
        raise BuildError(
            f"{sl.key}: the ground truth has {len(gt_index):,} rows but {expected_gt:,} "
            f"nuclei carry a cell-type code (0..{n_types - 1}). They are the same nuclei "
            "by definition.")

    return {"slide": sl.key, "nuclei": len(raw["nuc"]), "matched": len(matched),
            "spots": n_spots, "nuclei_in_spots": len(pseudo),
            "ground_truth_rows": len(gt_index), "type_codes": codes,
            "cell_types": list(sl.spec.cell_types)}
