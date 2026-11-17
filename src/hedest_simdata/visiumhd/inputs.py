"""Loading the inputs, and checking that they belong together.

Frames: HoVerNet ran on the capture-area pyramidal TIFF, so nucleus coordinates
are level-0 pixels of that image. bin2cell's ``spatial_cropped_<buffer>_buffer``
coordinates (cells and bins) are in the same frame; ``spatial`` is the full
``.btf`` frame. The two differ by a constant crop offset.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import hovernet, slide

from .config import MPP_TOLERANCE, REPO, Sample, log


class InputError(RuntimeError):
    pass


def check(cid: str, name: str, passed: bool, value=None, rule: str = "", hard: bool = True, **details) -> dict:
    return {"id": cid, "name": name, "passed": bool(passed), "value": value, "rule": rule,
            "hard": hard, **details}


# ---------------------------------------------------------------------------
# annotated cells
# ---------------------------------------------------------------------------
@dataclass
class Cells:
    ids: np.ndarray                 # obs names (str), file order
    object_id: np.ndarray           # int
    labels: pd.DataFrame            # level columns, index = ids
    total_counts: np.ndarray
    spatial: np.ndarray             # full-image px
    spatial_crop: np.ndarray        # capture-image px
    crop_key: str
    preprocessing: dict
    var_names: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)


def _read_obs_column(g) -> np.ndarray:
    """One obs column, categorical or plain."""
    if isinstance(g, h5py.Group):                       # categorical
        cats = g["categories"][:]
        cats = np.asarray([c.decode() if isinstance(c, bytes) else c for c in cats], dtype=object)
        codes = g["codes"][:]
        out = np.where(codes >= 0, cats[np.clip(codes, 0, None)], None)
        return out
    v = g[:]
    return np.asarray([x.decode() if isinstance(x, bytes) else x for x in v], dtype=object) \
        if v.dtype.kind in "OS" else v


def load_cells(s: Sample) -> Cells:
    """The annotated cells: geometry from bin2cell, cell types from the committed CSV.

    The split is the point. Everything measurable about a cell -- its ``object_id``, its
    centroid in both frames, its counts -- is re-derived from ``cell_data.h5ad``, which
    this pipeline produced. Only the **cell types** are frozen, because they were assigned
    by hand, and they arrive from ``annotations/visiumhd/`` re-keyed onto this build's
    cells by centroid; ``object_id`` is not a key across runs (see
    :mod:`hedest_simdata.visiumhd.cell_labels`).

    ``total_counts`` is taken from the **CSV**, not from ``cell_data``, deliberately: check
    C5 in :mod:`hedest_simdata.visiumhd.match` compares it against bin2cell's own row sums,
    and that comparison is only a proof while the two sides come from different places.
    Reading both from ``cell_data`` would turn it into a tautology that passes however
    badly the cells have drifted.

    Read with h5py rather than AnnData: only obs, obsm, var and the sparse X are needed,
    and going through AnnData would pull in uns entries this environment cannot always
    decode.
    """
    from . import cell_labels

    ann = cell_labels.load(paths_patient(s))

    with h5py.File(s.cell_data, "r") as f:
        obs = f["obs"]
        idx_key = obs.attrs.get("_index", "_index")
        all_ids = np.asarray([x.decode() if isinstance(x, bytes) else x
                              for x in obs[idx_key][:]], dtype=str)
        if "object_id" not in obs:
            raise InputError(f"{s.cell_data.name} has no obs['object_id']")
        all_oid = obs["object_id"][:].astype(np.int64)
        crop_keys = [k for k in f["obsm"].keys()
                     if re.fullmatch(r"spatial_cropped_\d+_buffer", k)]
        if len(crop_keys) != 1:
            raise InputError(
                f"expected one spatial_cropped_*_buffer key in obsm, found {crop_keys}")
        all_full = f["obsm/spatial"][:].astype(np.float64)
        all_crop = f[f"obsm/{crop_keys[0]}"][:].astype(np.float64)
        var = f["var"]
        vidx = var.attrs.get("_index", "_index")
        var_names = np.asarray([x.decode() if isinstance(x, bytes) else x
                                for x in var[vidx][:]], dtype=str)

    # the annotated cells are a QC-filtered subset of what bin2cell produced, and the CSV
    # is what defines the subset -- re-running QC would make its thresholds a second
    # irreproducible choice
    labels, report = cell_labels.join(ann, all_oid, all_crop)
    keep = np.isin(all_oid, labels.index.to_numpy())
    log(f"cells: {keep.sum():,} annotated of {len(all_oid):,} segmented; "
        f"centroid agreement {report['centroid_agreement']:.6f}, "
        f"shifted-centroid control {report['shifted_centroid_control']:.6f}, "
        f"object_id unchanged {report['object_id_unchanged']:.6f}")

    oid = all_oid[keep]
    # `labels` is keyed by THIS build's object_id. Looking rows up in `ann.table` instead
    # would use the annotation's own numbering, which is the retired one the re-key exists
    # to get away from -- and on P2 that numbering mislabels 99.9% of the cells.
    rows = labels.loc[oid]
    return Cells(
        ids=all_ids[keep],
        object_id=oid,
        labels=pd.DataFrame({c: rows[c].to_numpy() for c in s.levels},
                            index=pd.Index(all_ids[keep])),
        total_counts=rows["total_counts"].to_numpy().astype(np.float64),
        spatial=all_full[keep],
        spatial_crop=all_crop[keep],
        crop_key=crop_keys[0],
        preprocessing={"input": str(s.cell_data), "segmentation": "hovernet",
                       "labels": str(cell_labels.path_for(paths_patient(s))),
                       "join": report},
        var_names=var_names,
    )


def paths_patient(s: Sample) -> str:
    """``P1`` from ``Visium_HD_Human_Colon_Cancer_P1``; here to avoid a circular import."""
    return s.name.rsplit("_", 1)[-1]


def marker_expression(s: Sample, genes: list[str],
                      annotated: np.ndarray) -> pd.DataFrame:
    """Normalised expression of ``genes``, one row per annotated cell.

    Recomputed -- ``normalize_total`` to the median then ``log1p`` -- from the raw counts
    in ``cell_data.h5ad``, restricted to the annotated cells first so the median is taken
    over the same population the original was. It feeds check C6, a marker-score sanity
    test with generous thresholds, so recomputing rather than reading a stored normalised
    matrix is fine; it is not used for any published number.

    Args:
        annotated: the ``object_id``s of the annotated cells **in this build's numbering**,
            i.e. ``Cells.object_id``. This used to read them from the annotation itself,
            which is the retired numbering: on P2 that selected the wrong 283,047 cells of
            288,724 and the median was taken over them, so a check meant to be independent
            quietly agreed with whatever the mislabelling produced.
    """
    with h5py.File(s.cell_data, "r") as f:
        var = f["var"]
        names = np.asarray([x.decode() if isinstance(x, bytes) else x
                            for x in var[var.attrs.get("_index", "_index")][:]], dtype=str)
        keep = [g for g in genes if g in set(names)]
        if not keep:
            return pd.DataFrame()
        cols = pd.Series(np.arange(len(names)), index=names).loc[keep].to_numpy()
        obs = f["obs"]
        ids = np.asarray([x.decode() if isinstance(x, bytes) else x
                          for x in obs[obs.attrs.get("_index", "_index")][:]], dtype=str)
        oid = obs["object_id"][:].astype(np.int64)
        g = f["X"]
        if isinstance(g, h5py.Group):
            X = sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                              shape=tuple(g.attrs["shape"]))
            totals = np.asarray(X.sum(axis=1)).ravel()
            sub = X[:, cols].toarray()
        else:
            full = g[:]
            totals = full.sum(axis=1)
            sub = full[:, cols]

    # restrict to the annotated cells before normalising, so the median is taken over the
    # same population as it was originally
    rows = np.isin(oid, annotated)
    sub, totals, ids = sub[rows], totals[rows], ids[rows]
    scale = np.median(totals[totals > 0]) if (totals > 0).any() else 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(totals[:, None] > 0, sub / totals[:, None] * scale, 0.0)
    return pd.DataFrame(np.log1p(norm), columns=keep, index=ids)


# ---------------------------------------------------------------------------
# mpp
# ---------------------------------------------------------------------------
def read_mpp(s: Sample) -> dict:
    """mpp from spaceranger (used), cross-checked against the scanner and the 2 um grid."""
    sf = json.loads(Path(s.scalefactors).read_text())
    used = float(sf["microns_per_pixel"])
    out = {"used": used, "source": "spaceranger scalefactors_json.json", "spaceranger": used}
    if s.full_image is not None and Path(s.full_image).exists():
        import tifffile
        with tifffile.TiffFile(s.full_image) as t:
            desc = t.pages[0].description or ""
        m = re.search(r'PhysicalSizeX="([\d.eE+-]+)"', desc)
        out["scanner"] = float(m.group(1)) if m else None
    tp_path = Path(s.scalefactors).parent / "tissue_positions.parquet"
    if tp_path.exists():
        tp = pd.read_parquet(tp_path, columns=["array_row", "array_col",
                                               "pxl_row_in_fullres", "pxl_col_in_fullres"])
        A = np.c_[tp.array_col, tp.array_row, np.ones(len(tp))]
        cx, *_ = np.linalg.lstsq(A, tp.pxl_col_in_fullres.to_numpy(), rcond=None)
        cy, *_ = np.linalg.lstsq(A, tp.pxl_row_in_fullres.to_numpy(), rcond=None)
        out["bin_grid"] = float(sf.get("bin_size_um", 2.0)) / float(np.hypot(cx[0], cy[0]))
    vals = [v for k, v in out.items() if k in ("spaceranger", "scanner", "bin_grid") and v]
    out["spread"] = max(vals) / min(vals) - 1.0
    if out["spread"] > MPP_TOLERANCE:
        raise InputError(f"mpp sources disagree by {100 * out['spread']:.2f}%: {out}")
    return out


# ---------------------------------------------------------------------------
# nuclei
# ---------------------------------------------------------------------------
def load_nuclei(s: Sample) -> tuple[int | None, dict]:
    seg = hovernet.load(s.hovernet_json)
    return seg.get("mag"), seg["nuc"]


def load_nucleus_table(s: Sample, nuc: dict) -> tuple[pd.DataFrame, str]:
    """label -> HoVerNet key + centroid.

    The table written when the label image was painted is the reference. Without
    it, the painting rule (label = position in the JSON + 1) is assumed, and the
    label-image check proves or disproves it.
    """
    if Path(s.nucleus_table).exists():
        t = pd.read_csv(s.nucleus_table)
        t = pd.DataFrame({"key": t["hovernet_id"].astype(str).to_numpy(),
                          "x": t["x"].to_numpy(), "y": t["y"].to_numpy()},
                         index=pd.Index(t["label"].astype(np.int64), name="label"))
        return t, "nucleus_types.csv"
    keys = list(nuc)
    xy = hovernet.centroids(nuc, keys)
    t = pd.DataFrame({"key": keys, "x": xy[:, 0], "y": xy[:, 1]},
                     index=pd.Index(np.arange(1, len(keys) + 1), name="label"))
    log("WARNING: no nucleus table, assuming label = JSON position + 1 (checked below)")
    return t, "assumed (JSON order + 1)"


def load_label_mask(s: Sample) -> sp.csr_matrix:
    return sp.load_npz(s.label_mask).tocsr()


# ---------------------------------------------------------------------------
# bins and bin2cell cells
# ---------------------------------------------------------------------------
@dataclass
class Bins:
    array_row: np.ndarray
    array_col: np.ndarray
    labels_joint: np.ndarray
    xy_crop: np.ndarray             # capture-image px
    xy_full: np.ndarray             # full-image px
    total_destriped: np.ndarray
    uns: dict


def _h5_str(v):
    v = v[()] if isinstance(v, h5py.Dataset) else v
    return v.decode() if isinstance(v, bytes) else v


def load_bins(s: Sample, crop_key: str) -> Bins:
    with h5py.File(s.bin_data, "r") as f:
        uns = {}
        b2c = f.get("uns/bin2cell")
        if b2c is not None:
            if "labels_npz_paths" in b2c:
                uns["labels_npz_paths"] = {k: _h5_str(b2c["labels_npz_paths"][k]) for k in b2c["labels_npz_paths"]}
            if "hovernet" in b2c:
                uns["hovernet"] = {k: _h5_str(b2c["hovernet"][k]) for k in b2c["hovernet"]}
        return Bins(
            array_row=f["obs/array_row"][:].astype(np.int64),
            array_col=f["obs/array_col"][:].astype(np.int64),
            labels_joint=f["obs/labels_joint"][:].astype(np.int64),
            xy_crop=f[f"obsm/{crop_key}"][:].astype(np.float64),
            xy_full=f["obsm/spatial"][:].astype(np.float64),
            total_destriped=f["obs/n_counts_adjusted"][:].astype(np.float64),
            uns=uns,
        )


def load_bin_matrix(s: Sample) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """The filtered, destriped bins x genes matrix and its gene table."""
    with h5py.File(s.bin_data, "r") as f:
        g = f["X"]
        shape = tuple(g.attrs["shape"])
        X = sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
        var = {}
        idx_key = f["var"].attrs.get("_index", "_index")
        names = f["var"][idx_key][:].astype(str)
        for k in ("gene_ids", "feature_types"):
            if k in f["var"]:
                node = f["var"][k]
                if isinstance(node, h5py.Group):      # categorical
                    cats = node["categories"][:].astype(str)
                    var[k] = cats[node["codes"][:]]
                else:
                    var[k] = node[:].astype(str)
    return X, pd.DataFrame(var, index=pd.Index(names, name="gene"))


def cell_totals(s: Sample) -> pd.Series:
    """bin2cell's per-cell count totals, keyed by object_id."""
    with h5py.File(s.cell_data, "r") as f:
        ids = f["obs/object_id"][:].astype(np.int64)
        ip = f["X/indptr"][:]
        data = f["X/data"][:]
    rows = np.repeat(np.arange(len(ids)), np.diff(ip))
    return pd.Series(np.bincount(rows, weights=data, minlength=len(ids)), index=ids)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def _same(a, b, tail: int = 3) -> bool:
    """Do two recorded paths point at the same file?

    Exact match after resolving, or -- when the inputs were produced on another
    machine, so the recorded root differs -- the same trailing path components.
    What the files actually contain is checked by C1-C5, not here.
    """
    if a is None or b is None:
        return False
    pa = Path(a)
    pa = pa if pa.is_absolute() else REPO / pa
    if pa.resolve() == Path(b).resolve():
        return True
    n = min(tail, len(pa.parts), len(Path(b).parts))
    return pa.parts[-n:] == Path(b).parts[-n:]


def provenance(s: Sample, cells: Cells, nuc: dict, bins: Bins, mask_shape, slide_wh
               ) -> tuple[list[dict], np.ndarray]:
    """Do the inputs belong together?

    Three of the five checks this had in VisiumHD_analysis compared recorded path strings
    and file mtimes, because there the segmentation, the bins and the annotation were
    produced by separate manual runs that could silently go out of sync. Here one pipeline
    writes them in a declared order, so those checks would only ever confirm their own
    construction. They are replaced by the one thing that *can* still be wrong and matters:
    whether the frozen cell types still belong to this segmentation, which is P1 below and
    is a measurement, not a path comparison.
    """
    from . import cell_labels

    out = []
    join = cells.preprocessing.get("join", {})
    out.append(check(
        "P1", "the committed cell types belong to this segmentation",
        bool(join)
        # 0.99, not 1.0: P2's fresh segmentation does not reproduce 1,862 of its 283,047
        # annotated cells. `cell_labels.MATCH_MIN` is the single definition of that floor,
        # and this check reads it rather than restating it -- two numbers for one rule is
        # how a loosened threshold stays loose in one place and strict in the other.
        and join.get("centroid_agreement", 0.0) >= cell_labels.MATCH_MIN
        and join.get("shifted_centroid_control", 1.0) <= cell_labels.CONTROL_MAX,
        value={k: join.get(k) for k in
               ("rekeyed", "annotated_cells", "segmented_cells", "centroid_agreement",
                "shifted_centroid_control", "max_centroid_distance_px",
                "object_id_unchanged")},
        rule="the annotated cells are re-keyed onto cells at the same centroids, and the "
             "shifted-centroid control recovers none"))
    hn = bins.uns.get("hovernet", {})
    out.append(check(
        "P2", "bin2cell labelled the bins from the same HoVerNet JSON this stage read",
        int(hn.get("n_json", -1)) == len(nuc),
        value={"n_json": hn.get("n_json"), "nuclei_in_json": len(nuc)}))
    W, H = slide_wh
    out.append(check("P3", "label image has the size of the capture image",
                     tuple(mask_shape) == (H, W),
                     value={"mask": list(mask_shape), "image_hw": [H, W]}))
    off_cells = cells.spatial - cells.spatial_crop
    off_bins = bins.xy_full - bins.xy_crop
    offset = np.median(off_cells, axis=0)
    dev = max(np.abs(off_cells - offset).max(), np.abs(off_bins - offset).max())
    out.append(check("P4", "one constant crop offset for cells and bins", dev < 1e-3,
                     value={"offset_xy": offset.tolist(), "max_deviation_px": float(dev)}))
    return out, offset
