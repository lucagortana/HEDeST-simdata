"""Where each Xenium slide's cell types come from, and the check that they still fit.

The two slides differ, so they are handled separately rather than behind a flag:

* **Lung** -- a KNN label transfer against a single-cell lung atlas. Reproducing it would
  mean shipping the reference and making every user run the transfer for an input that is
  the same every time, so the result is committed as
  ``annotations/xenium/<slide>_annotation.csv.gz``. Keyed by the Xenium ``cell_id`` *and*
  carrying each cell's centroid -- which is the part that took a failed build to learn. The
  barcode is stable across re-*runs* of the vendor's pipeline but **not across re-releases
  of the dataset**: 10x re-published this slide with a different ``cell_id`` space, and the
  annotation as first committed joined 8 of 162,254 cells against the current download. The
  centroids are bit-identical between the two releases, so the join is checked against them
  by :func:`_check_centroids` rather than trusted.
* **Breast** -- a spreadsheet published on the dataset page, read directly. Its clusters
  are collapsed into working types and anything still under
  :data:`~hedest_simdata.config.XENIUM_BREAST_MIN_CELLS` cells is folded into
  ``Unlabeled``.

Both end at the same place: a ``cell_id -> cell_type`` series over the slide's own cells,
whose set of types, after the drops, must be exactly ``spec.cell_types``. That last part is
checked, not assumed, because the breast rule is count-based and so depends on the data it
runs over.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from .paths import Slide

#: Where the committed CSVs live, relative to the repository root.
ANNOTATION_DIR = Path(__file__).resolve().parents[3] / "annotations" / "xenium"


class AnnotationError(RuntimeError):
    pass


def committed_path(slide_name: str) -> Path:
    return ANNOTATION_DIR / f"{slide_name}_annotation.csv.gz"


def load(sl: Slide, cell_ids: pd.Index,
         centroids: pd.DataFrame | None = None) -> pd.Series:
    """``cell_id -> cell_type`` over the slide's own cells, before any type is dropped.

    **The two slides are keyed differently, and it is not a choice.** Lung's Xenium output
    carries barcodes (``aaaadpbp-1``), so the committed CSV joins on them -- and then has
    the join confirmed against the centroids, because the barcodes changed once already
    between releases of this dataset. Breast is from an earlier Xenium release whose output
    has **no barcodes at all** -- its ``obs_names`` are ``'0', '1', '2'`` and its
    ``cell_id`` column is ``1, 2, 3`` -- so its spreadsheet cannot be joined by key, and the
    published dataset was built by assigning the spreadsheet's rows to the table's cells
    **positionally**. That is reproduced here, with the length checked exactly, because a
    positional assignment that is one row off is wrong everywhere and looks fine.

    Args:
        sl: The slide.
        cell_ids: The ids the Xenium table holds, in table order (which matters for the
            positional case).
        centroids: Optional ``cell_id -> (x_centroid, y_centroid)`` from the slide's own
            archive. When given, and when the committed file carries centroids, the two are
            required to agree -- which is what distinguishes "these ids join" from "these
            ids join to the right cells".
    """
    cell_ids = pd.Index(cell_ids.astype(str))
    if sl.spec.annotation == "committed":
        df = _committed_frame(sl)
        types = df.set_index("cell_id")["cell_type"].astype(str)
        missing = cell_ids.difference(types.index)
        if len(missing):
            raise AnnotationError(
                f"{sl.key}: {len(missing):,} of {len(cell_ids):,} Xenium cells have no "
                f"cell type (e.g. {list(missing[:3])}). The committed annotation does not "
                "match this slide's cells; do not build from it. If the ids look like a "
                "different barcode space rather than a few strays, the dataset has probably "
                "been re-released -- annotations/extract_xenium.py re-keys by centroid.")
        _check_centroids(sl, df, centroids)
        types = types.reindex(cell_ids)
    elif sl.spec.annotation == "vendor":
        types = _vendor(sl, cell_ids)
    else:  # pragma: no cover - the registry only has the two
        raise AnnotationError(f"unknown annotation source {sl.spec.annotation!r}")

    check_types(sl, types)
    return types


#: How far a committed centroid may sit from the archive's own, in microns. The committed
#: values are the archive's, rounded to 4 decimals, so the only honest difference is that
#: rounding -- measured at 7.1e-5 um worst case. A cell is several microns across, so this
#: is tight enough that a one-cell-off join cannot pass.
CENTROID_TOL_UM = 0.01


def _check_centroids(sl: Slide, df: pd.DataFrame,
                     centroids: pd.DataFrame | None) -> None:
    """Confirms the ids joined to the cells they were meant to, not merely to some cells.

    An id join reports nothing when ids are *reused* for different cells, which is exactly
    what a dataset re-release can do. Comparing the committed centroid against the one in
    the archive catches that, and costs a subtraction.

    Silent when either side lacks centroids: the breast slide does not use this path, and
    an annotation committed before the centroid columns existed should still load.
    """
    if centroids is None or "x_centroid" not in df.columns:
        return
    have = df.set_index("cell_id")[["x_centroid", "y_centroid"]]
    both = have.join(centroids.rename(columns={"x_centroid": "ax", "y_centroid": "ay"}),
                     how="inner")
    if both.empty:
        raise AnnotationError(
            f"{sl.key}: no cell id is common to the committed annotation and the archive's "
            "centroids, so the join cannot be checked")
    d = ((both["x_centroid"] - both["ax"]) ** 2
         + (both["y_centroid"] - both["ay"]) ** 2) ** 0.5
    if float(d.max()) > CENTROID_TOL_UM:
        worst = d.idxmax()
        raise AnnotationError(
            f"{sl.key}: the committed annotation's ids join, but to the wrong cells -- "
            f"{int((d > CENTROID_TOL_UM).sum()):,} of {len(both):,} centroids disagree by "
            f"more than {CENTROID_TOL_UM} um (worst {float(d.max()):.3f} um at {worst}). "
            "The ids are being reused for different cells; re-key the annotation rather "
            "than building from it.")


def _committed_frame(sl: Slide) -> pd.DataFrame:
    p = committed_path(sl.spec.name)
    if not p.exists():
        raise AnnotationError(
            f"{p} is missing. The lung cell types are committed with the repository "
            "(annotations/extract_xenium.py made them); without it this slide cannot be "
            "built, since reproducing the label transfer needs a single-cell reference "
            "this repo does not ship.")
    df = pd.read_csv(p)
    for col in ("cell_id", "cell_type"):
        if col not in df.columns:
            raise AnnotationError(f"{p.name} lacks a {col!r} column")
    if df["cell_id"].duplicated().any():
        raise AnnotationError(f"{p.name}: cell_id is not unique")
    df["cell_id"] = df["cell_id"].astype(str)
    return df


def _vendor(sl: Slide, cell_ids: pd.Index) -> pd.Series:
    """Breast: the published spreadsheet, collapsed, thresholded and aligned positionally."""
    if not sl.annotation_xlsx.exists():
        raise AnnotationError(
            f"{sl.annotation_xlsx} is missing. It is on the dataset page and carries this "
            "slide's published cell types, which this repo reads rather than recomputing.")
    raw = pd.read_excel(sl.annotation_xlsx, sheet_name=config.XENIUM_BREAST_SHEET)
    for col in ("Barcode", "Cluster"):
        if col not in raw.columns:
            raise AnnotationError(
                f"{sl.annotation_xlsx.name}: sheet {config.XENIUM_BREAST_SHEET!r} has no "
                f"{col!r} column (found {list(raw.columns)[:6]})")
    if len(raw) != len(cell_ids):
        raise AnnotationError(
            f"{sl.key}: the spreadsheet has {len(raw):,} rows and the Xenium table "
            f"{len(cell_ids):,} cells. They are aligned by position, because this release "
            "of the Xenium output carries no barcodes to join on, so the counts have to "
            "match exactly. Check that the sheet is "
            f"{config.XENIUM_BREAST_SHEET!r} and the output bundle is this slide's.")

    types = raw["Cluster"].map(config.XENIUM_BREAST_CLUSTER_MAP).fillna(raw["Cluster"])
    # a type too rare to learn is not a type; the published dataset folded those into
    # Unlabeled, which is then dropped with it
    counts = types.value_counts()
    rare = sorted(counts[counts < config.XENIUM_BREAST_MIN_CELLS].index)
    if rare:
        print(f"  {sl.key}: folding into Unlabeled, under "
              f"{config.XENIUM_BREAST_MIN_CELLS} cells: "
              + ", ".join(f"{t} ({counts[t]:,})" for t in rare), flush=True)
    types = types.where(~types.isin(set(rare)), "Unlabeled")
    return pd.Series(types.to_numpy().astype(str), index=cell_ids).rename("cell_type")


def check_types(sl: Slide, types: pd.Series) -> None:
    """After the drops, the remaining types must be exactly the delivered ones."""
    present = set(types.unique()) - set(sl.spec.drop_types)
    want = set(sl.spec.cell_types)
    if present != want:
        raise AnnotationError(
            f"{sl.key}: after dropping {list(sl.spec.drop_types)} the annotation leaves "
            f"{sorted(present)}, but the delivered dataset has {sorted(want)}.\n"
            f"  only in the annotation: {sorted(present - want)}\n"
            f"  only in the dataset:    {sorted(want - present)}\n"
            "The published files define the contract -- their column order is zipped "
            "against the palette in HEDeST's runs.py, which raises on a mismatch.")


def keep_mask(sl: Slide, types: pd.Series) -> pd.Series:
    """True for cells whose type is one of the delivered ones."""
    return types.isin(list(sl.spec.cell_types))


#: Prefix of the per-type confidence columns in a committed annotation CSV.
CONFIDENCE_PREFIX = "conf:"


def confidence(sl: Slide, cell_ids: pd.Index) -> pd.DataFrame | None:
    """``cell_id x cell_type`` label-transfer confidence, or None when there is none.

    Lung's delivered ground truth is **soft**: it carries the label transfer's confidence
    over cell types rather than a one-hot. Only 60.2% of its 107,429 rows are one-hot, and
    the row sums run from 0.35 to 1.0 -- confidence that went to a dropped type is left out
    and *not* redistributed, which is why the rows do not all sum to one. Rebuilding that
    file from an argmax would produce something subtly different and plausible, so the
    confidence columns are committed and used directly.

    Breast has no confidence: its types came from a published spreadsheet of hard calls,
    and its ground truth is one-hot.
    """
    if sl.spec.annotation != "committed":
        return None
    df = pd.read_csv(committed_path(sl.spec.name))
    cols = [c for c in df.columns if c.startswith(CONFIDENCE_PREFIX)]
    if not cols:
        return None
    conf = df.set_index(df["cell_id"].astype(str))[cols]
    conf.columns = [c[len(CONFIDENCE_PREFIX):] for c in cols]
    missing = [t for t in sl.spec.cell_types if t not in conf.columns]
    if missing:
        raise AnnotationError(
            f"{sl.key}: the committed confidence columns lack {missing}; it must cover "
            f"every delivered cell type, in the delivered order {list(sl.spec.cell_types)}")
    return conf.reindex(columns=list(sl.spec.cell_types)).reindex(
        pd.Index(cell_ids.astype(str)))
