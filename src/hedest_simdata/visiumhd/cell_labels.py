"""The committed Visium HD annotation, and the proof that it still fits the segmentation.

Named ``cell_labels`` rather than ``annotations``: every module in this package starts
with ``from __future__ import annotations``, which binds that name in the package
namespace, so ``from hedest_simdata.visiumhd import annotations`` would silently hand
back Python's own future-feature object instead of this module.

The manual cell-type calls are the one thing here no code reproduces, so they ship as
``annotations/visiumhd/<sample>_annotation.csv.gz`` (see ``annotations/
extract_visiumhd.py`` for how they were extracted). Everything else about those cells comes
back from bin2cell.

**The join is by centroid, because ``object_id`` does not survive a re-run.** bin2cell
numbers its cells from the nucleus label image, and this module used to assume that a
re-run of StarDist/HoVerNet/bin2cell would paint the same labels in the same order. The
first build from a clean tree disproved it: the ids joined 281,034 of P2's 283,047
annotated cells, and **0.09%** of those sat where the annotation said, the largest gap
being 26,942 px. The displacements had no structure -- removing their median translation
left a median of 5,302 px -- so the frame had not moved, the numbering had.

The geometry, on the other hand, is exact. Matching each annotated centroid to its nearest
segmented one, in the same ``spatial_cropped_*_buffer`` frame the CSV stores:

===========  ==========================  ======================  ==================
sample       mutual NN within 0.01 px    median distance         5 px-shifted control
===========  ==========================  ======================  ==================
P2           281,185 of 283,047 (99.3%)  0.0004 px (p99 0.0007)  0 of 283,047
P5           178,601 of 178,601 (100%)   0.0004 px (p99 0.0007)  0 of 178,601
===========  ==========================  ======================  ==================

0.0004 px is the CSV's own 3-decimal rounding and nothing else, so these are the *same*
cells: the segmentation reproduces, only its labels are renumbered. The same shape of
problem, and the same fix, as the Xenium lung annotation -- see
``annotations/extract_xenium.py``.

The **shifted control is kept and is now the thing that makes the re-key a proof**: the
whole match is repeated with every annotated centroid displaced by 5 px, and that must
recover nothing. It recovers exactly zero cells on both samples, which is what says the
centroids identify cells rather than merely landing near some cell. ``total_counts``
travels with each row for the same reason and is checked the same way.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

#: Where the committed CSVs live, relative to the repository root.
ANNOTATION_DIR = Path(__file__).resolve().parents[3] / "annotations" / "visiumhd"

#: A centroid must agree to this many pixels. The CSV stores 3 decimals of bin2cell's own
#: float, so anything above rounding is a real disagreement: the measured p99 of a correct
#: match is 0.0007 px, which is exactly that rounding, and the next distance up in the
#: distribution is 26 px.
CENTROID_ATOL = 0.01

#: How far an annotated centroid is displaced for the control, in pixels. Two bin2cell
#: cells are never this close -- 5 px is about a nucleus wide at 0.27 um/px -- so a match
#: that still succeeds after the shift was not identifying cells in the first place.
CONTROL_SHIFT_PX = 5.0

#: Fraction of the *annotated* cells that must be re-keyed, and the ceiling the shifted
#: control must stay under. 0.99 rather than 0.999: P2's fresh segmentation does not
#: reproduce 1,862 of its 283,047 annotated cells (0.66%), which is a property of
#: HoVerNet's own run-to-run variation and not of the join -- P5 reproduces all 178,601.
#: The gap to the control's ceiling is wide on purpose: a correct re-key measures ~0.99+
#: and the control measures 0, so anything in between is something unexplained and should
#: be looked at rather than waved through.
MATCH_MIN = 0.99
CONTROL_MAX = 0.05


class AnnotationError(RuntimeError):
    pass


@dataclass
class Annotation:
    """The frozen labels for one sample, plus what they were extracted alongside."""

    patient: str
    table: pd.DataFrame        # indexed by object_id
    levels: list[str]
    unassigned_types: list[str]

    def __len__(self) -> int:
        return len(self.table)


def path_for(patient: str) -> Path:
    return ANNOTATION_DIR / f"{config.visiumhd_name(patient)}_annotation.csv.gz"


def load(patient: str) -> Annotation:
    """Reads the committed CSV for one patient."""
    p = path_for(patient)
    if not p.exists():
        raise AnnotationError(
            f"{p} is missing. It is committed with the repository; if this is a fresh "
            "checkout, the annotation was not included and the Visium HD family cannot "
            "be built (see annotations/extract_visiumhd.py).")
    df = pd.read_csv(p)
    needed = ["cell_id", "object_id", "x", "y", "total_counts", *config.VISIUMHD_LEVELS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise AnnotationError(f"{p.name} lacks columns {missing}")
    if df["object_id"].duplicated().any():
        raise AnnotationError(f"{p.name}: object_id is not unique, so it is not a key")
    return Annotation(patient=patient, table=df.set_index("object_id"),
                      levels=list(config.VISIUMHD_LEVELS),
                      unassigned_types=list(
                          config.VISIUMHD_UNASSIGNED_TYPES.get(patient, [])))


def _mutual_match(annotated_xy: np.ndarray, segmented_xy: np.ndarray,
                  atol: float) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest neighbours within ``atol``: ``(annotated row, segmented row)``.

    Mutual, not merely nearest: a one-sided nearest neighbour can send two annotated
    centroids to the same cell, and would then label one of them from the other's row.
    Measured on P2 and P5 the accepted matches collide zero times, which is what makes the
    mutual test cost nothing here and worth keeping anyway.
    """
    from scipy.spatial import cKDTree

    forward = cKDTree(segmented_xy).query(annotated_xy, k=1)
    back = cKDTree(annotated_xy).query(segmented_xy, k=1)[1]
    a_rows = np.arange(len(annotated_xy))
    ok = (forward[0] <= atol) & (back[forward[1]] == a_rows)
    return a_rows[ok], forward[1][ok]


def join(ann: Annotation, object_id: np.ndarray, centroid_xy: np.ndarray,
         total_counts: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    """Re-keys the frozen labels onto freshly segmented cells, and proves the re-key.

    Args:
        ann: The committed annotation.
        object_id: bin2cell ``object_id`` of each segmented cell.
        centroid_xy: ``(N, 2)`` centroids in the capture-image pixel frame -- the same
            frame the CSV's ``x``/``y`` are in (``spatial_cropped_*_buffer``).
        total_counts: Per-cell totals, when available, for the extra check.

    Returns:
        ``(labels, report)``. ``labels`` carries ``cell_id``, the level columns and
        ``total_counts``, indexed by **this build's** ``object_id`` -- so the caller needs
        no second lookup in the annotation's own id space, which no longer means anything.
        ``report`` records the re-key and its control.

    Raises:
        AnnotationError: if the re-key or its control does not behave.

    Note:
        Deciding which categories count as unassigned is **not** done here --
        :func:`hedest_simdata.visiumhd.labels.build` and ``labels.unassigned_mask`` do it,
        as they did for the reference datasets. Two implementations of that rule would
        drift, and theirs is the one the published data came out of.
    """
    if len(object_id) != len(centroid_xy):
        raise AnnotationError(
            f"{ann.patient}: {len(object_id)} object_ids but {len(centroid_xy)} centroids")

    a_xy = ann.table[["x", "y"]].to_numpy()
    a_rows, s_rows = _mutual_match(a_xy, centroid_xy, CENTROID_ATOL)
    agree = float(len(a_rows) / len(ann)) if len(ann) else 0.0

    # the control: the same match with every annotated centroid displaced, which must
    # recover nothing. This replaced a shifted-*id* control, which the re-key made
    # meaningless -- the ids are no longer the thing being trusted.
    ctrl_rows, _ = _mutual_match(a_xy + CONTROL_SHIFT_PX, centroid_xy, CENTROID_ATOL)
    control = float(len(ctrl_rows) / len(ann)) if len(ann) else 0.0

    d = np.linalg.norm(a_xy[a_rows] - centroid_xy[s_rows], axis=1) if len(a_rows) else \
        np.zeros(0)
    # how far the ids have drifted, reported because it is the reason this is a re-key
    id_kept = float(np.mean(ann.table.index.to_numpy()[a_rows] == object_id[s_rows])) \
        if len(a_rows) else 0.0
    report = {
        "patient": ann.patient,
        "segmented_cells": int(len(object_id)),
        "annotated_cells": len(ann),
        "rekeyed": int(len(a_rows)),
        "centroid_agreement": round(agree, 6),
        "shifted_centroid_control": round(control, 6),
        "max_centroid_distance_px": round(float(d.max()) if len(d) else 0.0, 4),
        "object_id_unchanged": round(id_kept, 6),
    }
    if total_counts is not None:
        same = np.isclose(total_counts[s_rows],
                          ann.table["total_counts"].to_numpy()[a_rows],
                          rtol=1e-6, atol=1e-6)
        report["total_counts_agreement"] = round(float(same.mean()), 6)

    if agree < MATCH_MIN:
        raise AnnotationError(
            f"{ann.patient}: only {agree:.4%} of the {len(ann):,} annotated cells found a "
            f"segmented cell within {CENTROID_ATOL} px. The segmentation has moved "
            f"relative to the frozen labels, not merely renumbered; do not use this "
            f"build. Report: {report}")
    if control > CONTROL_MAX:
        raise AnnotationError(
            f"{ann.patient}: the {CONTROL_SHIFT_PX:g} px-shifted control also matches "
            f"({control:.4%}), so the centroids are not identifying individual cells and "
            f"the re-key proves nothing. Report: {report}")

    rows = ann.table.iloc[a_rows]
    labels = rows[ann.levels].copy()
    labels.insert(0, "cell_id", rows["cell_id"].to_numpy())
    labels["total_counts"] = rows["total_counts"].to_numpy()
    labels.index = pd.Index(object_id[s_rows], name="object_id")
    return labels, report
