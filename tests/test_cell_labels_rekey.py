"""The Visium HD annotation re-key, on synthetic cells where the answer is known.

The bug this guards against produced no error of its own. ``object_id`` was used as the
key between the committed cell types and freshly segmented cells; a re-run of
StarDist/HoVerNet/bin2cell paints the same cells but numbers them differently, so the join
succeeded, every output file was well-formed, and 99.9% of P2's cells carried another
cell's type. What caught it was the centroid each row also carries -- and a real build is
the wrong place to test that, because it takes an hour of GPU and a 7 GB intermediate to
reach. Synthetic cells reach it in a second.

Four cases, each the sort of thing that actually happened or could:

* **renumbered** -- the real failure: same cells, permuted ids. Must re-key completely.
* **translated** -- the other candidate the first probe had to rule out. Must be refused,
  because a moved frame means the segmentation no longer corresponds and no amount of
  nearest-neighbour matching makes it correspond.
* **partly missing** -- a few annotated cells the new segmentation did not find, which is
  P2's real 0.66%. Must be tolerated, and reported, not refused.
* **jittered** -- every centroid moved a little. Must be refused: "near a cell" is not
  "is that cell", and this is exactly what the shifted control exists to detect.

    python tests/test_cell_labels_rekey.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hedest_simdata import config                                      # noqa: E402
from hedest_simdata.visiumhd import cell_labels as cl                  # noqa: E402

FAILURES: list[str] = []
RNG = np.random.default_rng(0)
N = 2_000


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def _annotation(n: int = N) -> tuple[cl.Annotation, np.ndarray]:
    """``n`` cells on a jittered grid, with ids that are not their position."""
    side = int(np.ceil(np.sqrt(n)))
    gx, gy = np.meshgrid(np.arange(side) * 20.0, np.arange(side) * 20.0)
    xy = np.c_[gx.ravel(), gy.ravel()][:n]
    xy = xy + RNG.uniform(-3, 3, xy.shape)
    # the CSV stores 3 decimals, and the tolerance is set against exactly that rounding
    xy = np.round(xy, 3)
    ids = np.arange(1, n + 1) * 3            # sparse, like real object_ids
    table = pd.DataFrame({
        "cell_id": [f"c{i}" for i in ids],
        "x": xy[:, 0], "y": xy[:, 1],
        "total_counts": RNG.integers(50, 5000, n).astype(float),
        **{lv: RNG.choice(["A", "B", "C"], n) for lv in config.VISIUMHD_LEVELS},
    }, index=pd.Index(ids, name="object_id"))
    return cl.Annotation(patient="T", table=table, levels=list(config.VISIUMHD_LEVELS),
                         unassigned_types=[]), xy


def main() -> int:
    ann, xy = _annotation()
    types = ann.table[config.VISIUMHD_LEVELS[-1]].to_numpy()

    print("\nrenumbered: the same cells, ids reassigned (the failure this replaced)")
    order = RNG.permutation(len(xy))                 # the segmentation's own row order
    seg_oid = (np.arange(len(xy)) + 7) * 5           # a numbering unrelated to the CSV's
    labels, rep = cl.join(ann, seg_oid, xy[order], total_counts=None)
    check("every annotated cell is re-keyed", rep["rekeyed"] == len(xy),
          f"{rep['rekeyed']:,}/{len(xy):,}")
    check("centroid agreement is 1.0", rep["centroid_agreement"] == 1.0)
    check("the shifted-centroid control recovers nothing",
          rep["shifted_centroid_control"] == 0.0)
    check("it reports that the ids did not survive",
          rep["object_id_unchanged"] < 0.01, f"{rep['object_id_unchanged']}")
    # the real test: each segmented cell carries the type of the cell AT ITS POSITION
    check("labels are keyed by this build's object_id",
          list(labels.index) == sorted(seg_oid.tolist()) or set(labels.index) == set(seg_oid))
    got = labels.loc[seg_oid, config.VISIUMHD_LEVELS[-1]].to_numpy()
    check("every cell carries the type belonging to its own position",
          bool((got == types[order]).all()),
          f"{int((got == types[order]).sum()):,}/{len(xy):,}")
    check("total_counts travels with the row",
          bool((labels.loc[seg_oid, "total_counts"].to_numpy()
                == ann.table["total_counts"].to_numpy()[order]).all()))
    # and the thing that makes the four checks above worth making: on this same input, the
    # join this replaced would have succeeded and been wrong. Stated here rather than left
    # to the reader, because a test of a fix is only as convincing as its account of the
    # bug -- and a *passing* assertion about the old behaviour cannot rot into a lie.
    id_join = ann.table.reindex(seg_oid)[config.VISIUMHD_LEVELS[-1]].to_numpy()
    overlap = np.isin(seg_oid, ann.table.index).mean()
    agreed = float(np.mean(id_join[np.isin(seg_oid, ann.table.index)]
                           == types[order][np.isin(seg_oid, ann.table.index)]))
    check("joining by object_id instead would have labelled from the wrong rows",
          agreed < 0.5, f"{overlap:.0%} of ids present, {agreed:.1%} of those correct")

    print("\ntranslated: the frame moved, so nothing corresponds")
    try:
        cl.join(ann, seg_oid, xy[order] + 50.0)
        check("a moved frame is refused", False, "it was accepted")
    except cl.AnnotationError as exc:
        check("a moved frame is refused", True, str(exc).split(".")[0][:70])

    print("\npartly missing: cells the new segmentation did not find")
    keep = np.ones(len(xy), bool)
    keep[RNG.choice(len(xy), 12, replace=False)] = False      # 0.6%, like P2
    labels, rep = cl.join(ann, seg_oid[keep], xy[order][keep])
    check("the rest are still re-keyed", rep["rekeyed"] == int(keep.sum()),
          f"{rep['rekeyed']:,}")
    check("and it is reported rather than hidden",
          0.99 <= rep["centroid_agreement"] < 1.0, f"{rep['centroid_agreement']}")

    print("\njittered: every centroid moved a little, which is not a correspondence")
    # 0.3 px: far enough to be a different measurement, close enough that the nearest
    # neighbour is still the right cell -- so only the tolerance can refuse it
    try:
        cl.join(ann, seg_oid, xy[order] + 0.3)
        check("sub-pixel drift is refused", False, "it was accepted")
    except cl.AnnotationError as exc:
        check("sub-pixel drift is refused", True, str(exc).split(".")[0][:70])

    print("\nthe tolerance admits the CSV's own rounding and nothing more")
    # 3 stored decimals put a correct match at <= 0.0005 per axis, so <= 0.0008 in 2-D
    labels, rep = cl.join(ann, seg_oid, xy[order] + 0.0004)
    check("rounding-sized disagreement is accepted", rep["rekeyed"] == len(xy),
          f"max {rep['max_centroid_distance_px']} px")
    check("and the tolerance is the stated one", cl.CENTROID_ATOL == 0.01,
          f"{cl.CENTROID_ATOL}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES))
        return 1
    print("all re-key checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
