"""The Xenium download stage.

The output bundle, the H&E and the alignment all have direct, verified URLs on 10x's CDN
and are downloaded here, the bundle unpacked in place. One file is not: breast's cell-type
spreadsheet, a supplementary download on the dataset page, which the error this raises
explains. See :mod:`hedest_simdata.common.sources`.

Until 2026-11-08 breast also installed a **committed H&E affine** from ``annotations/``,
on the belief that the published one was a different, wrong alignment. It is not. The
published affine differs from the committed one by a pure translation which, pushed through
the linear part, comes out as a crop offset of (2070.3, 2667.7) px -- and the H&E pyramid
the delivered dataset was built on is 20511 x 27587 where this repo's, from the
byte-identical OME-TIFF, is 24241 x 30786. The committed affine was re-fitted to a CROPPED
image that no code here produces, which is why it put only 39.1% of breast's Xenium nuclei
within 40 px of a HoVerNet nucleus (median 62.4 px) against the published affine's 74.3%
(median 6.85 px) on the same segmentation; the delivered build reached 75.8%. So the
published affine is simply correct for the image this repo builds, and a committed input
disappeared with the belief. The crop is not recoverable: no integer offset reproduces the
delivered pixels (best mean |difference| 9.2 of 255 at (2664, 2066), correlation 0.985), so
the delivered pyramid was re-encoded as well as cropped.
"""
from __future__ import annotations

from ..common import sources
from ..config import Layout
from . import paths


def run(layout: Layout, key: str, dry_run: bool = False) -> list[dict]:
    sl = paths.slide(layout, key)
    print(f"  checking raw inputs in {sl.data_dir}", flush=True)
    results = sources.fetch_or_explain(
        sources.xenium(layout.data_root, key, sl.spec.name),
        family="xenium", label=f"Xenium {key}",
        page=sources.page_for("xenium", key), dry_run=dry_run)

    # the bundle has to hold the two boundary tables and the cell table, or `convert`
    # fails much later with a less useful message
    needed = {
        "cells": sl.raw_outputs / "cells.parquet",
        "cell boundaries": sl.raw_outputs / "cell_boundaries.parquet",
        "nucleus boundaries": sl.raw_outputs / "nucleus_boundaries.parquet",
        "experiment": sl.raw_outputs / "experiment.xenium",
        "H&E alignment": sl.he_alignment,
    }
    missing = [f"{k}: {p}" for k, p in needed.items() if not p.exists()]
    if sl.he_ome_tif is None:
        missing.append(f"H&E image: none found under {sl.data_dir}")
    if sl.spec.annotation == "vendor" and not sl.annotation_xlsx.exists():
        missing.append(f"cell-type spreadsheet: {sl.annotation_xlsx}")
    if missing:
        raise sources.MissingSource(
            f"Xenium {key}: the download is there but incomplete:\n  " + "\n  ".join(missing))
    print("  raw inputs complete", flush=True)
    return results
