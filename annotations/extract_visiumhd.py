#!/usr/bin/env python3
"""Extracts the committed Visium HD annotation CSVs from the manually annotated h5ad files.

Run **once**, by whoever did the annotation; the result is committed and the pipeline reads
that. It is shipped so the provenance is auditable rather than a claim.

    python annotations/extract_visiumhd.py --source /cluster/CBIO/data1/lgortana

The manual annotation of the Visium HD cells is the one thing in this repository that no
code can reproduce. Everything else about those cells -- the segmentation, the centroids,
the counts -- comes back from bin2cell, so only the labels are frozen.

**Why the file carries more than the labels.** These rows have to find their way back onto
freshly segmented nuclei, and ``object_id`` cannot do it: a re-run of
StarDist/HoVerNet/bin2cell paints the same cells but numbers them differently, which the
first clean-tree build measured as 0.09% of P2's cells landing where the annotation said.
So each row also carries the **centroid** and the total counts bin2cell recorded, and
:mod:`hedest_simdata.visiumhd.cell_labels` re-keys by mutual nearest centroid (0.01 px
tolerance; the stored 3 decimals put a correct match at 0.0004 px) with a 5 px-shifted
control that must recover nothing. Drop those columns and the labels become unattachable.

**Why the row count is not the cell count.** The annotated files hold fewer cells than
bin2cell produced (178,601 of 184,197 for P5), because QC ran before annotation. The CSV
therefore also *defines* which cells are annotated, which is better than re-deriving it:
re-running QC would make those thresholds a second irreproducible choice.
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hedest_simdata import config  # noqa: E402

#: The annotation columns, coarse to fine.
LEVELS = list(config.VISIUMHD_LEVELS)

#: Written with this many decimals. The centroids are bin2cell's own floats in
#: capture-image pixels; 3 decimals is a nanometre at 0.27 um/px, far below what the
#: verification needs, and keeps the committed file small.
COORD_DECIMALS = 3


def _obs_column(g) -> np.ndarray:
    """One obs column, categorical or plain, decoded to python objects."""
    if isinstance(g, h5py.Group):
        cats = np.asarray([c.decode() if isinstance(c, bytes) else c
                           for c in g["categories"][:]], dtype=object)
        codes = g["codes"][:]
        return np.where(codes >= 0, cats[np.clip(codes, 0, None)], None)
    v = g[:]
    if v.dtype.kind in "OS":
        return np.asarray([x.decode() if isinstance(x, bytes) else x for x in v],
                          dtype=object)
    return v


def extract(annotated: Path) -> pd.DataFrame:
    with h5py.File(annotated, "r") as f:
        obs = f["obs"]
        idx = obs.attrs.get("_index", "_index")
        ids = np.asarray([x.decode() if isinstance(x, bytes) else x for x in obs[idx][:]],
                         dtype=str)
        missing = [c for c in LEVELS + ["object_id", "total_counts"] if c not in obs]
        if missing:
            raise SystemExit(f"{annotated.name}: obs lacks {missing}")

        crop_keys = [k for k in f["obsm"].keys()
                     if re.fullmatch(r"spatial_cropped_\d+_buffer", k)]
        if len(crop_keys) != 1:
            raise SystemExit(
                f"{annotated.name}: expected one spatial_cropped_*_buffer in obsm, "
                f"found {crop_keys}")
        xy = f[f"obsm/{crop_keys[0]}"][:].astype(np.float64)

        df = pd.DataFrame({
            "cell_id": ids,
            "object_id": obs["object_id"][:].astype(np.int64),
            "x": np.round(xy[:, 0], COORD_DECIMALS),
            "y": np.round(xy[:, 1], COORD_DECIMALS),
            "total_counts": obs["total_counts"][:].astype(np.float64),
        })
        for lv in LEVELS:
            df[lv] = _obs_column(obs[lv])
        df.attrs["crop_key"] = crop_keys[0]
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="/cluster/CBIO/data1/lgortana",
                    help="folder holding the Visium_HD_Human_Colon_Cancer_* samples")
    ap.add_argument("--out", default=str(REPO / "annotations" / "visiumhd"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for patient in config.VISIUMHD_PATIENTS:
        name = config.visiumhd_name(patient)
        src = Path(args.source) / name / f"{name}_annotated.h5ad"
        if not src.exists():
            print(f"  {patient}: {src} not found, skipped")
            continue
        df = extract(src)
        # still required to be unique: it is this build's own identifier for the row, and a
        # duplicate would mean two cells' labels were merged before the file was written.
        # It is no longer the JOIN key -- the centroid is -- but a file whose own ids
        # collide is a file that was assembled wrongly.
        if df["object_id"].duplicated().any():
            raise SystemExit(f"{patient}: object_id is not unique, so this file was built "
                             "wrongly -- two cells' rows have been merged")
        dest = out_dir / f"{name}_annotation.csv.gz"
        with gzip.open(dest, "wt", newline="", compresslevel=9) as fh:
            df.to_csv(fh, index=False)
        mb = dest.stat().st_size / 2**20
        n_types = {lv: df[lv].nunique(dropna=True) for lv in LEVELS}
        print(f"  {patient}: {len(df):,} cells, {mb:.1f} MiB gz, "
              f"crop frame {df.attrs['crop_key']}, types {n_types}")
        summary.append({"patient": patient, "cells": len(df), "mib": round(mb, 1),
                        "crop_key": df.attrs["crop_key"], **
                        {f"n_{lv}": v for lv, v in n_types.items()}})
    if summary:
        print()
        print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
