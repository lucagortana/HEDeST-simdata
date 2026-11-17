#!/usr/bin/env python3
"""Extracts the committed Xenium Lung annotation CSV.

Run **once**; the result is committed and the pipeline reads that. Shipped so the
provenance is auditable rather than a claim.

    python annotations/extract_xenium.py --source /cluster/CBIO/data1/lgortana

Only the **Lung** slide needs this. Its cell types came from a KNN label transfer against
a single-cell lung atlas, and reproducing that would mean shipping the reference and making
every user run the transfer -- for an input that is the same every time. (Tangram was tried
too and not used; the shipped labels are the cosine-KNN ones.) **Breast** needs nothing
committed: its annotation is a spreadsheet on the dataset page, which the pipeline reads.

The key is the Xenium ``cell_id``, the vendor's own barcode -- **and that is not as stable
as it looks.** It survives re-runs of the vendor pipeline, but 10x re-released this slide
with an entirely different id space, so the annotation as first committed joined 8 of
162,254 cells against a fresh download. The cells are unchanged (every reported centroid is
bit-identical between the releases), so :func:`_rekey_to_published_ids` puts the labels onto
the published ids by matching geometry, and the centroids are committed alongside. The
pipeline then checks the join against them, exactly the centroid safeguard Visium HD's
``object_id`` has -- an earlier version of this docstring argued that Xenium needed no such
safeguard, which is the sentence this one replaces.
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hedest_simdata import config  # noqa: E402

#: Where the label transfer left its output, inside the lung data folder.
SOURCE_REL = "sim/LuCA/cosine_annotation.h5ad"

#: The confidence matrix names this type differently from everything downstream.
CONFIDENCE_RENAME = {"Macrophage": "Macrophage/Monocyte"}


def _obs_column(g) -> np.ndarray:
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


def _rekey_to_published_ids(df: pd.DataFrame, zarr_path: Path) -> pd.DataFrame:
    """Re-keys the annotation onto the ids the *currently published* bundle uses.

    The label transfer was run against an earlier release of this dataset, whose cell ids
    were a sequential space (``aaaaaaaa-1``, ``aaaaaaab-1``, ...). 10x has since re-released
    the slide with different ids (``aaaadpbp-1``, ...), so the annotation as first committed
    joined **8 of 162,254** cells against a fresh download -- which is how this was found.

    The cells themselves did not change: every one of the 162,254 reported centroids is
    bit-identical between the two releases. The delivered archive happens to carry both
    keyings -- ``cell_boundaries`` is indexed by the old ids and ``cell_circles`` by the
    published ones -- so the map between them is recovered here by matching geometry, and
    the centroids are written out so the pipeline can *check* the join instead of trusting
    it (:func:`hedest_simdata.xenium.annotation._check_centroids`).

    Matching is by mutual nearest neighbour, which settles 162,248 pairs on its own; the
    polygon centroid of a boundary sits a median 0.13 um from the vendor's reported
    centroid for the same cell and 52x further from any other cell's, so the pairing is not
    close to ambiguous. The handful left over are contested only between themselves and are
    resolved by minimum total distance.
    """
    import spatialdata as sd
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    if not zarr_path.exists():
        raise SystemExit(
            f"{zarr_path} not found; it carries both id spaces and is what makes the "
            "re-key possible. Without it the annotation cannot be put onto the published "
            "ids, and the pipeline will refuse to build from it.")
    sdata = sd.read_zarr(str(zarr_path))
    cb, cc = sdata.shapes["cell_boundaries"], sdata.shapes["cell_circles"]
    old = np.c_[cb.geometry.centroid.x.values, cb.geometry.centroid.y.values]
    new = np.c_[cc.geometry.x.values, cc.geometry.y.values]
    old_ids = cb.index.astype(str).to_numpy()
    new_ids = cc.index.astype(str).to_numpy()

    _, to_new = cKDTree(new).query(old, k=1)
    _, to_old = cKDTree(old).query(new, k=1)
    mutual = to_old[to_new] == np.arange(len(old))
    pair = {int(i): int(to_new[i]) for i in np.flatnonzero(mutual)}
    left_old = np.flatnonzero(~mutual)
    left_new = np.setdiff1d(np.arange(len(new)), to_new[mutual])
    if len(left_old):
        cost = np.linalg.norm(old[left_old][:, None, :] - new[left_new][None, :, :], axis=2)
        ri, ci = linear_sum_assignment(cost)
        print(f"  {len(pair):,} mutual nearest pairs; {len(left_old)} contested, settled by "
              f"assignment (worst {cost[ri, ci].max():.3f} um)")
        for i, j in zip(ri, ci):
            pair[int(left_old[i])] = int(left_new[j])
    if len(pair) != len(old) or len(set(pair.values())) != len(old):
        raise SystemExit("the old-to-published id map is not a bijection; refusing to write")

    idx = list(pair)
    remap = pd.DataFrame({"old_id": old_ids[idx],
                          "cell_id": new_ids[list(pair.values())],
                          "x_centroid": new[list(pair.values()), 0].round(4),
                          "y_centroid": new[list(pair.values()), 1].round(4)})
    out = remap.merge(df.rename(columns={"cell_id": "old_id"}), on="old_id", how="inner")
    if len(out) != len(df):
        raise SystemExit(f"re-key covered {len(out):,} of {len(df):,} annotated cells")
    out = out.rename(columns={"old_id": "source_cell_id"})
    first = ["cell_id", "x_centroid", "y_centroid", "cell_type"]
    rest = [c for c in out.columns if c not in first and c != "source_cell_id"]
    return out.sort_values("cell_id")[first + rest + ["source_cell_id"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="/cluster/CBIO/data1/lgortana")
    ap.add_argument("--out", default=str(REPO / "annotations" / "xenium"))
    args = ap.parse_args(argv)

    slide = config.XENIUM_SLIDES["lung"]
    src = Path(args.source) / slide.name / SOURCE_REL
    if not src.exists():
        raise SystemExit(f"{src} not found")

    import anndata as ad

    adata = ad.read_h5ad(src)
    for col in ("cell_id", "cell_type"):
        if col not in adata.obs:
            raise SystemExit(f"{src.name}: obs has no {col!r}")
    df = pd.DataFrame({"cell_id": adata.obs["cell_id"].astype(str).to_numpy(),
                       "cell_type": adata.obs["cell_type"].astype(str).to_numpy()})

    # The ground truth this feeds is SOFT: the delivered sim_*_gt.csv carries the label
    # transfer's confidence over cell types, not a one-hot -- only 60.2% of its rows are
    # one-hot and its row sums run from 0.35 to 1.0, because confidence that went to a
    # dropped type is simply left out rather than redistributed. So the confidence columns
    # are committed alongside the argmax, or that file cannot be rebuilt.
    if "annotation_confidence" not in adata.obsm:
        raise SystemExit(f"{src.name}: obsm has no 'annotation_confidence'")
    conf = pd.DataFrame(adata.obsm["annotation_confidence"]).rename(
        columns=CONFIDENCE_RENAME)
    keep = [t for t in slide.cell_types if t in conf.columns]
    missing = [t for t in slide.cell_types if t not in conf.columns]
    if missing:
        raise SystemExit(f"{src.name}: the confidence matrix lacks {missing}")
    for t in keep:
        df[f"conf:{t}"] = conf[t].to_numpy()

    if df["cell_id"].duplicated().any():
        raise SystemExit("cell_id is not unique, so it cannot be the key")

    df = _rekey_to_published_ids(df, Path(args.source) / slide.name / "data.zarr")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{slide.name}_annotation.csv.gz"
    with gzip.open(dest, "wt", newline="", compresslevel=9) as fh:
        df.to_csv(fh, index=False)

    cols = [c for c in df.columns if c.startswith("conf:")]
    print(f"  confidence columns: {len(cols)} ({[c[5:] for c in cols]})")
    counts = df["cell_type"].value_counts()
    kept = [t for t in counts.index if t not in slide.drop_types]
    print(f"  lung: {len(df):,} cells, {dest.stat().st_size / 2**20:.1f} MiB gz")
    print(f"  {len(counts)} types; dropping {list(slide.drop_types)} leaves {len(kept)}")
    print(f"  expected {len(slide.cell_types)}: "
          f"{'MATCH' if set(kept) == set(slide.cell_types) else 'MISMATCH'}")
    print()
    print(counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
