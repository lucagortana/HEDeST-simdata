"""CPU post-processing: spot-filter the HoVerNet segmentation and assemble the deliverables.

Steps:

1. **check_staging** -- confirm the HoVerNet stage left what the next step needs: the raw
   segmentation (all detected cells, raw 0-5 types, un-padded, reindexed, no matching) and
   the matching cell crops. In HEDeST-bench this step *copied* those two files, because
   there they were produced under a different dataset family's folder and ``bench_data``
   had to end up self-contained. Here the HoVerNet stage writes them straight into this
   dataset's own staging area under ``_work``, so there is nothing to copy and the step is
   a check.
2. **filter_and_index** -- drop cells whose centroid is > ``DIST_THRESH_UM`` from
   the nearest pseudo-Visium spot (reproduces ``seg_postprocessing.
   filter_by_st_proximity``; removes cells on adjacent irrelevant tissue,
   important for prostate/skin), reindex 0..n-1, and write ``hovernet.json`` (raw
   HoVerNet ``type`` 0-5 + ``type_name``) and the matching ``image_dict.pt``.
3. **copy_he / copy_anndata** -- verbatim copies of ``he.tiff`` and the
   pseudo-Visium AnnData (only spot coords + transcriptome; nothing changes).
4. **build_spot_dict** -- ``spot_dict.json`` (spot -> member cells, ``only_in``),
   reproducing ``postseg.map_cells_to_spots`` on the filtered cells.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict

import anndata as ad
import numpy as np
from scipy.spatial import KDTree

from ..config import Layout
from . import hnio
from . import paths

# Same default as external/hovernet/seg_postprocessing.filter_by_st_proximity.
DIST_THRESH_UM = 300.0


def check_staging(layout: Layout, name: str) -> dict:
    """Confirm the HoVerNet stage produced what this one consumes."""
    ds = paths.dataset(layout, name)
    missing = [str(p) for p in (ds.staged_seg, ds.staged_image_dict) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{name}: the HoVerNet stage left nothing at:\n  " + "\n  ".join(missing)
            + "\nRun `--stages hovernet` for this dataset first.")
    return {"dataset": name,
            "segmentation": str(ds.staged_seg),
            "segmentation_bytes": ds.staged_seg.stat().st_size,
            "cell_crops": str(ds.staged_image_dict),
            "cell_crops_bytes": ds.staged_image_dict.stat().st_size}


def filter_and_index(layout: Layout, name: str) -> dict:
    import torch

    ds = paths.dataset(layout, name)
    data, nuc = hnio.load_nuc(ds.staged_seg)
    mag, mpp = data.get("mag"), data.get("mpp")
    mpp = float(mpp) if mpp else ds.mpp

    adata = ad.read_h5ad(ds.adata_path)
    spot_xy = np.asarray(adata.obsm["spatial"]).astype("int64")
    tree = KDTree(spot_xy)
    thr_px = DIST_THRESH_UM / mpp

    raw_images = torch.load(ds.staged_image_dict, weights_only=True)
    out_nuc: dict[str, dict] = {}
    new_images: dict = {}
    new_i = 0
    for old_id, e in nuc.items():                    # keys "0".."n-1" in order
        dist, _ = tree.query(np.asarray(e["centroid"], dtype=float))
        if dist > thr_px:
            continue
        raw_type = int(e.get("type", 0))
        ee = dict(e)
        ee["type"] = raw_type
        ee["type_name"] = hnio.HOVERNET_TYPE_NAMES.get(raw_type, str(raw_type))
        out_nuc[str(new_i)] = ee
        if old_id in raw_images:
            new_images[str(new_i)] = raw_images[old_id]
        new_i += 1

    hnio.write_nuc(ds.hovernet_json, out_nuc, mag, mpp)
    torch.save(new_images, ds.image_dict)
    assert len(new_images) == len(out_nuc), \
        f"{name}: image_dict ({len(new_images)}) != hovernet.json ({len(out_nuc)})"
    # raw copies no longer needed (deliverables written); free the big files.
    # The raw segmentation and crops are deliberately NOT deleted here. HEDeST-bench
    # deleted them because its staging folder sat inside the deliverable; here it is inside
    # _work, which is removed wholesale on success. Deleting them would buy nothing and
    # would cost idempotence: the "has hovernet run?" check looks for exactly these files,
    # so a resumed build would redo a 27-minute GPU stage to recreate what it just erased.
    return {"dataset": name, "n_raw": len(nuc), "n_kept": len(out_nuc),
            "n_dropped": len(nuc) - len(out_nuc),
            "dist_thresh_um": DIST_THRESH_UM, "dist_thresh_px": round(thr_px, 2)}


def copy_he(layout: Layout, name: str) -> None:
    ds = paths.dataset(layout, name)
    ds.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ds.he_tiff, ds.he_out)


def copy_anndata(layout: Layout, name: str) -> None:
    ds = paths.dataset(layout, name)
    ds.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ds.adata_path, ds.adata_out)


def build_spot_dict(layout: Layout, name: str) -> dict:
    ds = paths.dataset(layout, name)
    adata = ad.read_h5ad(ds.adata_path)
    spot_xy = np.asarray(adata.obsm["spatial"]).astype("int64")
    diameter = adata.uns["spatial"][ds.adata_name]["scalefactors"]["spot_diameter_fullres"]
    spot_ids = list(adata.obs.index)

    _, nuc = hnio.load_nuc(ds.hovernet_json)
    cell_ids, cell_xy = hnio.centroids(nuc)          # ids "0".."m-1"

    tree = KDTree(spot_xy)
    mapping: dict[str, list[str]] = defaultdict(list)
    for cid, xy in zip(cell_ids, cell_xy):
        for idx in tree.query_ball_point(xy, r=diameter / 2):
            mapping[spot_ids[idx]].append(str(cid))
    mapping = {k: v for k, v in mapping.items() if v}
    with open(ds.spot_dict, "w") as f:
        json.dump(mapping, f)
    return {"dataset": name, "n_spots_nonempty": len(mapping),
            "n_cells_assigned": sum(len(v) for v in mapping.values()),
            "n_cells_total": len(cell_ids)}
