"""Run CellViT on the bench_data samples, as a second segmentation backend.

Every bench_data sample already carries a ``hovernet.json``. This stage produces
the CellViT counterpart, ``cellvit.json``, in the *same* schema and over the
*same* tissue region, so the two backends are interchangeable downstream.

It reuses the HEDeST CellViT code unchanged
(``external/CellViT/cellvit/detect_cells.py``, model ``SAM``, taxonomy
``pannuke``), driven with the HoVer-Net-compatible export flags that HEDeST
added to it:

* ``--export_cells``  -> a ``{"mag": .., "nuc": {..}}`` json, ids 0..n-1;
* ``--image_dict_path`` -> optional 64px/20um cell crops (``--image-dict``),
  off by default since only the segmentation was asked for.

CellViT also offers ``--adata_path`` for the 300um spot-proximity filter, but
**we deliberately do not use it**: it imports ``anndata``, which ``cellvit-env``
does not have, so the run dies *after* the GPU work is finished. The filtering is
done here instead, in :func:`finalize`, using the same KDTree rule and the same
:data:`hedest_bench.bench.postprocess.DIST_THRESH_UM` threshold that
:func:`hedest_bench.bench.postprocess.filter_and_index` applies to HoVerNet --
so ``cellvit.json`` and ``hovernet.json`` still cover the same cells-near-spots
region (which matters for prostate/skin, with their adjacent irrelevant tissue),
and ``cellvit-env`` is left untouched.

**No reflect-padding here, unlike the HoVerNet stage.** HoVerNet *fast* mode
predicts a 164px output from a 256px input, so it cannot segment within ~46px of
the image border and ``hier.hovernet_stage`` pads the slide to recover that
strip. CellViT-SAM predicts the full 1024px patch (patches overlap by 64px and
``OverlapCellCleaner`` merges them), so there is no valid-region shrink and
nothing to recover.

Two schema fixes are also applied when the output is finalised, because CellViT's
``save_nuc_json`` writes only ``mag`` + ``nuc``:

* ``mpp`` is added (the sample's own mpp, as in ``hovernet.json``);
* ``type_name`` is added per cell, from the PanNuke ids CellViT and HoVerNet
  share, so ``cellvit.json`` is byte-for-byte schema-identical to
  ``hovernet.json``.

Run it with::

    python -m hedest_bench.bench.run cellvit <name>
    python -m hedest_bench.bench.submit --stages cellvit
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from ..config import Layout
from ..hedest_bridge import Bridge
from . import hnio
from . import paths

MAGNIFICATION = 40  # STHELAR H&E is 40x (mpp ~0.274); CellViT parses this as an int
# CellViT-SAM at 1024px is VRAM-hungry; 8 suits a 48GB A40, while a 16GB card
# (T4) needs 4 or less. ``CELLVIT_BATCH_SIZE`` lets the job script pick without
# touching the code, so one sbatch can serve heterogeneous GPUs.
BATCH_SIZE = int(os.environ.get("CELLVIT_BATCH_SIZE", 8))


def _cellvit_dir(ds: paths.Dataset) -> Path:
    """CellViT's own output directory, inside the sample's staging folder."""
    return ds.staging_dir / "cellvit"


def run_infer(layout: Layout, bridge: Bridge, name: str, gpu: int = 0,
              image_dict: bool = False, batch_size: int = BATCH_SIZE) -> dict:
    """
    Segment one sample with CellViT and write ``cellvit.json``.

    Args:
        layout: Where the data lives.
        bridge: Resolves the CellViT entry point and its interpreter.
        name: The bench_data sample name.
        gpu: GPU id to run on.
        image_dict: Also extract the 64px/20um cell crops (a few GB per sample).
        batch_size: CellViT inference batch size.

    Returns:
        The summary returned by :func:`finalize`.
    """

    ds = paths.dataset(layout, name)
    outdir = _cellvit_dir(ds)
    # CellViT skips a WSI listed in processed_files.json, and would otherwise
    # leave a stale json behind; clear the whole directory for a fresh run.
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not ds.he_out.exists():
        raise FileNotFoundError(
            f"{name}: {ds.he_out} is missing; run the cpu_post stage first "
            "(it copies he.tiff into bench_data).")

    bridge.cellvit_detect(
        slide=ds.he_out, out_dir=outdir, mpp=ds.mpp, magnification=MAGNIFICATION,
        gpu=gpu, batch_size=batch_size,
        image_dict=(outdir / "image_dict_raw.pt") if image_dict else None)

    return finalize(layout, name, image_dict=image_dict)


def finalize(layout: Layout, name: str, image_dict: bool = False) -> dict:
    """
    Filter CellViT's output to the spot region and write it as ``cellvit.json``.

    CellViT names its output after the input file (``he.tiff`` -> ``he.json``) and
    writes only ``mag`` + ``nuc`` over the whole slide. This

    1. drops cells further than ``postprocess.DIST_THRESH_UM`` from the nearest
       pseudo-Visium spot and reindexes 0..n-1 -- the same rule
       :func:`hedest_bench.bench.postprocess.filter_and_index` applies to
       HoVerNet, so both backends cover the same region;
    2. adds ``mpp`` and the per-cell ``type_name`` for schema parity;
    3. drops CellViT's own bulky per-WSI folder (``cells.json``, ...).

    Args:
        name: The bench_data sample name.
        image_dict: Whether crops were extracted (moves the ``.pt`` as well).

    Returns:
        Cell counts before and after filtering, and HoVerNet's for reference.
    """

    import anndata as ad
    import numpy as np
    from scipy.spatial import KDTree

    from .postprocess import DIST_THRESH_UM

    ds = paths.dataset(layout, name)
    outdir = _cellvit_dir(ds)
    stem = ds.he_out.stem  # "he"

    produced = outdir / f"{stem}.json"
    if not produced.exists():
        raise FileNotFoundError(
            f"{name}: CellViT wrote no {produced}; did --export_cells fail?")

    with open(produced) as f:
        data = json.load(f)
    raw_nuc = data["nuc"]

    # Spot-proximity filter, in this env because cellvit-env has no anndata.
    adata = ad.read_h5ad(ds.adata_path)
    tree = KDTree(np.asarray(adata.obsm["spatial"]).astype("int64"))
    thr_px = DIST_THRESH_UM / ds.mpp

    nuc: dict = {}
    for entry in raw_nuc.values():                   # keys "0".."n-1", in order
        dist, _ = tree.query(np.asarray(entry["centroid"], dtype=float))
        if dist > thr_px:
            continue
        entry["type_name"] = hnio.HOVERNET_TYPE_NAMES.get(int(entry.get("type", 0)),
                                                          str(entry.get("type")))
        nuc[str(len(nuc))] = entry
    hnio.write_nuc(ds.cellvit_json, nuc, data.get("mag", MAGNIFICATION), ds.mpp)

    if image_dict:
        crops = outdir / "image_dict_raw.pt"
        if crops.exists():
            shutil.move(str(crops), str(ds.cellvit_image_dict))

    # CellViT's per-WSI folder holds the un-reindexed cells.json/cell_detection.json
    # (gigabytes); the deliverable is written, so drop it along with the rest.
    shutil.rmtree(outdir, ignore_errors=True)

    summary = {"dataset": name, "n_cellvit": len(nuc), "n_raw": len(raw_nuc),
               "n_dropped": len(raw_nuc) - len(nuc),
               "dist_thresh_um": DIST_THRESH_UM, "mpp": ds.mpp,
               "json": str(ds.cellvit_json)}
    if ds.hovernet_json.exists():
        _, hn = hnio.load_nuc(ds.hovernet_json)
        summary["n_hovernet"] = len(hn)
    return summary
