"""Xenium: the Lung and Breast slides -> pseudo-Visium spots.

    download -> convert -> pyr_tif -> mask -> hovernet -> spots -> hoptimus -> verify

Ported from the two notebooks in
``HEDeST/simulations/semi_simulations/Xenium/dataset_construction/``. Only the six files
``HEDeST/gridsearch/semi_sim.yaml`` resolves are rebuilt; the twenty-odd others in those
data folders were experiments. See :mod:`hedest_simdata.xenium.build` for what had to be
recovered from the delivered files rather than read off the notebooks -- which is most of
it, because they were last saved in a configuration that did not ship.

The annotation is **not** recomputed: Lung's label transfer ships as a CSV in
``annotations/xenium/``, Breast's comes from the vendor's own spreadsheet.
"""
from __future__ import annotations

import json
import time

from .. import config
from ..config import Layout
from ..hedest_bridge import Bridge
from ..pipeline import REGISTRY
from ..pipeline import Resources
from . import paths

#: Set by the driver (``pipeline.set_bridge``); resolves HEDeST and its environments.
BRIDGE: Bridge | None = None

#: Pyramid level the tissue mask is computed at, and HoVerNet's batch size here.
MASK_LEVEL = 3
HOVERNET_BATCH = 16


def _bridge() -> Bridge:
    if BRIDGE is None:  # pragma: no cover - the driver always sets it
        raise RuntimeError("the HEDeST bridge was not configured; run via make_simdata.py")
    return BRIDGE


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# "already built?" predicates
# --------------------------------------------------------------------------------------


def _done_download(layout: Layout, key: str) -> bool:
    from ..common import sources
    sl = paths.slide(layout, key)
    return all(s.present for s in sources.xenium(layout.data_root, key, sl.spec.name)
               if not s.optional)


def _done_convert(layout: Layout, key: str) -> bool:
    return (paths.slide(layout, key).sdata_zarr / ".zgroup").exists()


def _done_pyr_tif(layout: Layout, key: str) -> bool:
    return paths.slide(layout, key).pyr_tif.exists()


def _done_mask(layout: Layout, key: str) -> bool:
    return paths.slide(layout, key).mask.exists()


def _done_hovernet(layout: Layout, key: str) -> bool:
    return paths.slide(layout, key).seg_json.exists()


def _done_spots(layout: Layout, key: str) -> bool:
    sl = paths.slide(layout, key)
    return all(p.exists() for n, p in sl.deliverables().items() if n != "hoptimus")


def _done_hoptimus(layout: Layout, key: str) -> bool:
    return paths.slide(layout, key).hoptimus_embed.exists()


# --------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------


@REGISTRY.stage("xenium", "download", "xenium",
                resources=Resources(cpus=2, mem_gb=4, hours=8), done=_done_download,
                doc="check / fetch the Xenium output bundle, the H&E and the alignment CSV")
def stage_download(layout: Layout, key: str) -> None:
    from . import fetch
    fetch.run(layout, key)


@REGISTRY.stage("xenium", "convert", "xenium", needs=["download"],
                resources=Resources(cpus=8, mem_gb=96, hours=4), done=_done_convert,
                doc="the Xenium output bundle -> a SpatialData archive")
def stage_convert(layout: Layout, key: str) -> None:
    import shutil

    from spatialdata_io import xenium

    sl = paths.slide(layout, key)
    _log(f"{key}: parsing {sl.raw_outputs}")
    # `cells_boundaries`, plural: the original converter in HEDeST's
    # dataset_construction/utils/to_zarr.py says `cell_boundaries`, which is not a parameter
    # of any spatialdata-io this repo can install (0.1.5 and 0.3.0 both spell it
    # `cells_boundaries`) and raises TypeError. The singular form is the *element* name in
    # the written archive, which is probably how the two got crossed. Faithfully copying it
    # here meant `xenium:convert` could never run -- and until 2026-11-08 it never did, the
    # archive being carried over from the original notebooks instead.
    #
    # Everything not named here keeps its default of True, which is what the delivered
    # archives contain: both have cell/nucleus boundaries, labels, transcripts and the
    # cell table, breast additionally has morphology_mip because its bundle ships one and
    # lung's newer multi-file bundle does not. So the defaults are not laziness, they are
    # what reproduces the reference element set on both slides.
    sdata = xenium(path=str(sl.raw_outputs), n_jobs=8, cells_boundaries=True,
                   nucleus_boundaries=True, morphology_focus=True, cells_as_circles=True)
    if sl.sdata_zarr.exists():
        shutil.rmtree(sl.sdata_zarr)
    sl.sdata_zarr.parent.mkdir(parents=True, exist_ok=True)
    sdata.write(str(sl.sdata_zarr))
    _log(f"{key}: wrote {sl.sdata_zarr}")


@REGISTRY.stage("xenium", "pyr_tif", "xenium", needs=["download"],
                resources=Resources(cpus=8, mem_gb=96, hours=3), done=_done_pyr_tif,
                doc="the vendor H&E -> a tiled pyramidal TIFF OpenSlide can read")
def stage_pyr_tif(layout: Layout, key: str) -> None:
    from ..common import tiff

    sl = paths.slide(layout, key)
    src = sl.he_ome_tif
    if src is None:
        raise FileNotFoundError(
            f"{key}: no H&E image found under {sl.data_dir}; the download stage lists it")
    _log(f"{key}: reading {src.name}")
    img = tiff.read_whole(src)
    mpp = _he_mpp(src)
    _log(f"{key}: {img.shape[1]}x{img.shape[0]} at {mpp:.5f} um/px -> pyramid")
    _log(f"{key}: {json.dumps(tiff.write_pyramid(img, sl.pyr_tif, mpp))}")


def _he_mpp(path) -> float:
    """Microns per pixel of the vendor H&E, from its own resolution tags.

    Not guessed and not taken from a constant: the two slides differ (lung's TIFF records
    0.2738, breast's carries a meaningless 72 dpi default), and the value drives the
    physical size of every cell window downstream.
    """
    import tifffile as tf

    with tf.TiffFile(str(path)) as t:
        page = t.pages[0]
        res = page.tags.get("XResolution")
        unit = page.tags.get("ResolutionUnit")
        if res is not None and unit is not None and int(unit.value) == 3:
            num, den = res.value
            if num and den:
                per_cm = num / den
                return 1e4 / per_cm
    raise ValueError(
        f"{path} records no usable resolution (its ResolutionUnit is not CENTIMETER). "
        "Pass the slide's mpp explicitly rather than letting it be guessed -- it sets the "
        "physical size of every cell window.")


@REGISTRY.stage("xenium", "mask", "xenium", needs=["pyr_tif"],
                resources=Resources(cpus=4, mem_gb=96, hours=2), done=_done_mask,
                doc="Otsu tissue mask at pyramid level 3")
def stage_mask(layout: Layout, key: str) -> None:
    sl = paths.slide(layout, key)
    _log(f"{key}: tissue mask")
    _bridge().hovernet_mask(sl.pyr_tif, sl.mask)


@REGISTRY.stage("xenium", "hovernet", "xenium", needs=["mask"],
                resources=Resources(cpus=8, mem_gb=64, gpu=True, hours=6),
                done=_done_hovernet,
                doc="HoVerNet WSI inference over the aligned H&E")
def stage_hovernet(layout: Layout, key: str) -> None:
    sl = paths.slide(layout, key)
    mpp = _he_mpp(sl.he_ome_tif) if sl.he_ome_tif else None
    _log(f"{key}: HoVerNet inference at {mpp}")
    _bridge().hovernet_infer(
        wsi_dir=sl.pyr_tif.parent, mask_dir=sl.mask_dir, out_dir=sl.seg_dir,
        mpp=mpp, cache_dir=sl.cache_dir, image_dict=None,
        batch_size=HOVERNET_BATCH)


@REGISTRY.stage("xenium", "spots", "xenium", needs=["convert", "hovernet"],
                resources=Resources(cpus=8, mem_gb=128, hours=4), done=_done_spots,
                doc="align, match nuclei to Xenium cells, brick lattice, proportions, GT")
def stage_spots(layout: Layout, key: str) -> None:
    from . import build
    sl = paths.slide(layout, key)
    _log(f"{key}: building the five CPU deliverables")
    _log(f"{key}: {json.dumps(build.run(sl))}")


@REGISTRY.stage("xenium", "hoptimus", "xenium", needs=["spots"],
                resources=Resources(cpus=6, mem_gb=48, gpu=True, hours=3),
                done=_done_hoptimus,
                doc="H-Optimus-0 tile embeddings for the ground-truth nuclei")
def stage_hoptimus(layout: Layout, key: str) -> None:
    import pandas as pd
    import torch

    sl = paths.slide(layout, key)
    mpp = _he_mpp(sl.he_ome_tif) if sl.he_ome_tif else None
    _log(f"{key}: H-Optimus-0 tile embeddings")
    emb = _bridge().hoptimus_embeddings(
        slide=sl.pyr_tif, segmentation=sl.annotated_seg, out=sl.hoptimus_all, mpp=mpp)
    dims = {tuple(v.shape) for v in emb.values()}
    if dims != {(1536,)}:
        raise ValueError(f"{key}: embeddings are {dims}, expected {{(1536,)}}")

    # The deliverable carries the ground-truth nuclei only, in ground-truth order. Measured
    # from both delivered files: lung 107,429 entries with keys running to 141,256, breast
    # 120,336 running to 175,202 -- in each case exactly `gt.index` in that order, out of
    # segmentations of 141,257 and 175,204. Keeping every nucleus instead made the file
    # 1.3x larger with 33,828 extra keys that nothing downstream can use: the spot
    # dictionary's cells are a subset of the ground truth, which `verify` checks.
    order = [str(i) for i in pd.read_csv(sl.ground_truth, index_col=0, usecols=[0]).index]
    missing = [i for i in order if i not in emb]
    if missing:
        raise ValueError(
            f"{key}: {len(missing):,} ground-truth nuclei have no embedding "
            f"(e.g. {missing[:3]}); the segmentation and the ground truth disagree")
    sl.hoptimus_embed.parent.mkdir(parents=True, exist_ok=True)
    torch.save({i: emb[i] for i in order}, sl.hoptimus_embed)
    _log(f"{key}: {len(order):,} of {len(emb):,} embeddings of dim 1536 kept "
         f"(the ground-truth nuclei) -> {sl.hoptimus_embed.name}")


@REGISTRY.stage("xenium", "verify", "xenium", needs=["spots", "hoptimus"],
                resources=Resources(cpus=4, mem_gb=48, hours=1),
                doc="independent checks of everything written, against the manifest")
def stage_verify(layout: Layout, key: str) -> None:
    from . import verify
    sl = paths.slide(layout, key)
    report = verify.run(sl)
    _log(f"{key}: {json.dumps(report, indent=2)}")
    if not report["ok"]:
        raise SystemExit(f"{key}: verification failed with {len(report['issues'])} issues")
