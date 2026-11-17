"""STHELAR: seven Xenium 5K slides -> pseudo-Visium -> the 8 benchmark datasets.

Ported from HEDeST-bench, with four changes:

* **one entry point.** The per-module CLIs are gone; stages are declared below and run by
  ``make_simdata.py``.
* **no ``hier/``.** That half of HEDeST-bench built a second, now-abandoned dataset
  family. Only ``hnio`` (HoVer-Net schema IO) and ``hovernet_stage`` were shared with the
  benchmark, so they live here; ``hierarchies``, ``colors`` and ``trees`` turned out to be
  unused by it and are dropped.
* **no MoCo.** The embeddings were not reproducible (trained without a fixed seed) and are
  superseded by H-Optimus-0, which nothing in HEDeST-bench produced -- the ``hoptimus``
  stage below is new. ``image_dict.pt`` is unaffected: it is written by ``post``, not by
  the MoCo stage.
* **no slide or crop figures.** The per-slide pseudo-Visium datasets and the crops are
  intermediates in ``_work/``; only ``bench_data/<name>/plots/`` is a deliverable. That
  dropped ~1,200 lines of figure code with nothing downstream depending on it.

The stage order, and where each one runs::

    download -> slides -> crops -> stage -> hovernet -> post -> cellvit  -\\
                                                             -> hoptimus -+-> verify
                                                             -> plots    -/
"""
from __future__ import annotations

import json
import time

from .. import config
from ..config import Layout
from ..common import sources
from ..hedest_bridge import Bridge
from ..pipeline import REGISTRY
from ..pipeline import Resources
from . import paths

#: The bridge is supplied by the driver, which knows the command-line overrides.
BRIDGE: Bridge | None = None


def _bridge() -> Bridge:
    if BRIDGE is None:  # pragma: no cover - the driver always sets it
        raise RuntimeError("the HEDeST bridge was not configured; run via make_simdata.py")
    return BRIDGE


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# "already built?" predicates
# --------------------------------------------------------------------------------------
# Each one names the file its stage writes LAST, so a half-written dataset never counts as
# finished: if a stage died partway, its marker is absent and the stage re-runs. That is
# the whole reason these are specific files rather than "the output folder exists".


def _done_download(layout: Layout, slide: str) -> bool:
    return all(src.present for src in sources.sthelar(layout.data_root, slide))


def _done_slides(layout: Layout, slide: str) -> bool:
    return paths.slide(layout, slide).metadata_path.exists()


def _done_crops(layout: Layout, name: str) -> bool:
    if name not in config.STHELAR_CROPS:
        return True                      # used whole; there is nothing to crop
    return paths.dataset(layout, name).metadata_path.exists()


def _done_stage(layout: Layout, name: str) -> bool:
    ds = paths.dataset(layout, name)
    # the padded H&E is a temporary that `hovernet` deletes once it has finished with it,
    # so its absence does not mean this stage has to re-export several GB -- the staged
    # segmentation existing is just as good a witness that staging happened.
    return ds.staged_mask.exists() and (ds.staged_wsi.exists() or ds.staged_seg.exists())


def _done_hovernet(layout: Layout, name: str) -> bool:
    return paths.dataset(layout, name).staged_seg.exists()


def _done_post(layout: Layout, name: str) -> bool:
    # write_levels_json is the last thing build_sim does
    return (paths.dataset(layout, name).sim_dir / "levels.json").exists()


def _done_cellvit(layout: Layout, name: str) -> bool:
    return paths.dataset(layout, name).cellvit_json.exists()


def _done_hoptimus(layout: Layout, name: str) -> bool:
    return paths.dataset(layout, name).hoptimus_embed.exists()


def _done_plots(layout: Layout, name: str) -> bool:
    # the level-tree diagram is drawn last
    return (paths.dataset(layout, name).plots_dir / "03_levels_tree.png").exists()


# --------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------


@REGISTRY.stage("sthelar", "download", "slide",
                resources=Resources(cpus=2, mem_gb=4, hours=12),
                doc="fetch one slide's .zarr.zip and cell-metadata parquet", done=_done_download)
def stage_download(layout: Layout, slide: str) -> None:
    _log(f"{slide}: download")
    sources.fetch_or_explain(
        sources.sthelar(layout.data_root, slide), family="sthelar",
        label=f"STHELAR {slide}", page=sources.page_for("sthelar"))


@REGISTRY.stage("sthelar", "slides", "slide", needs=["download"],
                resources=Resources(cpus=8, mem_gb=96, hours=6),
                doc="build one slide's pseudo-Visium dataset (grid, counts, DAPI truth, H&E)", done=_done_slides)
def stage_slides(layout: Layout, slide: str) -> None:
    from . import preprocess
    _log(f"{slide}: pseudo-Visium dataset")
    meta = preprocess.preprocess_sample(layout, slide)
    _log(f"{slide}: {meta['n_spots']} spots, {meta['n_nuclei_total']} nuclei "
         f"({meta['n_nuclei_no_other']} typed), mpp {meta['mpp']:.4f}")


@REGISTRY.stage("sthelar", "crops", "dataset", needs=["slides"],
                resources=Resources(cpus=8, mem_gb=96, hours=4),
                doc="materialize the 6 cropped datasets (the 2 as-is slides are a no-op)", done=_done_crops)
def stage_crops(layout: Layout, name: str) -> None:
    from . import crop
    if name not in config.STHELAR_CROPS:
        _log(f"{name}: used whole, nothing to crop")
        return
    _log(f"{name}: crop")
    crop.crop_dataset(layout, name)


@REGISTRY.stage("sthelar", "stage", "dataset", needs=["crops"],
                resources=Resources(cpus=4, mem_gb=96, hours=3),
                doc="reflect-pad the H&E and build the tissue mask for HoVerNet", done=_done_stage)
def stage_stage(layout: Layout, name: str) -> None:
    from . import hovernet_stage
    _log(f"{name}: stage padded H&E")
    hovernet_stage.stage_inputs(layout, name)
    _log(f"{name}: tissue mask")
    hovernet_stage.run_mask(layout, name)


@REGISTRY.stage("sthelar", "hovernet", "dataset", needs=["stage"],
                resources=Resources(cpus=8, mem_gb=64, gpu=True, hours=4),
                doc="HoVerNet WSI inference (measured 55 min - 1 h 05 per dataset)", done=_done_hovernet)
def stage_hovernet(layout: Layout, name: str) -> None:
    _log(f"{name}: HoVerNet inference")
    from . import hovernet_stage
    hovernet_stage.run_infer(layout, _bridge(), name)


@REGISTRY.stage("sthelar", "post", "dataset", needs=["hovernet"],
                resources=Resources(cpus=8, mem_gb=96, hours=3),
                doc="spot-proximity filter, reindex, copies, spot_dict and sim/ levels", done=_done_post)
def stage_post(layout: Layout, name: str) -> None:
    from . import postprocess, sim
    _log(f"{name}: check the HoVerNet output")
    _log(f"{name}: {json.dumps(postprocess.check_staging(layout, name))}")
    _log(f"{name}: spot-proximity filter + reindex")
    _log(f"{name}: {json.dumps(postprocess.filter_and_index(layout, name))}")
    postprocess.copy_he(layout, name)
    postprocess.copy_anndata(layout, name)
    _log(f"{name}: he.tiff + anndata copied")
    _log(f"{name}: {json.dumps(postprocess.build_spot_dict(layout, name))}")
    _log(f"{name}: sim/ levels")
    _log(f"{name}: {json.dumps(sim.build_sim(layout, name))}")


@REGISTRY.stage("sthelar", "cellvit", "dataset", needs=["post"],
                resources=Resources(cpus=8, mem_gb=64, gpu=True, hours=4),
                doc="CellViT-SAM, the second backend (measured 19 - 31 min) -> cellvit.json", done=_done_cellvit)
def stage_cellvit(layout: Layout, name: str) -> None:
    from . import cellvit_stage
    _log(f"{name}: CellViT (SAM / pannuke)")
    _log(f"{name}: {json.dumps(cellvit_stage.run_infer(layout, _bridge(), name))}")


@REGISTRY.stage("sthelar", "hoptimus", "dataset", needs=["post"],
                resources=Resources(cpus=6, mem_gb=48, gpu=True, hours=2),
                doc="H-Optimus-0 tile embeddings (measured 9 - 25 min per dataset on a P100)", done=_done_hoptimus)
def stage_hoptimus(layout: Layout, name: str) -> None:
    from . import hoptimus
    _log(f"{name}: H-Optimus-0 tile embeddings")
    _log(f"{name}: {json.dumps(hoptimus.extract(layout, _bridge(), name))}")


@REGISTRY.stage("sthelar", "plots", "dataset", needs=["post"],
                resources=Resources(cpus=4, mem_gb=32, hours=2),
                doc="the bench_data figures: overviews, per-level segmentation, pies, tree", done=_done_plots)
def stage_plots(layout: Layout, name: str) -> None:
    from . import plots
    _log(f"{name}: plots")
    _log(f"{name}: {json.dumps(plots.make_all_plots(layout, _bridge(), name))}")


@REGISTRY.stage("sthelar", "verify", "dataset",
                needs=["post", "cellvit", "hoptimus", "plots"],
                resources=Resources(cpus=4, mem_gb=48, hours=1),
                doc="independent checks of everything written, against the manifest")
def stage_verify(layout: Layout, name: str) -> None:
    from . import verify
    report = verify.verify(layout, name)
    _log(f"{name}: {json.dumps(report, indent=2)}")
    if not report["ok"]:
        raise SystemExit(f"{name}: verification failed with {len(report['issues'])} issues")
