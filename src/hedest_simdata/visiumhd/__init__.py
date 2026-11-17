"""Visium HD: three colon-cancer samples -> cell-level data -> pseudo-Visium.

    download -> bin2cell -> seg -> spots -> embed -> plots -> verify

Ported from ``VisiumHD_analysis`` (``analysis/preprocessing`` plus ``pseudovisium/``), with
four changes:

* **the cell types are committed, not recomputed.** They were assigned by hand, so they
  ship as ``annotations/visiumhd/*.csv.gz`` and are re-keyed onto freshly segmented nuclei
  **by centroid**, with a shifted control -- ``object_id`` is not a key across runs, which
  cost a build to find out; see :mod:`hedest_simdata.visiumhd.cell_labels`. The 4.7 GB
  ``*_annotated.h5ad`` is no longer an input.
* **``pyr_tif`` is not a stage.** bin2cell's ``prepare_he_image`` writes the pyramidal
  capture-area TIFF on its way to segmenting it, so the deliverable comes out of
  ``bin2cell`` rather than a step of its own.
* **one H-Optimus implementation.** ``embed`` calls the same bridge function STHELAR and
  Xenium do, instead of its own copy of the tile-inference loop -- which in any case
  imported a HEDeST path that no longer exists.
* **no README written into the data folder.**

``P1`` carries a per-sample exclusion that is not optional; see
:data:`hedest_simdata.config.VISIUMHD_UNASSIGNED_TYPES`.
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

#: bin2cell settings the reference datasets were built with. ``run_bin2cell.sh`` passed
#: these on the command line; they are constants here so a rebuild cannot differ by a
#: forgotten flag. ``use_gex_segmentation=False`` is the ``--no-gex`` that script passed.
BIN2CELL = {"mpp": 0.5, "segmenter": "hovernet", "buffer": 150,
            "use_gex_segmentation": False, "hovernet_batch_size": 16}


def _bridge() -> Bridge:
    if BRIDGE is None:  # pragma: no cover - the driver always sets it
        raise RuntimeError("the HEDeST bridge was not configured; run via make_simdata.py")
    return BRIDGE


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# "already built?" predicates -- each names the file its stage writes last
# --------------------------------------------------------------------------------------


def _done_download(layout: Layout, patient: str) -> bool:
    s = paths.sample(layout, patient)
    raw = paths.raw_inputs(s)
    return all(p.exists() for p in raw.values())


def _done_bin2cell(layout: Layout, patient: str) -> bool:
    s = paths.sample(layout, patient)
    return s.cell_data.exists() and s.hovernet_json.exists() and s.pyr_tif.exists()


def _done_seg(layout: Layout, patient: str) -> bool:
    s = paths.sample(layout, patient)
    return (s.out_dir / "sim" / "cell_types.csv").exists()


def _done_spots(layout: Layout, patient: str) -> bool:
    s = paths.sample(layout, patient)
    return (s.out_dir / "sim" / "levels.json").exists()


def _done_embed(layout: Layout, patient: str) -> bool:
    # The file alone is not enough. `embed` writes the `.pt` through the bridge, then
    # re-embeds 64 nuclei on their own to show each embedding depends only on its own tile,
    # and only then records the result. A crash in between -- there was one, a call to a
    # `load_nuc` this package's `hovernet` does not have -- leaves a complete-looking `.pt`
    # that this predicate used to accept, so the stage was pruned on the next run and the
    # check it exists for never ran at all.
    from .config import read_metadata
    from .embed import OUT_NAME
    out_dir = paths.sample(layout, patient).out_dir
    if not (out_dir / OUT_NAME).exists():
        return False
    try:
        return "embed" in read_metadata(out_dir)
    except (OSError, ValueError):
        return False


def _done_plots(layout: Layout, patient: str) -> bool:
    # 07_matching is the last figure drawn
    return (paths.sample(layout, patient).out_dir / "plots" / "07_matching.png").exists()


# --------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------


@REGISTRY.stage("visiumhd", "download", "patient",
                resources=Resources(cpus=2, mem_gb=4, hours=12), done=_done_download,
                doc="fetch the binned outputs, the btf H&E, the spatial outputs")
def stage_download(layout: Layout, patient: str) -> None:
    from . import fetch
    fetch.run(layout, patient)


@REGISTRY.stage("visiumhd", "bin2cell", "patient", needs=["download"],
                # 120 GB, not more: every cbio GPU node has ~126 GB, so a larger
                # request is never schedulable and the task waits forever instead of
                # failing.
                #
                # 4 h, not 2. The 2 h came from reference runs of 6-20 min (P1 11:31,
                # P2 12:34) -- but those reused an existing segmentation, which this
                # stage's own code reports as "tissue mask exists, reusing". A cold run
                # adds a full HoVerNet WSI pass, and P5's capture crop is 533 Mpx: the
                # Xenium slides measured 32 min at 522 Mpx and 54 min at 746 Mpx, so
                # about 35 min here, on top of ~25 min of read/destripe/StarDist observed
                # before HoVerNet started on 2026-11-08. Roughly 70 min cold, so 4 h is
                # ~3x headroom rather than a guess.
                #
                # max_concurrent=1: this stage's HoVerNet call keeps HoVerNet's own worker
                # counts (8 and 16), which is what the delivered data was built with and
                # which the bridge lowers everywhere else to stay inside /dev/shm. Running
                # two of these on one node is what that lowering exists to prevent, so
                # serialise the stage rather than change the numbers and move the cell ids.
                resources=Resources(cpus=8, mem_gb=120, gpu=True, hours=4,
                                    env="bin2cell", max_concurrent=1),
                done=_done_bin2cell,
                doc="pyramidal TIFF, HoVerNet nuclei and the bin2cell cell-level AnnData")
def stage_bin2cell(layout: Layout, patient: str) -> None:
    from .visiumhd_to_cell import process_visiumhd
    s = paths.sample(layout, patient)
    raw = paths.raw_inputs(s)
    _log(f"{patient}: bin2cell ({BIN2CELL['segmenter']}, buffer {BIN2CELL['buffer']})")
    s.cell_data.parent.mkdir(parents=True, exist_ok=True)
    s.pyr_tif.parent.mkdir(parents=True, exist_ok=True)
    process_visiumhd(
        binned_output_path=str(raw["bins_002um"]),
        he_image_path=str(raw["tissue_image"]),
        spaceranger_spatial_path=str(paths.spaceranger_spatial(s)),
        output_dir=str(layout.visiumhd_work(patient)),
        pyr_tif_dir=str(s.pyr_tif.parent),
        bridge=_bridge(),
        **BIN2CELL)


@REGISTRY.stage("visiumhd", "seg", "patient", needs=["bin2cell"],
                resources=Resources(cpus=8, mem_gb=160, hours=4), done=_done_seg,
                doc="re-key the committed cell types onto the nuclei by centroid, and "
                    "prove it")
def stage_seg(layout: Layout, patient: str) -> None:
    from . import stages
    s = paths.sample(layout, patient)
    missing = s.missing()
    if missing:
        raise FileNotFoundError("missing inputs:\n  " + "\n  ".join(missing))
    _log(f"{patient}: seg -> {s.out_dir}")
    stages.stage_seg(s)


@REGISTRY.stage("visiumhd", "spots", "patient", needs=["seg"],
                resources=Resources(cpus=8, mem_gb=160, hours=4), done=_done_spots,
                doc="honeycomb grid, 2 um bin sums, proportions, spot_dict, the AnnData")
def stage_spots(layout: Layout, patient: str) -> None:
    from . import spots
    from .config import read_metadata
    s = paths.sample(layout, patient)
    _log(f"{patient}: spots")
    spots.run(s, read_metadata(s.out_dir)["seg"])


@REGISTRY.stage("visiumhd", "embed", "patient", needs=["spots"],
                resources=Resources(cpus=6, mem_gb=48, gpu=True, hours=2),
                done=_done_embed,
                doc="H-Optimus-0 tile embeddings (measured ~16 min per sample on a P100)")
def stage_embed(layout: Layout, patient: str) -> None:
    from . import embed
    s = paths.sample(layout, patient)
    _log(f"{patient}: H-Optimus-0 tile embeddings")
    _log(f"{patient}: {json.dumps({k: v for k, v in embed.run(s, _bridge()).items() if k != 'hedest'})}")


@REGISTRY.stage("visiumhd", "plots", "patient", needs=["spots"],
                resources=Resources(cpus=4, mem_gb=160, hours=2), done=_done_plots,
                doc="the 20 verification figures")
def stage_plots(layout: Layout, patient: str) -> None:
    from . import plots
    s = paths.sample(layout, patient)
    _log(f"{patient}: plots")
    plots.run(s)


@REGISTRY.stage("visiumhd", "verify", "patient", needs=["spots", "embed", "plots"],
                resources=Resources(cpus=4, mem_gb=48, hours=1),
                doc="independent checks of everything written, against the manifest")
def stage_verify(layout: Layout, patient: str) -> None:
    from . import verify
    s = paths.sample(layout, patient)
    verify.run(s)
