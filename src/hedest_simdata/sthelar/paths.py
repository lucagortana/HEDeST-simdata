"""Every path of the STHELAR family, bound to a :class:`~hedest_simdata.config.Layout`.

This replaces ``hedest_bench.config.DATA_ROOT`` (a module constant that had to be edited
before a run) and merges ``bench/datasets.py`` with the staging half of
``hier/datasets.py``. The property names are kept identical to the originals, so the
ported modules read the same; only the construction changes, from ``datasets.get(name)``
to ``paths.dataset(layout, name)``.

Two things are deliberately different from HEDeST-bench:

* **The staging area is not called ``hier_sim``.** In HEDeST-bench the HoVerNet output
  was read out of ``<DATA_ROOT>/hier_sim/<name>/_staging``, a folder named after a
  different, now-deleted dataset family -- its own README calls the name "historical".
  Here it is ``_work/sthelar/hovernet/<name>``, which is what it is.
* **The source pseudo-Visium datasets live in ``_work``.** They are inputs to the
  benchmark, not deliverables: ``bench_data/<name>/`` takes its own copies of ``he.tiff``
  and ``pseudovisium.h5ad``, so nothing downstream reads them once it is built.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from .. import config
from ..config import Layout


@dataclass(frozen=True)
class Slide:
    """One source slide: the raw archive and the pseudo-Visium dataset built from it."""

    layout: Layout
    name: str

    @property
    def zarr_zip(self) -> Path:
        return self.layout.sthelar_raw / f"sdata_{self.name}.zarr.zip"

    @property
    def zarr_inner_root(self) -> str:
        """Top-level folder inside the ``.zarr.zip`` archive."""
        return f"sdata_{self.name}.zarr"

    @property
    def cell_metadata(self) -> Path:
        return self.layout.sthelar_raw / f"{self.name}_cell_metadata.parquet"

    @property
    def out_dir(self) -> Path:
        return self.layout.sthelar_slide_work(self.name)

    @property
    def he_tiff(self) -> Path:
        return self.out_dir / "he.tiff"

    @property
    def adata_path(self) -> Path:
        return self.out_dir / "pseudovisium.h5ad"

    @property
    def metadata_path(self) -> Path:
        return self.out_dir / "metadata.json"

    @property
    def dapi_seg_dir(self) -> Path:
        return self.out_dir / "dapi_seg"

    @property
    def proportions_dir(self) -> Path:
        return self.out_dir / "proportions"


@dataclass(frozen=True)
class Dataset:
    """One of the 8 benchmark datasets: a crop of a slide, or a slide used whole."""

    layout: Layout
    name: str
    is_crop: bool

    # --- source: the pseudo-Visium dataset in _work (read-only from here on) ----------
    @property
    def source_dir(self) -> Path:
        return self.layout.sthelar_dataset_source(self.name)

    @property
    def source_slide(self) -> str:
        """The slide this dataset comes from."""
        return config.STHELAR_CROPS[self.name][0] if self.is_crop else self.name

    @property
    def he_tiff(self) -> Path:
        return self.source_dir / "he.tiff"

    @property
    def dapi_json(self) -> Path:
        return self.source_dir / "dapi_seg" / "label2_no_other.json"

    @property
    def dapi_mapping(self) -> Path:
        return self.source_dir / "dapi_seg" / "mapping_no_other.csv"

    @property
    def adata_path(self) -> Path:
        return self.source_dir / "pseudovisium.h5ad"

    @property
    def metadata_path(self) -> Path:
        return self.source_dir / "metadata.json"

    @property
    def prop_original(self) -> Path:
        return self.source_dir / "proportions" / "proportions_original.csv"

    @property
    def prop_label2(self) -> Path:
        return self.source_dir / "proportions" / "proportions_label2.csv"

    @cached_property
    def mpp(self) -> float:
        return float(json.loads(self.metadata_path.read_text())["mpp"])

    @property
    def adata_name(self) -> str:
        # adata.uns['spatial'] is always keyed by the dataset name
        return self.name

    # --- HoVerNet staging (in _work) --------------------------------------------------
    @property
    def staging_dir(self) -> Path:
        return self.layout.hovernet_staging(self.name)

    @property
    def staged_wsi_dir(self) -> Path:
        """Single-file input folder: ``run_infer.py`` names its outputs after the one
        file it finds there, so it cannot be pointed at a folder of slides."""
        return self.staging_dir / "wsi"

    @property
    def staged_wsi(self) -> Path:
        return self.staged_wsi_dir / f"{self.name}.tiff"

    @property
    def staged_mask_dir(self) -> Path:
        return self.staging_dir / "mask"

    @property
    def staged_mask(self) -> Path:
        return self.staged_mask_dir / f"{self.name}.png"

    @property
    def staged_seg(self) -> Path:
        """Raw HoVerNet output: every detected cell, unpadded, before the spot filter."""
        return self.staging_dir / f"{self.name}.json"

    @property
    def staged_image_dict(self) -> Path:
        return self.staging_dir / "image_dict_raw.pt"

    # --- outputs: the only place anything is written ----------------------------------
    @property
    def out_dir(self) -> Path:
        return self.layout.bench_dir(self.name)

    @property
    def hovernet_json(self) -> Path:
        """Spot-filtered, reindexed HoVerNet JSON with raw HoVerNet 0-5 types."""
        return self.out_dir / "hovernet.json"

    @property
    def image_dict(self) -> Path:
        return self.out_dir / "image_dict.pt"

    @property
    def cellvit_json(self) -> Path:
        """CellViT (SAM / PanNuke) segmentation, HoVerNet schema, same ids."""
        return self.out_dir / "cellvit.json"

    @property
    def cellvit_image_dict(self) -> Path:
        return self.out_dir / "cellvit_image_dict.pt"

    @property
    def spot_dict(self) -> Path:
        return self.out_dir / "spot_dict.json"

    @property
    def adata_out(self) -> Path:
        return self.out_dir / "pseudovisium.h5ad"

    @property
    def he_out(self) -> Path:
        return self.out_dir / "he.tiff"

    @property
    def hoptimus_embed(self) -> Path:
        """The feature file ``gridsearch/semi_sim.yaml`` reads for this dataset."""
        return self.out_dir / f"hoptimus_tile_embed_bench_{self.name}.pt"

    @property
    def sim_dir(self) -> Path:
        return self.out_dir / "sim"

    def level_dir(self, level: int) -> Path:
        return self.sim_dir / f"level{level}"

    @property
    def plots_dir(self) -> Path:
        return self.out_dir / "plots"


def slide(layout: Layout, name: str) -> Slide:
    if name not in config.STHELAR_SLIDES:
        raise KeyError(f"unknown slide {name!r}; known: {config.STHELAR_SLIDES}")
    return Slide(layout=layout, name=name)


def dataset(layout: Layout, name: str) -> Dataset:
    if name not in config.BENCH_NAMES:
        raise KeyError(f"unknown dataset {name!r}; known: {config.BENCH_NAMES}")
    return Dataset(layout=layout, name=name, is_crop=name in config.STHELAR_CROPS)


def all_datasets(layout: Layout) -> list[Dataset]:
    return [dataset(layout, n) for n in config.BENCH_NAMES]
