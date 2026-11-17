"""Every path of the Xenium family, bound to a Layout.

The output paths are the historical ones -- breast in ``sim/``, lung in ``sim/LuCA/`` --
because ``HEDeST/gridsearch/semi_sim.yaml`` hard-codes them and a regenerated folder has to
be usable by that config unchanged. The lung subfolder is named after the atlas its labels
were transferred from.

Everything that is not one of the six deliverables goes to ``_work``: the SpatialData
archive, the pyramidal H&E, the tissue mask and the raw segmentation are all intermediates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..config import Layout
from ..config import XeniumSlide


@dataclass(frozen=True)
class Slide:
    layout: Layout
    key: str
    spec: XeniumSlide

    # --- raw inputs -------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.layout.xenium_dir(self.key)

    @property
    def raw_outputs(self) -> Path:
        """The unpacked Xenium output bundle (cells.parquet, nucleus_boundaries, ...)."""
        return self.data_dir / "data"

    @property
    def he_ome_tif(self) -> Path | None:
        """The vendor H&E, whose extension differs between the two downloads."""
        for folder in (self.data_dir / "ome_tif", self.data_dir / "tif", self.data_dir):
            if folder.is_dir():
                hits = sorted(p for p in folder.iterdir()
                              if p.suffix in {".tif", ".tiff"} and "he" in p.name.lower())
                if hits:
                    return hits[0]
        return None

    @property
    def he_alignment(self) -> Path:
        return self.data_dir / f"{self.spec.name}_he_imagealignment.csv"

    @property
    def annotation_xlsx(self) -> Path:
        """Breast only: the published cell-type spreadsheet."""
        return self.data_dir / "Cell_Barcode_Type_Matrices.xlsx"

    # --- intermediates (in _work) ------------------------------------------------------
    @property
    def work(self) -> Path:
        return self.layout.xenium_work(self.key)

    @property
    def sdata_zarr(self) -> Path:
        return self.work / "data.zarr"

    @property
    def pyr_tif(self) -> Path:
        return self.work / "pyr_tif" / f"{self.spec.name}_he_image.tif"

    @property
    def mask_dir(self) -> Path:
        return self.work / "mask"

    @property
    def mask(self) -> Path:
        return self.mask_dir / f"{self.spec.name}_he_image.png"

    @property
    def seg_dir(self) -> Path:
        return self.work / "seg_json"

    @property
    def seg_json(self) -> Path:
        """Raw HoVerNet nuclei, before the Xenium types are attached.

        Directly under :attr:`seg_dir`, **not** in a ``json/`` subdirectory.
        ``infer/wsi.py`` picks between the two: ``output_dir/json/<name>.json`` when
        ``--save_mask`` or ``--save_thumb`` is given, ``output_dir/<name>.json`` otherwise.
        :meth:`~hedest_simdata.hedest_bridge.Bridge.hovernet_infer` passes neither, so it is
        always the flat one -- and the STHELAR stage, which clears exactly that flat path
        before re-segmenting, had it right all along.

        This said ``json/`` until 2026-11-08, which made :func:`_done_hovernet` permanently
        false and would have failed ``xenium:spots`` with a missing file after the hour of
        GPU work that produced it. It looked plausible because ``process_wsi_list``
        ``rm_n_mkdir``s ``json/`` unconditionally, so an empty directory of that name really
        is there next to the output.
        """
        return self.seg_dir / f"{self.spec.name}_he_image.json"

    @property
    def cache_dir(self) -> Path:
        return self.work / "cache"

    @property
    def hoptimus_all(self) -> Path:
        """Every nucleus's embedding, before the deliverable is cut to the ground truth.

        HEDeST keys embeddings by *position in the segmentation file*
        (``hedest/features/hoptimus.py``: ``cell_ids = [str(i) for i in range(len(nuclei))]``),
        so embedding a pre-filtered segmentation would renumber from zero and silently
        break the correspondence with the spot dictionary. The full pass is the only one
        whose keys mean anything; the delivered file is that pass **subset** to the
        ground-truth rows, which is why its keys run up to 141,256 (lung) and 175,202
        (breast) while holding only 107,429 and 120,336 entries.

        Writing the full dict to the deliverable and overwriting it in place would leave a
        complete-looking file behind if the job died between the two writes, and
        ``_done_hoptimus`` only checks that the file exists.
        """
        return self.work / "hoptimus" / f"all_cells_{self.spec.code}.pt"

    # --- the six deliverables ----------------------------------------------------------
    @property
    def out_dir(self) -> Path:
        return self.layout.xenium_out(self.key)

    @property
    def annotated_seg(self) -> Path:
        return self.out_dir / "pannuke_fast_mask_lvl3_annotated.json"

    @property
    def adata(self) -> Path:
        return self.out_dir / "pseudo_adata_real.h5ad"

    @property
    def spot_dict(self) -> Path:
        return self.out_dir / "spot_dict_real.json"

    @property
    def proportions(self) -> Path:
        return self.out_dir / f"sim_{self.spec.name}_prop_real.csv"

    @property
    def ground_truth(self) -> Path:
        return self.out_dir / f"sim_{self.spec.name}_gt.csv"

    @property
    def hoptimus_embed(self) -> Path:
        return self.out_dir / f"hoptimus_tile_embed_{self.spec.code}.pt"

    def deliverables(self) -> dict[str, Path]:
        return {"annotated_seg": self.annotated_seg, "adata": self.adata,
                "spot_dict": self.spot_dict, "proportions": self.proportions,
                "ground_truth": self.ground_truth, "hoptimus": self.hoptimus_embed}


def slide(layout: Layout, key: str) -> Slide:
    if key not in config.XENIUM_SLIDES:
        raise KeyError(f"unknown Xenium slide {key!r}; "
                       f"known: {list(config.XENIUM_SLIDES)}")
    return Slide(layout=layout, key=key, spec=config.XENIUM_SLIDES[key])
