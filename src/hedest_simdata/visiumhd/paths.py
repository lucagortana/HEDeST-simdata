"""Builds a :class:`~hedest_simdata.visiumhd.config.Sample` from the shared Layout.

Replaces ``pseudovisium.config.from_args``, which read argparse. The field names are
unchanged, so every ported module reads as it did; only where the paths come from changed.
"""
from __future__ import annotations

from pathlib import Path

from .. import config as root_config
from ..config import Layout
from .config import DEFAULT_LEVELS
from .config import Sample


def sample(layout: Layout, patient: str) -> Sample:
    """Every path for one patient, rooted at the layout.

    The deliverables go to ``<DATA_ROOT>/<name>/pseudovisium/`` and ``.../pyr_tif/``; the
    bin2cell and segmentation intermediates to ``_work/visiumhd/<name>/``, because nothing
    reads them once ``pseudovisium/`` exists.
    """
    if patient not in root_config.VISIUMHD_PATIENTS:
        raise KeyError(f"unknown patient {patient!r}; "
                       f"known: {root_config.VISIUMHD_PATIENTS}")
    name = root_config.visiumhd_name(patient)
    data = layout.visiumhd_dir(patient)
    work = layout.visiumhd_work(patient)
    binned = data / "binned_outputs"

    full = sorted(data.glob(f"{name}_tissue_image.*"))
    return Sample(
        name=name,
        data_dir=data,
        # kept as an optional cross-check only; the labels come from the committed CSV
        annotated=data / f"{name}_annotated.h5ad",
        pyr_tif=layout.visiumhd_pyr_tif(patient),
        scalefactors=binned / "square_002um" / "spatial" / "scalefactors_json.json",
        full_image=full[0] if full else None,
        hovernet_json=work / "hovernet" / "seg_json" / f"{name}_capture.json",
        label_mask=work / "hovernet" / "he_nuclei.npz",
        nucleus_table=work / "hovernet" / "nucleus_types.csv",
        bin_data=work / "bin2cell" / "bin_data.h5ad",
        cell_data=work / "bin2cell" / "cell_data.h5ad",
        out_dir=layout.visiumhd_out(patient),
        work_dir=work,
        levels=list(DEFAULT_LEVELS),
        unassigned_pattern=root_config.VISIUMHD_UNASSIGNED_PATTERN,
        unassigned_types=list(root_config.VISIUMHD_UNASSIGNED_TYPES.get(patient, [])),
        spot_diameter_um=root_config.SPOT_DIAMETER_UM,
        spot_spacing_um=root_config.SPOT_SPACING_UM,
    )


def patient_of(s: Sample) -> str:
    """``P1`` from ``Visium_HD_Human_Colon_Cancer_P1``."""
    return s.name.rsplit("_", 1)[-1]


def raw_inputs(s: Sample) -> dict[str, Path]:
    """What the download stage must have produced, by role.

    The top-level ``spatial/`` folder is **not** in here, although the original
    ``run_bin2cell.sh`` passed it: P5's published download does not contain one, and its
    bin2cell run succeeded anyway, because the spatial outputs are also inside each bin
    size's own folder. Requiring it would reject a complete download.
    """
    binned = s.data_dir / "binned_outputs"
    return {
        "tissue_image": s.full_image or (s.data_dir / f"{s.name}_tissue_image.btf"),
        "scalefactors": s.scalefactors,
        "bins_002um": binned / "square_002um",
        "bins_008um": binned / "square_008um",
        "bins_016um": binned / "square_016um",
        "bin_spatial": binned / "square_002um" / "spatial",
    }


def spaceranger_spatial(s: Sample) -> Path:
    """The spatial outputs folder to hand to ``bin2cell.read_visium``.

    Prefers the top-level ``spatial/`` when the download has one, since that is what the
    original pipeline passed, and otherwise falls back to the 2 um bin's own copy -- which
    holds the same files and is always present.
    """
    top = s.data_dir / "spatial"
    return top if top.is_dir() else s.data_dir / "binned_outputs" / "square_002um" / "spatial"
