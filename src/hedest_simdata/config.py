"""Where everything lives, and the registry of what gets built.

Unlike the three pipelines this repo replaces, **no path is a module constant**. The data
root arrives from the command line (``make_simdata.py <DATA_ROOT>``) and is carried in a
:class:`Layout`, so the same process can build into a scratch folder and into the real one
without editing a file.

Layout under ``<DATA_ROOT>``::

    STHELAR/bench_data/<name>/          the 8 benchmark datasets      [deliverable]
    Visium_HD_Human_Colon_Cancer_<P>/   pseudovisium/ + pyr_tif/       [deliverable]
    Xenium_<slide>/sim[/LuCA]/          the 6 files per slide          [deliverable]
    _work/                              everything transient          [removed on success]

``_work/`` matters: the STHELAR benchmark is built from per-slide pseudo-Visium datasets
and from crops of them, and those are ~117 GB that nothing downstream reads --
``bench_data/<name>/`` keeps its own copies of ``he.tiff`` and ``pseudovisium.h5ad``. They
are intermediates, so they go to ``_work`` and are deleted unless ``--keep-work``.

The Xenium output paths are deliberately the historical ones (breast in ``sim/``, lung in
``sim/LuCA/``): ``HEDeST/gridsearch/semi_sim.yaml`` hard-codes them, and a regenerated
folder has to be usable by that config without editing it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------------------
# STHELAR
# --------------------------------------------------------------------------------------

#: The seven source slides, as named in the STHELAR archive.
STHELAR_SLIDES = ["breast_s6", "cervix_s0", "lung_s3", "lymph_node_s0", "ovary_s1",
                  "prostate_s0", "skin_s4"]

#: ``crop name -> (source slide, (x0, y0, x1, y1) in scale-0 H&E px)``. Clipped to the
#: image at run time (only breast's x1 exceeds the slide, by ~90 px). ``cervix_s0`` is
#: split in two size-driven halves.
STHELAR_CROPS: dict[str, tuple[str, tuple[int, int, int, int]]] = {
    "breast_s6":     ("breast_s6",     (21888, 8064, 57472, 25344)),
    "cervix_s0_0":   ("cervix_s0",     (0, 2880, 37440, 23040)),
    "cervix_s0_1":   ("cervix_s0",     (960, 23040, 33600, 36480)),
    "lung_s3":       ("lung_s3",       (0, 0, 25920, 25920)),
    "lymph_node_s0": ("lymph_node_s0", (7680, 3072, 26112, 24576)),
    "ovary_s1":      ("ovary_s1",      (8640, 0, 30720, 38400)),
}

#: Slides used whole, with no crop.
STHELAR_ASIS = ["prostate_s0", "skin_s4"]

#: The 8 benchmark datasets, crops first.
BENCH_NAMES = list(STHELAR_CROPS) + STHELAR_ASIS

# --------------------------------------------------------------------------------------
# Visium HD
# --------------------------------------------------------------------------------------

#: Patient codes; the folder is ``Visium_HD_Human_Colon_Cancer_<code>``.
VISIUMHD_PATIENTS = ["P1", "P2", "P5"]

#: The three annotation levels, coarse to fine, as columns of the annotation table.
VISIUMHD_LEVELS = ["lv1_annot", "lv2_annot", "lv3_annot"]

#: Case-insensitive pattern for annotation categories that mean "no call". A cell
#: unassigned at any level is dropped from every level.
VISIUMHD_UNASSIGNED_PATTERN = "unassigned"

#: Per-sample categories to treat as unassigned although they are not named "unassigned",
#: as ``lv<k>:NAME`` (the level index, not the column name).
#:
#: **P1 needs one, and it is not optional.** Its finest level carries a bare ``Epithelial``
#: for 5,718 cells -- not a cell type but an absence of one, a cell the annotation could
#: not resolve past the compartment. The published P1 dataset was built with it excluded,
#: so leaving it in regenerates 13 fine types where the reference has 12 and shifts every
#: P1 proportion. Recovered from ``metadata.json``'s own record of the run:
#: ``{"pattern": "unassigned", "explicit_types": ["lv2:Epithelial"],
#: "cells_removed": 17306}``.
VISIUMHD_UNASSIGNED_TYPES: dict[str, list[str]] = {
    "P1": ["lv2:Epithelial"],
    "P2": [],
    "P5": [],
}

#: How many cells the two rules together removed in the reference run, so a rebuild that
#: drops a different number fails instead of quietly disagreeing.
VISIUMHD_CELLS_REMOVED = {"P1": 17_306, "P2": 585, "P5": 0}


def visiumhd_name(patient: str) -> str:
    return f"Visium_HD_Human_Colon_Cancer_{patient}"


# --------------------------------------------------------------------------------------
# Xenium
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class XeniumSlide:
    """One Xenium slide, and every choice its published dataset was built with.

    The two notebooks this comes from explored several variants and were last saved in a
    configuration that is **not** the one that shipped, so none of these values can be read
    off the code. Each is pinned from the delivered files instead, and ``verify`` checks the
    rebuild against them.
    """

    name: str           # folder under <DATA_ROOT>
    code: str           # "HL" / "HB", used in the embedding file name
    sim_subdir: str     # "sim" or "sim/LuCA" -- historical, pinned by semi_sim.yaml

    diameter: float
    spacing: float

    #: How HoVerNet nuclei are tied to Xenium cells.
    #:
    #: ``closest``    the HoVerNet nucleus is matched to the nearest Xenium **nucleus**,
    #:                kept when the centroid distance is under ``match_threshold`` px, and
    #:                one HoVerNet nucleus per Xenium nucleus -- the closest.
    #: ``intersect``  the HoVerNet nucleus is intersected with the Xenium **cell**
    #:                boundary and kept when the overlap ratio clears ``match_threshold``.
    #:
    #: Both slides use ``closest``, which had to be **measured** rather than read: each
    #: notebook contains both routes, side by side, with the intersect one under a heading
    #: that reads like the one that was used, and their saved cell outputs are from
    #: inconsistent executions (one cell prints Xenium barcodes as ``nucleus_id``, the next
    #: prints HoVerNet integers). Resolved by rebuilding each candidate and comparing the
    #: resulting nucleus-id **sets** against the delivered annotated segmentation, on lung:
    #:
    #:     closest,   distance < 40   117,998 of 118,005 reference nuclei   99.99%
    #:     closest,   distance < 20   113,224                               95.95%
    #:     intersect, overlap >= 0.10 102,574                               86.92%
    #:     intersect, overlap >= 0.55  97,623                               82.73%
    #:
    #: So the intersect route is not what produced these files. With ``closest`` and the
    #: nucleus-to-cell hop below, breast rebuilds exactly and lung to within 4 ground-truth
    #: rows in 107,429; see :mod:`hedest_simdata.xenium.build` for the full table.
    match: str
    match_threshold: float


    #: Cell types, in the order the delivered ``*_gt.csv`` and ``*_prop_real.csv`` carry
    #: them. The order is a contract, not a presentation choice: ``runs.LUNG_COLORS`` /
    #: ``BREAST_COLORS`` are zipped against these columns and ``load_analyzer`` raises if
    #: they disagree. In the notebooks it fell out of a ``.unique()`` call, which would
    #: change silently with the data; here it is explicit.
    cell_types: tuple[str, ...]

    #: Categories dropped before anything is written, and why. Every family in this
    #: repository has a catch-all label that is an absence of a call rather than a cell
    #: type, and it is named differently every time: ``Other`` in STHELAR, ``unassigned``
    #: in Visium HD, ``Unlabeled`` here on breast, ``low_counts`` here on lung.
    drop_types: tuple[str, ...]

    #: ``vendor``: the published spreadsheet on the dataset page.
    #: ``committed``: a CSV in ``annotations/xenium/``, because reproducing it would mean
    #: shipping a single-cell reference and a label-transfer run.
    annotation: str

    #: Overlap a Xenium **nucleus** must have with a Xenium **cell** to inherit its type,
    #: or ``None`` when the slide's nuclei *are* its cells.
    #:
    #: This is the real difference between the two slides, and it is a property of the
    #: data, not a choice. On **breast** (an earlier Xenium release) ``nucleus_boundaries``
    #: has exactly one entry per cell, keyed by the same ``cell_id``: 167,780 nuclei for
    #: 167,780 cells, so a matched nucleus already names its cell. On **lung** (the
    #: multimodal-segmentation release) nuclei are separate objects in their own id space
    #: -- 156,628 of them, and only 6 of the first 5,000 ids appear among the cell ids at
    #: all -- so a matched nucleus says nothing about cell type until it is joined to the
    #: cell that contains it. Hence the extra hop, and hence lung's chain is
    #: HoVerNet nucleus -> Xenium nucleus -> Xenium cell -> cell type.
    nucleus_to_cell_overlap: float | None = None

    #: A spot is kept only if it holds at least this many annotated nuclei.
    min_nuclei_per_spot: int = 4

    #: How many spots the **delivered** dataset has, measured from its own
    #: ``spot_dict_real.json``. An anchor outside the code: ``verify`` already checked that
    #: no kept spot holds fewer than :attr:`min_nuclei_per_spot`, which passes whenever the
    #: build and the setting agree with each other -- and they did, at a setting that was
    #: wrong for lung and silently dropped 144 of its spots. A count taken from the
    #: delivered data cannot be satisfied by the code agreeing with itself.
    delivered_spots: int | None = None

    # Both slides use the published ``*_he_imagealignment.csv``, and there is no setting
    # for it. Breast carried a committed affine until 2026-11-08, on the belief that the
    # published one was a different alignment -- it is the same alignment for a cropped
    # image; see :mod:`hedest_simdata.xenium.fetch` for the measurement that settled it.


XENIUM_SLIDES = {
    "lung": XeniumSlide(
        name="Xenium_V1_humanLung_Cancer_FFPE", code="HL", sim_subdir="sim/LuCA",
        diameter=283.0, spacing=440.0,
        match="closest", match_threshold=40.0,
        cell_types=("T cell", "Epithelial cell", "Macrophage/Monocyte", "Stromal",
                    "Plasma cell", "B cell", "Endothelial cell"),
        # the annotation calls 13 types; five are too rare to learn (NK cell 66 cells,
        # pDC 16, Neutrophils 202, cDC 1,607, Mast cell 1,179) and low_counts (21,151) is
        # not a type at all, which is how 13 becomes the 7 above
        drop_types=("Mast cell", "cDC", "Neutrophils", "NK cell", "pDC", "low_counts"),
        # lung's nuclei are their own objects, so they need joining to a cell first
        nucleus_to_cell_overlap=0.95,
        # 1, not the default 4. Measured against the delivered spot_dict_real.json: lung
        # keeps 54 spots holding a single annotated nucleus, 50 holding two and 40 holding
        # three, so the default dropped exactly those 144 of its 1,957 spots -- while breast
        # has none below four and so genuinely used the default. The two differ because
        # lung's spots are sparser, and a threshold shared by both looked right for as long
        # as only breast was compared.
        min_nuclei_per_spot=1,
        delivered_spots=1957,
        annotation="committed"),
    "breast": XeniumSlide(
        name="Xenium_FFPE_Human_Breast_Cancer_Rep1", code="HB", sim_subdir="sim",
        diameter=350.0, spacing=500.0,
        match="closest", match_threshold=40.0,
        cell_types=("Tumor_Cells", "Macrophages", "Stromal_Cells", "Myoepithelial_Cells",
                    "T_Cells", "Endothelial_Cells", "B_Cells"),
        drop_types=("Unlabeled",),
        # breast's nucleus_boundaries is one entry per cell, keyed by cell_id: no hop
        nucleus_to_cell_overlap=None,
        # no min_nuclei_per_spot here: breast's delivered spots genuinely start at four,
        # so the shared default is right for this slide and only this slide
        delivered_spots=2651,
        annotation="vendor"),
}

#: Breast only: the published spreadsheet's clusters collapsed into working types, then
#: anything left with fewer than :data:`XENIUM_BREAST_MIN_CELLS` cells folded into
#: ``Unlabeled`` and dropped. Reproduced from the notebook; the rule's output is checked
#: against ``XENIUM_SLIDES["breast"].cell_types`` rather than trusted, because a
#: count-based rule depends on the data it is run over.
XENIUM_BREAST_CLUSTER_MAP = {
    "CD4+_T_Cells": "T_Cells", "CD8+_T_Cells": "T_Cells",
    "T_Cell_&_Tumor_Hybrid": "T_Cells",
    "Macrophages_1": "Macrophages", "Macrophages_2": "Macrophages",
    "IRF7+_DCs": "Dendritic_Cells", "LAMP3+_DCs": "Dendritic_Cells",
    "Myoepi_ACTA2+": "Myoepithelial_Cells", "Myoepi_KRT15+": "Myoepithelial_Cells",
    "Invasive_Tumor": "Tumor_Cells", "Prolif_Invasive_Tumor": "Tumor_Cells",
    "DCIS_1": "Tumor_Cells", "DCIS_2": "Tumor_Cells",
    "Stromal": "Stromal_Cells", "Stromal_&_T_Cell_Hybrid": "Stromal_Cells",
    "Endothelial": "Endothelial_Cells", "B_Cells": "B_Cells",
    "Perivascular-Like": "Perivascular_Cells", "Mast_Cells": "Mast_Cells",
    "Unlabeled": "Unlabeled",
}
XENIUM_BREAST_MIN_CELLS = 1000
XENIUM_BREAST_SHEET = "Xenium R1 Fig1-5 (supervised)"

#: The ``type`` code every nucleus carries in ``pannuke_fast_mask_lvl3_annotated.json``.
#: Codes 0..6 are the slide's :attr:`XeniumSlide.cell_types`, in that order, and then:
XENIUM_TYPE_CATCHALL_OFFSET = 0   # 7 == len(cell_types): the dropped catch-all type
XENIUM_TYPE_UNMATCHED_OFFSET = 1  # 8 == len(cell_types) + 1: no Xenium cell matched
#:
#: Read off the delivered files rather than inferred, because the notebooks maintained two
#: competing id spaces and were last saved in a third configuration that mapped everything
#: onto three PanNuke names. Measured 2026-11-05, and the measurement is unambiguous: each
#: code maps to one cell type with **100% purity** (lung code 0 is T cell for 5,757 of
#: 5,757 nuclei; breast code 0 is Tumor_Cells for 48,365 of 48,365), and the arithmetic
#: closes exactly on both slides --
#:
#:     lung    141,257 nuclei - 23,252 unmatched - 10,576 catch-all = 107,429 = GT rows
#:     breast  175,204 nuclei - 48,070 unmatched -  6,798 catch-all = 120,336 = GT rows
#:
#: so the ground truth is exactly the nuclei whose code is 0..6.

# --------------------------------------------------------------------------------------
# Pseudo-Visium v2 spot geometry (STHELAR and Visium HD)
# --------------------------------------------------------------------------------------

SPOT_DIAMETER_UM = 55.0
SPOT_SPACING_UM = 100.0

# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Layout:
    """Resolves every path from the one root the user passed."""

    data_root: Path
    work_root: Path

    @classmethod
    def build(cls, data_root: str | Path, work_dir: str | Path | None = None) -> "Layout":
        root = Path(data_root).expanduser().resolve()
        work = Path(work_dir).expanduser().resolve() if work_dir else root / "_work"
        return cls(data_root=root, work_root=work)

    # --- STHELAR ----------------------------------------------------------------------
    @property
    def sthelar_raw(self) -> Path:
        """Where the ``.zarr.zip`` archives and cell-metadata parquets are downloaded."""
        return self.data_root / "STHELAR" / "raw"

    @property
    def bench_root(self) -> Path:
        return self.data_root / "STHELAR" / "bench_data"

    def bench_dir(self, name: str) -> Path:
        return self.bench_root / name

    def sthelar_slide_work(self, slide: str) -> Path:
        """Intermediate per-slide pseudo-Visium dataset."""
        return self.work_root / "sthelar" / "slides" / slide

    def sthelar_crop_work(self, crop: str) -> Path:
        return self.work_root / "sthelar" / "crops" / crop

    def sthelar_dataset_source(self, name: str) -> Path:
        """The pseudo-Visium dataset ``bench_data/<name>`` is built from."""
        return (self.sthelar_crop_work(name) if name in STHELAR_CROPS
                else self.sthelar_slide_work(name))

    def hovernet_staging(self, name: str) -> Path:
        return self.work_root / "sthelar" / "hovernet" / name

    # --- Visium HD --------------------------------------------------------------------
    def visiumhd_dir(self, patient: str) -> Path:
        return self.data_root / visiumhd_name(patient)

    def visiumhd_out(self, patient: str) -> Path:
        return self.visiumhd_dir(patient) / "pseudovisium"

    def visiumhd_pyr_tif(self, patient: str) -> Path:
        name = visiumhd_name(patient)
        return self.visiumhd_dir(patient) / "pyr_tif" / f"{name}_capture.tif"

    def visiumhd_work(self, patient: str) -> Path:
        """bin2cell / segmentation intermediates (the old ``outputs/<name>/``)."""
        return self.work_root / "visiumhd" / visiumhd_name(patient)

    # --- Xenium -----------------------------------------------------------------------
    def xenium_dir(self, key: str) -> Path:
        return self.data_root / XENIUM_SLIDES[key].name

    def xenium_out(self, key: str) -> Path:
        return self.xenium_dir(key) / XENIUM_SLIDES[key].sim_subdir

    def xenium_work(self, key: str) -> Path:
        return self.work_root / "xenium" / XENIUM_SLIDES[key].name
