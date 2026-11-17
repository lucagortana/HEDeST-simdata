"""STHELAR-specific constants: the taxonomy, the TIFF export and the figure settings.

Split out of ``hedest_bench.config``, whose other half -- the one path everything resolved
from -- is now :class:`~hedest_simdata.config.Layout`. Nothing here is a path, so nothing
here has to be edited before a run.
"""
from __future__ import annotations

from ..config import SPOT_DIAMETER_UM
from ..config import SPOT_SPACING_UM

#: Re-exported, not redefined: STHELAR and Visium HD build the same Visium v2 lattice, so
#: the geometry lives in :mod:`hedest_simdata.config` and is named here only so the ported
#: modules keep reading ``constants.SPOT_*`` as they always did.
__all__ = ["SPOT_DIAMETER_UM", "SPOT_SPACING_UM"]

# --- Cell-type ground truth -----------------------------------------------------------
# Two annotation levels are produced for every slide:
#   * "original" -- the coarse harmonised taxonomy (cells_final_label_group),
#   * "label2"   -- the finer curated taxonomy (cells_label2), which nests strictly
#                   inside the coarse one (every subtype belongs to exactly one coarse
#                   group; verified per slide).
CELLTYPE_COL = "cells_final_label_group"       # coarse / "original"
CELLTYPE_COL_FINE = "cells_label2"             # fine-grained / "label2"
ANNOT_ORIGINAL = "original"
ANNOT_FINE = "label2"

#: Catch-all collapsing the low-quality groups (Unknown / Less10 / Stem_like). It is kept
#: in the ``with_other`` DAPI segmentation JSON and dropped everywhere else. The *same*
#: cells are relabelled in the label2 taxonomy, so the two annotations agree exactly on
#: which cells are Other.
OTHER_LABEL = "Other"

#: Layer holding the per-spot log-normalised counts; ``X`` keeps the raw summed counts.
LOGNORM_LAYER = "log_norm"

# --- Xenium global coordinate system --------------------------------------------------
# Boundaries and transcripts are stored in microns; the "global" (morphology image) frame
# is in pixels of this size. The nucleus shapes carry a scale transform == 1 / mpp, which
# is read per slide, so this is only a documented fallback.
MORPHO_MPP_UM = 0.2125

# --- H&E TIFF export ------------------------------------------------------------------
TIFF_TILE = (512, 512)
TIFF_COMPRESSION = "zlib"
TIFF_N_PYRAMID_LEVELS = 6        # max pyramid sub-levels to copy from the zarr

# --- bench_data figures ---------------------------------------------------------------
#: Side of the detail window the per-level segmentation figures are drawn in, in px.
PLOT_WINDOW_PX = 2000
PLOT_SEED = 0

#: Whole-slide overview plots use the finest H&E pyramid level whose longest side is at
#: most this many pixels.
SLIDE_PLOT_MAX_DIM = 3000
