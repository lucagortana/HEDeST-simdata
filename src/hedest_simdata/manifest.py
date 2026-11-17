"""What each pipeline must produce, declared once.

This is the contract. The pipelines' ``verify`` stages check against it, and
``tests/inventory.py`` checks a freshly built folder against a recording of the reference
folders, so "it matches what we already have" is a check rather than an opinion.

Every file carries a **comparison policy**, because almost nothing here is byte-stable:
a pyramidal TIFF depends on the zlib build, a matplotlib PNG on the freetype version, a
``.pt`` on the order tensors were stacked. Comparing bytes would report noise as failure
and hide real drift behind it. The policies say what actually has to agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from . import config

#: Comparison policies, from strictest to loosest.
#:
#: ``bytes``   identical bytes; for files this repo writes deterministically.
#: ``json``    parsed and compared as data, key order ignored, floats to ``FLOAT_RTOL``.
#:
#:             The tolerance is not slack. Rebuilding ``skin_s4`` reproduced the reference
#:             exactly -- same file set, bit-identical proportions, bit-identical
#:             ``pseudovisium.h5ad`` down to every spot coordinate -- except that 21,567
#:             of 86,532 nucleus centroids in the DAPI segmentation JSONs differed by
#:             **exactly one ULP**, at most 7.3e-12 px (2.0e-12 um). ``bbox``, ``contour``,
#:             ``type`` and ``type_prob`` were identical for every entry, and nothing
#:             downstream moved, because the centroid only feeds spot assignment and the
#:             spot assignment came out the same. The cause is float64 association order
#:             in the micron-to-pixel matrix multiply, which BLAS blocks differently at
#:             different thread counts; it is a property of the machine, not of the data.
#:             Comparing those files byte-for-byte would report that as a failure and bury
#:             a real regression next to it.
#: ``csv``     same index and columns; numbers within ``rtol``.
#: ``h5ad``    same obs/var names, same X within ``rtol``, same layers and obsm keys.
#: ``tensor``  same keys, shapes and dtypes; values within ``rtol``.
#: ``tiff``    same dimensions, pyramid levels, mpp and tile layout; pixels within
#:             ``atol`` on a fixed sample of tiles -- a pyramidal TIFF is not
#:             byte-reproducible across zlib builds.
#: ``figure``  same content to within text rasterisation: sides within 4 px and at most
#:             2% of pixels differing. Matplotlib output is not byte-reproducible across
#:             freetype versions -- rebuilding ``skin_s4``'s 14 figures, every one differed
#:             in 0.27-0.96% of its pixels, and in all 14 the differing pixels began at
#:             the legend column with the data panel to its left pixel-identical, the same
#:             shifted text extents that moved 8 of the 14 by 1-2 px of width. An earlier
#:             version compared only the dimensions, which passed 5 of these files while
#:             they differed in half a percent of their pixels: a check that cannot fail
#:             is not a check.
#: ``segment`` a HoVer-Net-schema segmentation, compared as an unordered **set** of cells
#:             rather than entry by entry. Necessary, not lenient: HoVerNet's detections
#:             are exactly reproducible but the order it emits them in is not, so the ids
#:             permute between builds (measured on skin_s4: the same 98,678 cells, 100%
#:             coincident to 0 px in both directions, but only 68.7% at the same id).
#:             Comparing per id reports a permutation as ~31,000 differences and buries
#:             anything real underneath them.
#: ``segment_amp`` as ``segment``, but for a segmentation whose **detections** are only
#:             reproducible to within run-to-run noise, rather than exactly. The
#:             distinction from ``segment`` is the point: HoVerNet on a slide image gives
#:             exact cells in a varying order, while CellViT-SAM runs the published AMP
#:             checkpoint and merges overlapping 1024 px patches across ray workers, so
#:             fp16-borderline cells and edge duplicates land differently from run to run.
#:             ``cellvit.json`` needs it for that reason, and Visium HD's ``hovernet.json``
#:             for a different one: it is a tiled WSI pass over a *regenerated* capture-area
#:             pyramid, and the tile merge moves 34 of P5's 187,378 nuclei by up to 0.089 px
#:             -- far inside these bounds and far outside ``segment``'s 1e-6 px. Bounds
#:             (cell count within 0.1%, 99% of cells coincident within 1 px each way, 99.5%
#:             type agreement) sit above the measured noise and far below what a wrong mpp,
#:             model or lost filter would do.
#: ``celltable`` the Visium HD ``cell_types.csv``: one row per cell with its centroid and
#:             its type at each level. Its index is this build's own cell numbering, which
#:             permutes for the same reason the segmentation ids do, so the rows are
#:             aligned by **centroid** and then the level labels must agree exactly. Row
#:             order was tried first and is useless: it called 26% of P5's rows different
#:             where every cell type in fact matches.
#: ``gt``      the Xenium ground truth, and ``tensor_gt`` the Xenium and Visium HD
#:             embeddings: all are keyed by nucleus id with a key SET that moves with the
#:             numbering -- Xenium's holds only the *callable* nuclei, where the
#:             segmentation's keys are 0..n-1 either way, and Visium HD's is bin2cell's
#:             sparse id space. Compared as sets they looked 22% apart on lung and 4.6%
#:             apart on P5 where nothing is, so both translate the ids through the
#:             segmentation's centroids first.
#: ``xenium_segment`` the Xenium annotated segmentation. Cells by position; the codes that
#:             ARE cell types must agree cell for cell, while the two codes that both mean
#:             "no ground truth" (matched a dropped type / matched nothing) may swap, which
#:             is measured and reported. Lung swaps 2,551 of them for a reason that lives in
#:             the delivered archive, and exactly 1 cell of 107,576 disagrees about its type.
#: ``metadata`` ``metadata.json``, on the paths that carry data rather than provenance.
#: ``spotdict`` spot -> cells, compared on the spot set and the per-spot cell count,
#:              which is what survives the id permutation above.
POLICIES = ("bytes", "json", "csv", "h5ad", "tensor", "tensor_gt", "tiff", "figure",
            "segment", "segment_amp", "xenium_segment", "celltable", "gt",
            "spotdict", "metadata")

#: Relative tolerance for floats under the ``json``, ``csv``, ``h5ad`` and ``tensor``
#: policies. 1e-9 is ~1000x the one-ULP drift measured above and still ~1e6 times tighter
#: than anything physically meaningful here (a nucleus is ~10 px across), so it separates
#: machine noise from a real change without being able to hide one.
FLOAT_RTOL = 1e-9


@dataclass(frozen=True)
class FileSpec:
    """One deliverable file.

    ``path`` is relative to the dataset's output folder and may contain ``{name}``,
    ``{code}``, ``{slide}`` and ``{level}``.
    """

    path: str
    policy: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"{self.path}: unknown policy {self.policy!r}")


@dataclass(frozen=True)
class DatasetSpec:
    """The full file set of one dataset folder."""

    family: str
    files: list[FileSpec] = field(default_factory=list)

    def expand(self, **fmt) -> list[FileSpec]:
        """Resolves ``{...}`` placeholders, repeating ``{level}`` specs over the levels."""
        levels = fmt.pop("levels", None)
        out: list[FileSpec] = []
        for f in self.files:
            if "{level}" in f.path:
                if levels is None:
                    raise ValueError(f"{f.path} needs levels=")
                for lv in levels:
                    out.append(FileSpec(f.path.format(level=lv, **fmt), f.policy, f.note))
            else:
                out.append(FileSpec(f.path.format(**fmt), f.policy, f.note))
        return out


# --------------------------------------------------------------------------------------
# STHELAR -- bench_data/<name>/
# --------------------------------------------------------------------------------------

STHELAR = DatasetSpec("sthelar", [
    # "segment_amp", not "segment". The clean-tree run of 2026-11-08 measured HoVerNet's
    # cell count against the delivered datasets as +8 of 98,678 on skin_s4 (0.008%), +7 of
    # 317,764 on ovary_s1 (0.002%) and -445 of 171,239 on lung_s3 (0.26%) -- so its
    # detections are very nearly, but not exactly, reproducible. The earlier claim that
    # they were exact came from a measurement taken in a work directory that still held
    # the delivered segmentation, i.e. from comparing a file with itself.
    FileSpec("hovernet.json", "segment_amp",
             "HoVerNet cells, raw PanNuke classes, ids 0..n-1; those ids belong to one build"),
    FileSpec("cellvit.json", "segment_amp",
             "CellViT cells, same schema and spot-proximity region. The one deliverable "
             "here whose detections are not exactly reproducible: the published "
             "checkpoint is AMP and overlapping patches are merged across ray workers, "
             "so fp16-borderline cells and edge duplicates land differently. Measured on "
             "skin_s4: 86,125 cells vs 86,119, 99.65% coincident at 0 px, 99.96% within "
             "1 px, 99.97% type agreement among those"),
    FileSpec("image_dict.pt", "tensor", "64 px / 20 um cell images, same ids"),
    FileSpec("hoptimus_tile_embed_bench_{name}.pt", "tensor",
             "H-Optimus-0 tile embeddings pooled per cell; added by this repo, "
             "no code in HEDeST-bench produced it"),
    FileSpec("spot_dict.json", "spotdict", "spot -> ids of the cells inside it"),
    FileSpec("pseudovisium.h5ad", "h5ad", "copy of the source dataset's AnnData"),
    FileSpec("he.tiff", "tiff", "copy of the source dataset's H&E"),
    FileSpec("sim/levels.json", "json", "per level: cell types, which level is original/label2"),
    FileSpec("sim/level{level}/proportions.csv", "csv", "per-spot cell-type fractions"),
    FileSpec("sim/level{level}/dapi_seg.json", "segment",
             "DAPI ground truth typed at this level"),
    FileSpec("plots/00_hovernet_seg.png", "figure"),
    FileSpec("plots/00_slide_overview_dapi.png", "figure"),
    FileSpec("plots/00_slide_overview_hovernet.png", "figure"),
    FileSpec("plots/01_dapi_seg_level{level}.png", "figure"),
    FileSpec("plots/02_prop_level{level}.png", "figure"),
    FileSpec("plots/03_levels_tree.png", "figure"),
])

#: Dropped on purpose, with the reason. Checked as *absent* so a stray file is caught.
STHELAR_EXCLUDED = {
    "moco_embed_bench_{name}.pt": "MoCo-v3 embeddings: not reproducible (no fixed seed) "
                                  "and superseded by H-Optimus-0",
    "_moco_model/": "MoCo training checkpoints",
    "hoptimus_embed_bench_{name}.pt": "cell-crop H-Optimus-0 variant; the tile variant "
                                      "is the one the gridsearch reads",
    "_staging/": "HoVerNet staging; lives in _work/ now",
}

# --------------------------------------------------------------------------------------
# Visium HD -- <sample>/pseudovisium/ and <sample>/pyr_tif/
# --------------------------------------------------------------------------------------

VISIUMHD = DatasetSpec("visiumhd", [
    # "segment_amp", not "segment", and the bin2cell cell id is NOT stable across builds.
    # Both corrections come from the first clean-tree run. The id first: bin2cell numbers
    # its cells from the nucleus label image, and a re-run paints the same cells with a
    # different numbering -- 1.0% of P5's ids and 0.09% of P2's survived, which is why the
    # committed cell types are re-keyed by centroid now (see
    # :mod:`hedest_simdata.visiumhd.cell_labels`). The policy second: these nuclei come
    # from a HoVerNet pass over a *regenerated* capture-area pyramid, and its tiled WSI
    # merge puts 34 of P5's 187,378 nuclei in a slightly different place -- up to 0.089 px,
    # which "segment" rejects at its 1e-6 px and "segment_amp" was written for.
    FileSpec("hovernet.json", "segment_amp",
             "nuclei of annotated cells, keyed by this build's bin2cell cell id"),
    FileSpec("spot_dict.json", "spotdict"),
    FileSpec("pseudovisium.h5ad", "h5ad", "spots x genes, summed 2 um bins"),
    # "tensor_gt", not "tensor", for the reason that policy gives: these are keyed by
    # bin2cell's cell id, drawn from a space so sparse (1..187,378 for 178,601 cells) that
    # two independent numberings of it overlap by chance. On P5 the nuclei coincide 100%
    # within 1 px and the embeddings cover all of them, yet the key sets agree on only
    # 95.4%; translating the ids through the two hovernet.json files measures the cells.
    FileSpec("hoptimus_tile_embed.pt", "tensor_gt",
             "one embedding per nucleus in hovernet.json, keyed the same way"),
    # "metadata", not "json": this file mixes data (the palette HEDeST reads, the mpp, the
    # geometry, the counts) with provenance (dates, absolute input paths, timings, the
    # device name, the check list). A deep comparison fails on all of the latter and so
    # never gets to the former; the policy names the data paths explicitly.
    FileSpec("metadata.json", "metadata",
             "mpp, transform, spot geometry, palette, counts -- and the provenance that "
             "cannot match between builds"),
    FileSpec("sim/levels.json", "json"),
    # "celltable", not "csv": its index is this build's cell numbering, which permutes, so
    # a row-by-row comparison reports a renumbering as a total disagreement -- 26% of rows
    # "differed" on P5, where in fact every cell type matches. Compared by centroid, the
    # levels agree 100.000000% on both samples.
    FileSpec("sim/cell_types.csv", "celltable",
             "cell -> type per level; the gridsearch's ground truth"),
    FileSpec("sim/proportions_{level}.csv", "csv"),
    FileSpec("plots/00_levels_tree.png", "figure"),
    FileSpec("plots/01_overview_hovernet.png", "figure"),
    FileSpec("plots/01_overview_{level}.png", "figure"),
    FileSpec("plots/02_window_hovernet.png", "figure"),
    FileSpec("plots/02_window_{level}.png", "figure"),
    FileSpec("plots/03_props_{level}.png", "figure"),
    FileSpec("plots/04_dominant_{level}.png", "figure"),
    FileSpec("plots/05_gallery_{level}.png", "figure"),
    FileSpec("plots/06_spot_qc.png", "figure"),
    FileSpec("plots/07_matching.png", "figure"),
])

#: Levels as they appear in Visium HD output file names.
VISIUMHD_LEVEL_TAGS = ["lv0", "lv1", "lv2"]

VISIUMHD_EXCLUDED = {
    "README.md": "no READMEs inside the simulated-data folders",
}

#: Written next to ``pseudovisium/``, not inside it.
VISIUMHD_PYR_TIF = FileSpec("pyr_tif/{slide}_capture.tif", "tiff",
                            "capture-area image, the frame hovernet.json lives in")

# --------------------------------------------------------------------------------------
# Xenium -- <slide>/sim[/LuCA]/
# --------------------------------------------------------------------------------------
# Exactly the files HEDeST/gridsearch/semi_sim.yaml resolves for the two xenium datasets,
# minus the MoCo embeddings. Everything else in those folders was an experiment.

XENIUM = DatasetSpec("xenium", [
    FileSpec("pannuke_fast_mask_lvl3_annotated.json", "xenium_segment",
             "HoVerNet cells carrying the Xenium cell type, plus two codes that mean 'no "
             "ground truth' -- matched a dropped type, and matched nothing"),
    FileSpec("pseudo_adata_real.h5ad", "h5ad", "pseudo-spots"),
    FileSpec("spot_dict_real.json", "spotdict"),
    FileSpec("sim_{slide}_prop_real.csv", "csv", "per-spot cell-type fractions"),
    # "gt" and "tensor_gt", not "csv" and "tensor": both are keyed by nucleus id, which is
    # a build's own numbering, and both hold only the *callable* nuclei -- so unlike the
    # segmentation, whose key set is 0..n-1 either way, their key SETS move with the
    # permutation. Compared as sets they looked 22% apart on lung where nothing is; the two
    # policies translate the ids through the segmentation's centroids first.
    FileSpec("sim_{slide}_gt.csv", "gt",
             "nucleus_id x cell type. Soft on lung (the label transfer's confidence; only "
             "60.2% of rows are one-hot and row sums run 0.35-1.0) and one-hot on breast, "
             "whose types came from a spreadsheet of hard calls. The gridsearch argmaxes "
             "either"),
    FileSpec("hoptimus_tile_embed_{code}.pt", "tensor_gt",
             "one embedding per ground-truth nucleus, keyed like the segmentation; the "
             "delivered files hold 107,429 of lung's 141,257 nuclei and 120,336 of "
             "breast's 175,204, in ground-truth order"),
])

XENIUM_EXCLUDED = {
    # two generations of naming: moco_embed_HL_64px_20um.pt and the older
    # moco_embed_moco-XENHL-rn50.pt
    "moco_embed*.pt": "MoCo-v3 embeddings",
    "image_dict_64.pt": "cell crops, only MoCo consumed them",
    "lemon_embeddings.pt": "another feature variant, not read by the gridsearch",
    "*_grouped.*": "the spot_orga != 'real' lattice variant",
    "*_small.*": "a reduced test variant",
    "*_perturb_*.csv": "proportion perturbations, built by HEDeST not by this repo",
    "spot_dict_adjust_*.json": "an older PPSA input; hedest/ppsa.py builds the "
                               "neighbourhood from the spot coordinates itself",
    "cosine_annotation.h5ad": "the label-transfer output; shipped as a CSV in "
                              "annotations/xenium/ instead",
}


# --------------------------------------------------------------------------------------


def sthelar_files(name: str, n_levels: int) -> list[FileSpec]:
    return STHELAR.expand(name=name, levels=list(range(n_levels)))


def visiumhd_files(patient: str) -> list[FileSpec]:
    return VISIUMHD.expand(levels=VISIUMHD_LEVEL_TAGS)


def xenium_files(key: str) -> list[FileSpec]:
    slide = config.XENIUM_SLIDES[key]
    return XENIUM.expand(slide=slide.name, code=slide.code)
