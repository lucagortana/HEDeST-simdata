"""Independent checks of one built Xenium slide.

Written against the delivered datasets' own invariants, so it would have caught every
ambiguity the notebooks left behind: the type-code contract, the spot sets agreeing, the
column order the palette is zipped against, and the ground truth being exactly the nuclei
with a cell-type code.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config
from .. import manifest
from .paths import Slide

#: ViT-g/14 patch tokens pooled per cell.
EMBED_DIM = 1536


def run(sl: Slide) -> dict:
    issues: list[str] = []

    def need(cond: bool, msg: str) -> None:
        if not cond:
            issues.append(msg)

    # --- the declared file set --------------------------------------------------------
    for spec in manifest.xenium_files(sl.key):
        if not (sl.out_dir / spec.path).exists():
            issues.append(f"{spec.path} missing")
    if issues:
        return {"slide": sl.key, "ok": False, "issues": issues}

    n_types = len(sl.spec.cell_types)
    catchall = n_types + config.XENIUM_TYPE_CATCHALL_OFFSET
    unmatched = n_types + config.XENIUM_TYPE_UNMATCHED_OFFSET

    # --- the annotated segmentation ----------------------------------------------------
    seg = json.loads(sl.annotated_seg.read_text())
    nuc = seg["nuc"]
    need(set(nuc) == {str(i) for i in range(len(nuc))},
         "the annotated segmentation's ids are not 0..n-1")
    codes = pd.Series([int(e["type"]) for e in nuc.values()])
    bad = sorted(set(codes.unique()) - set(range(unmatched + 1)))
    need(not bad, f"type codes outside 0..{unmatched}: {bad}")

    # --- ground truth ------------------------------------------------------------------
    gt = pd.read_csv(sl.ground_truth, index_col=0)
    gt.index = gt.index.astype(str)
    need(list(gt.columns) == list(sl.spec.cell_types),
         f"ground-truth columns {list(gt.columns)} != {list(sl.spec.cell_types)}; "
         "HEDeST's runs.py zips the palette against this order and raises on a mismatch")
    expected = int((codes < n_types).sum())
    need(len(gt) == expected,
         f"ground truth has {len(gt):,} rows but {expected:,} nuclei carry a cell-type "
         f"code (0..{n_types - 1}); they are the same nuclei by definition")
    need(gt.index.isin(set(nuc)).all(),
         "the ground truth references nuclei that are not in the segmentation")
    # A row must carry *some* mass. It must not be required to carry 1: lung's ground truth
    # is the label transfer's confidence, and confidence that went to a dropped type is left
    # out rather than redistributed, so its row sums legitimately run 0.35 to 1.0. This
    # check asked for `>= 1` while reporting "all zero", which is two different claims; it
    # passed on breast, whose ground truth is one-hot, and lung's verify had never run.
    row_sums = gt.to_numpy().sum(axis=1)
    need(bool((row_sums > 0).all()),
         f"{int((row_sums <= 0).sum()):,} of {len(gt):,} ground-truth rows are all zero")
    # Where the ground truth *is* hard, every row should still sum to exactly one, so the
    # weaker check above does not quietly stop testing the one-hot case.
    from . import annotation
    if annotation.confidence(sl, gt.index) is None:
        need(bool(np.allclose(row_sums, 1.0, atol=1e-9)),
             f"this slide's ground truth is one-hot, but {int((~np.isclose(row_sums, 1.0)).sum()):,} "
             "rows do not sum to 1")

    # --- proportions, adata and spot_dict describe the SAME spots ----------------------
    props = pd.read_csv(sl.proportions, index_col=0)
    props.index = props.index.astype(str)
    need(list(props.columns) == list(sl.spec.cell_types),
         f"proportion columns {list(props.columns)} != {list(sl.spec.cell_types)}")
    need(np.allclose(props.to_numpy().sum(axis=1), 1.0, atol=1e-9),
         "proportion rows do not sum to 1")

    spot_dict = json.loads(sl.spot_dict.read_text())
    need(set(spot_dict) == set(props.index),
         f"spot_dict has {len(spot_dict):,} spots, proportions {len(props):,}; "
         "they must be the same set")
    smallest = min((len(v) for v in spot_dict.values()), default=0)
    need(smallest >= sl.spec.min_nuclei_per_spot,
         f"a kept spot holds only {smallest} nuclei, below the "
         f"{sl.spec.min_nuclei_per_spot} this dataset requires")
    # The check above is satisfied by the build agreeing with the setting, which is no
    # evidence at all when the setting is wrong: lung's min_nuclei_per_spot was the shared
    # default of 4 while its delivered spot_dict_real.json starts at 1, so 144 of its 1,957
    # spots were dropped and every check here still passed. The delivered count is the only
    # number in this function that the code cannot talk itself into.
    #
    # Within a tolerance, not exactly: HoVerNet's own run-to-run variation moves a handful
    # of nuclei, and a spot holding only those appears or disappears with them. Measured on
    # lung after the fix: 1,958 spots against the delivered 1,957, the one extra holding a
    # single nucleus, with every one of the 1,957 present and the nuclei-in-spots total
    # equal to the byte (34,716). The error this guards against is two orders of magnitude
    # larger -- 144 spots, 7.4% -- so 1% separates them with room to spare.
    if sl.spec.delivered_spots is not None:
        off = len(spot_dict) - sl.spec.delivered_spots
        need(abs(off) <= max(2, round(0.01 * sl.spec.delivered_spots)),
             f"{len(spot_dict):,} spots, but the delivered dataset has "
             f"{sl.spec.delivered_spots:,} ({off:+,}, "
             f"{abs(off) / sl.spec.delivered_spots:.1%}); "
             f"min_nuclei_per_spot={sl.spec.min_nuclei_per_spot} is the usual cause")
    in_spots = {c for v in spot_dict.values() for c in v}
    need(in_spots <= set(gt.index),
         "spot_dict references nuclei with no ground truth")

    import anndata as ad
    adata = ad.read_h5ad(sl.adata)
    need(set(adata.obs_names.astype(str)) == set(props.index),
         f"the AnnData has {adata.n_obs:,} spots, proportions {len(props):,}")
    try:
        diameter = adata.uns["spatial"][sl.spec.name]["scalefactors"][
            "spot_diameter_fullres"]
        need(diameter > 0, "spot_diameter_fullres is not positive")
    except (KeyError, TypeError):
        issues.append("the AnnData has no uns['spatial'][name]['scalefactors']"
                      "['spot_diameter_fullres']")

    # --- embeddings --------------------------------------------------------------------
    import torch
    emb = torch.load(sl.hoptimus_embed, weights_only=True)
    # Keyed by position in the annotated segmentation, but covering the ground-truth
    # nuclei only -- the others have no label, and the spot dictionary never names them.
    # This asked for every nucleus until 2026-11-08, which both delivered files fail:
    # lung's holds 107,429 of 141,257 and breast's 120,336 of 175,204, each exactly its
    # own ground-truth index.
    need(list(emb) == list(gt.index),
         f"the embeddings cover {len(emb):,} cells, the ground truth {len(gt):,}; "
         "the deliverable is the ground-truth nuclei, in ground-truth order"
         + ("" if set(emb) != set(gt.index) else " (same set, different order)"))
    need(set(emb) <= set(nuc),
         f"{len(set(emb) - set(nuc)):,} embedded cells are not in the segmentation")
    dims = {tuple(v.shape) for v in emb.values()}
    need(dims == {(EMBED_DIM,)}, f"embeddings are {dims}, expected {{({EMBED_DIM},)}}")

    return {"slide": sl.key, "ok": not issues, "nuclei": len(nuc),
            "spots": len(props), "ground_truth_rows": len(gt),
            "cell_types": list(sl.spec.cell_types),
            "type_codes": {int(k): int(v) for k, v in
                           codes.value_counts().sort_index().items()},
            "issues": issues}
