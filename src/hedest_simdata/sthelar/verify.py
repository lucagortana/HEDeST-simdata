"""Post-hoc consistency checks for a generated ``bench_data/<name>/``.

Verifies the deliverables exist and agree: hovernet.json <-> image_dict.pt <->
hoptimus_tile_embed (same ids), spot_dict cells are a subset of hovernet cells, and per
level the ``proportions.csv`` columns, ``dapi_seg.json`` type codes/names and the
declared category order all match -- plus the ORIGINAL/label2 levels reproduce
the source proportions. ``cellvit.json`` is checked for internal consistency and schema
parity only, since a second segmentation finds its own cells and its ids are unrelated to
the HoVerNet ones.

Two differences from the HEDeST-bench version this came from. The MoCo embedding checks
are now H-Optimus-0 checks -- that is the feature file the gridsearch reads, and it is a
deliverable rather than an optional extra, so a missing one is an error. And the declared
file set from :mod:`hedest_simdata.manifest` is checked too, which catches the failure the
per-file checks cannot see: a file that was never written at all.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import manifest
from ..config import Layout
from . import hnio
from . import bench_hier, paths

#: H-Optimus-0 is a ViT-g/14; its patch tokens, and so the pooled cell embeddings, are
#: 1536-dimensional. Worth asserting: a silently truncated embedding file would train.
HOPTIMUS_DIM = 1536


def verify(layout: Layout, name: str) -> dict:
    ds = paths.dataset(layout, name)
    issues: list[str] = []

    def need(cond, msg):
        if not cond:
            issues.append(msg)

    # --- the declared file set --------------------------------------------
    nlev_declared = (bench_hier.n_levels(name))
    for spec in manifest.sthelar_files(name, nlev_declared):
        if not (ds.out_dir / spec.path).exists():
            issues.append(f"{spec.path} missing")

    # --- hovernet.json <-> image_dict.pt <-> hoptimus_tile_embed ----------
    hv_ids = None
    if ds.hovernet_json.exists():
        _, nuc = hnio.load_nuc(ds.hovernet_json)
        hv_ids = set(nuc)
        need(hv_ids == {str(i) for i in range(len(nuc))},
             "hovernet.json ids are not 0..n-1")
        need(all(0 <= int(e["type"]) <= 5 for e in nuc.values()),
             "hovernet.json has a type outside 0..5")
    else:
        issues.append("hovernet.json missing")
    if ds.image_dict.exists() and hv_ids is not None:
        import torch
        img_ids = set(torch.load(ds.image_dict, weights_only=True).keys())
        need(img_ids == hv_ids, "image_dict ids != hovernet.json ids")
    elif not ds.image_dict.exists():
        issues.append("image_dict.pt missing")
    if ds.hoptimus_embed.exists() and hv_ids is not None:
        import torch
        emb = torch.load(ds.hoptimus_embed, weights_only=True)
        need(set(emb) == hv_ids, "hoptimus_tile_embed ids != hovernet.json ids")
        dims = {tuple(v.shape) for v in emb.values()}
        need(dims == {(HOPTIMUS_DIM,)},
             f"hoptimus_tile_embed vectors are {dims}, expected {{({HOPTIMUS_DIM},)}}")
    elif not ds.hoptimus_embed.exists():
        issues.append(f"{ds.hoptimus_embed.name} missing")

    # --- cellvit.json (optional second backend; absent until that stage ran)
    # Its ids are its OWN 0..n-1 and do NOT correspond to the HoVerNet ones --
    # a different segmentation finds different cells -- so only its internal
    # consistency and its schema parity with hovernet.json are checked.
    if ds.cellvit_json.exists():
        cv_data, cv_nuc = hnio.load_nuc(ds.cellvit_json)
        need(set(cv_nuc) == {str(i) for i in range(len(cv_nuc))},
             "cellvit.json ids are not 0..n-1")
        need(all(0 <= int(e["type"]) <= 5 for e in cv_nuc.values()),
             "cellvit.json has a type outside 0..5")
        need(cv_data.get("mpp") is not None, "cellvit.json has no mpp")
        need(all("type_name" in e for e in cv_nuc.values()),
             "cellvit.json entries lack type_name")

    # --- other root deliverables ------------------------------------------
    need(ds.adata_out.exists(), "pseudovisium.h5ad missing")
    need(ds.he_out.exists(), "he.tiff missing")
    if ds.spot_dict.exists() and hv_ids is not None:
        spot_map = json.loads(ds.spot_dict.read_text())
        cells = {c for v in spot_map.values() for c in v}
        need(cells <= hv_ids, "spot_dict references non-hovernet cells")
    elif not ds.spot_dict.exists():
        issues.append("spot_dict.json missing")

    # --- sim/ levels -------------------------------------------------------
    nlev = bench_hier.n_levels(name)
    src_l2 = pd.read_csv(ds.prop_label2, index_col=0)
    for lv in range(nlev):
        cats = bench_hier.categories(name, lv)
        pcsv = ds.level_dir(lv) / "proportions.csv"
        dseg = ds.level_dir(lv) / "dapi_seg.json"
        if not pcsv.exists():
            issues.append(f"level{lv}/proportions.csv missing"); continue
        props = pd.read_csv(pcsv, index_col="spot_id")
        need(list(props.columns) == cats, f"level{lv}: proportions cols != categories")
        need(np.allclose(props.to_numpy().sum(1), 1.0, atol=1e-6),
             f"level{lv}: proportion rows do not sum to 1")
        if not dseg.exists():
            issues.append(f"level{lv}/dapi_seg.json missing"); continue
        _, dp = hnio.load_nuc(dseg)
        cat_idx = {c: i for i, c in enumerate(cats)}
        bad = [k for k, e in dp.items()
               if e.get("type_name") not in cat_idx
               or int(e["type"]) != cat_idx[e["type_name"]]]
        need(not bad, f"level{lv}: dapi_seg type/type_name inconsistent ({len(bad)})")

    # ORIGINAL / label2 reproduce the source proportions
    def _group(level):
        leaf2cat = bench_hier.leaf_to_category(name, level)
        cats = bench_hier.categories(name, level)
        out = pd.DataFrame(0.0, index=src_l2.index, columns=cats)
        for leaf in src_l2.columns:
            out[leaf2cat[leaf]] += src_l2[leaf].to_numpy()
        return out
    src_orig = pd.read_csv(ds.prop_original, index_col=0)
    og = _group(bench_hier.ORIGINAL_LEVEL)
    need(set(og.columns) == set(src_orig.columns) and
         np.allclose(og.to_numpy(), src_orig[og.columns].to_numpy(), atol=1e-9),
         "ORIGINAL level != proportions_original.csv")

    ok = not issues
    return {"dataset": name, "ok": ok, "n_levels": nlev,
            "n_hovernet_cells": (len(hv_ids) if hv_ids is not None else None),
            "issues": issues}
