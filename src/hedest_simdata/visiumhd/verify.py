"""Re-read the written folders and check them, independently of how they were made.

V1  one spot set, same order: pseudovisium.h5ad, every proportions CSV, spot_dict
V2  one cell set, same order: hovernet.json, embeddings, cell_types.csv;
    spot_dict holds each of those cells at most once; cell_types.spot_id agrees
V3  proportions recomputed from spot_dict + cell_types equal the CSVs; rows sum
    to 1; every coarse column equals the sum of its children
V4  n_cells = spot_dict sizes; X row sums = total_counts; random spots re-summed
    from the bins equal X
V5  embeddings finite, 1536-d float32
V6  no unassigned label or column anywhere
V7  nucleus centroids and spot centres inside the image
V8  every written nucleus is its source HoVerNet nucleus (same centroid and
    contour) and its key is the one the label table gives for that cell id
V9  every written label equals the annotated file's label for that cell id
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import hovernet, slide

from . import inputs
from .config import Sample, log, read_metadata, update_metadata
from .embed import OUT_NAME
from .spots import hierarchy_from_json, read_cell_table

N_RESUM = 50


class VerifyError(RuntimeError):
    pass


def _v(results, vid, name, ok, **info):
    results.append({"id": vid, "name": name, "passed": bool(ok), **info})
    log(f"  {'PASS' if ok else 'FAIL'} {vid} {name} {info if info else ''}")


def run(s: Sample) -> dict:
    import re
    import torch

    W, H = slide.size(s.pyr_tif)
    src_nuc = inputs.load_nuclei(s)[1]
    table, _ = inputs.load_nucleus_table(s, src_nuc)
    cells_ann = inputs.load_cells(s)
    ann = cells_ann.labels.astype(str)
    ann.index = cells_ann.ids
    pat = re.compile(s.unassigned_pattern, re.I)
    report: dict[str, list] = {}

    vdir = s.out_dir
    res: list[dict] = []
    meta = read_metadata(vdir)
    h = hierarchy_from_json(meta["seg"]["hierarchy"])
    mpp = meta["seg"]["mpp"]["used"]
    pv = ad.read_h5ad(vdir / "pseudovisium.h5ad")
    props = {k: pd.read_csv(vdir / "sim" / f"proportions_lv{k}.csv", index_col="spot_id")
             for k in range(h.n_levels)}
    sd = json.loads((vdir / "spot_dict.json").read_text())
    nuc = hovernet.load(vdir / "hovernet.json")["nuc"]
    emb = torch.load(vdir / OUT_NAME, map_location="cpu", weights_only=True)
    cells = read_cell_table(vdir)
    spots = list(pv.obs_names)

    # V1
    same = all(list(p.index) == spots for p in props.values()) and list(sd) == spots
    _v(res, "V1", "one spot set across h5ad, proportions and spot_dict", same, spots=len(spots))

    # V2
    ids = list(nuc)
    in_sd = [c for v in sd.values() for c in v]
    ok = (ids == list(emb) == cells["cell_id"].tolist()
          and len(in_sd) == len(set(in_sd)) and set(in_sd) <= set(ids))
    sd_of = {c: sid for sid, v in sd.items() for c in v}
    ok &= bool((cells["spot_id"] == cells["cell_id"].map(sd_of).fillna("")).all())
    _v(res, "V2", "one cell set across hovernet.json, embeddings and cell_types",
       ok, cells=len(ids), cells_in_spots=len(in_sd))

    # V3
    lab = cells.set_index("cell_id")
    worst, sums, nest = 0.0, 0.0, 0.0
    for k in range(h.n_levels):
        cats = h.levels[k]
        rec = pd.DataFrame(0.0, index=spots, columns=cats)
        for sid, members in sd.items():
            vc = lab.loc[members, f"lv{k}"].value_counts()
            rec.loc[sid, vc.index] = vc.to_numpy() / len(members)
        worst = max(worst, float(np.abs(rec.to_numpy() - props[k][cats].to_numpy()).max()))
        sums = max(sums, float(np.abs(props[k].sum(axis=1) - 1).max()))
        if k:
            for parent in h.levels[k - 1]:
                kids = [c for c in cats if h.parent[k][c] == parent]
                nest = max(nest, float(np.abs(props[k][kids].sum(axis=1) - props[k - 1][parent]).max()))
    _v(res, "V3", "proportions = recomputed; rows sum to 1; coarse = sum of children",
       worst < 1e-9 and sums < 1e-9 and nest < 1e-9, max_diff=worst, max_row_sum_err=sums,
       max_nesting_err=nest)

    # V4
    ncell = np.array([len(sd[sid]) for sid in spots])
    X = pv.X.tocsr()
    rs = np.asarray(X.sum(axis=1)).ravel()
    ok = (ncell == pv.obs["n_cells"].to_numpy()).all() and np.allclose(rs, pv.obs["total_counts"], rtol=1e-5)
    bins = inputs.load_bins(s, meta["seg"]["crop_key"])
    Xb, _ = inputs.load_bin_matrix(s)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(spots), size=min(N_RESUM, len(spots)), replace=False)
    tree = cKDTree(bins.xy_crop * mpp)
    centres = pv.obs[["x_um", "y_um"]].to_numpy()
    worst4 = 0.0
    for i in pick:
        members = tree.query_ball_point(centres[i], s.spot_diameter_um / 2)
        ref = np.asarray(Xb[members].sum(axis=0)).ravel()
        got = X[i].toarray().ravel()
        worst4 = max(worst4, float(np.abs(ref - got).max() / max(ref.max(), 1)))
    del Xb, bins
    _v(res, "V4", "n_cells, total_counts and re-summed bins match", ok and worst4 < 1e-5,
       respot_checked=len(pick), max_rel_diff=worst4)

    # V5
    vec = torch.stack([emb[c] for c in ids]).numpy()
    _v(res, "V5", "embeddings finite, 1536-d, float32",
       vec.dtype == np.float32 and vec.shape[1] == 1536 and np.isfinite(vec).all(), shape=list(vec.shape))

    # V6
    removed = meta["seg"]["unassigned"]["types_removed"]      # by pattern and by name
    bad_lab = sum(int(cells[f"lv{k}"].str.contains(pat).sum()) for k in range(h.n_levels))
    bad_col = sum(any(pat.search(c) for c in p.columns) for p in props.values())
    for k in range(h.n_levels):
        names = set(removed.get(f"lv{k}", []))
        bad_lab += int(cells[f"lv{k}"].isin(names).sum())
        bad_col += int(sum(c in names for c in props[k].columns))
    _v(res, "V6", "no unassigned label or column (pattern and named types)",
       bad_lab == 0 and bad_col == 0, labels=int(bad_lab), columns=int(bad_col),
       removed_types={k: v for k, v in removed.items() if v})

    # V7
    cxy = cells[["x", "y"]].to_numpy(dtype=float)
    inside = lambda p: ((p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)).all()
    _v(res, "V7", "nuclei and spot centres inside the image",
       inside(cxy) and inside(pv.obsm["spatial"]))

    # V8
    keys = cells["hovernet_key"].to_numpy()
    want = table.loc[cells["cell_id"].astype(np.int64), "key"].to_numpy()
    same_key = bool((keys == want).all())
    same_geo = all(np.allclose(nuc[c]["centroid"], src_nuc[k]["centroid"])
                   and nuc[c]["contour"] == src_nuc[k]["contour"] for c, k in zip(ids, keys))
    _v(res, "V8", "written nuclei = source nuclei for the tabled key", same_key and same_geo)

    # V9
    written = cells.set_index("cell_id")[[f"lv{k}" for k in range(h.n_levels)]]
    same_lab = bool((written.to_numpy() == ann.loc[written.index].to_numpy()).all())
    _v(res, "V9", "written labels = annotated labels for that cell id", same_lab)

    report["checks"] = res
    failed = [r for r in res if not r["passed"]]
    stamp = datetime.now().isoformat(timespec="seconds")
    update_metadata(s.out_dir, "verify", {"date": stamp, "passed": not failed, "checks": res})
    if failed:
        raise VerifyError(f"{len(failed)} verification checks failed: {[r['id'] for r in failed]}")
    return report
