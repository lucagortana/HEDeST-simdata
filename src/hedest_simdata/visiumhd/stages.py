"""The Visium HD stage bodies.

Ported from ``VisiumHD_analysis/pseudovisium/__main__.py``. Three things changed:

* **no argparse.** Stage selection, ordering and dependencies are declared in
  :mod:`hedest_simdata.visiumhd` and resolved by ``make_simdata.py``.
* **no ``palette`` stage.** It existed to re-record the colours in an already-built
  ``metadata.json`` without redoing ``seg`` -- a maintenance operation on an existing
  dataset, which belongs to the analysis repository, not to one that builds datasets from
  scratch.
* **``sys.exit`` became :class:`StageError`.** A library that calls ``sys.exit`` cannot be
  driven: the driver has to be able to report one sample's failure and carry on with the
  others.

The order in which a sample is built::

    seg    the committed cell types matched to the segmented nuclei, proved, then
           hovernet.json + sim/cell_types.csv + metadata.json["seg"]
    spots  grid, counts from the 2 um bins, proportions, spot_dict, pseudovisium.h5ad
    embed  H-Optimus-0 tile embeddings (GPU)
    plots  the verification figures
    verify independent checks of everything written
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import colors, hovernet, slide

from ..config import Layout
from . import config, inputs, labels, match, paths
from .config import log


class StageError(RuntimeError):
    """A check failed, so nothing was written."""

REPO = Path(__file__).resolve().parents[3]

def _prepare_out(s: config.Sample, overwrite: bool) -> None:
    """Empty a previous output, but only one this pipeline wrote."""
    if not s.out_dir.exists():
        return
    stray = [p.name for p in s.out_dir.iterdir() if p.name not in WRITTEN]
    if stray:
        raise StageError(f"refusing to touch {s.out_dir}: it holds files this pipeline did not write ({stray})")
    for name in WRITTEN:
        p = s.out_dir / name
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)


def _report(checks: list[dict]) -> None:
    for c in checks:
        mark = "PASS" if c["passed"] else ("FAIL" if c["hard"] else "WARN")
        extra = ""
        if "control" in c:
            extra = f" | control {c['control']}"
        val = c.get("value")
        val = f"{val:.5f}" if isinstance(val, float) else val
        log(f"  {mark} {c['id']:4s} {c['name']}: {val}{extra}")


def stage_seg(s: config.Sample, markers: bool = True) -> None:
    """Match the committed cell types to the segmented nuclei, prove it, and write.

    Nothing is written until every provenance and matching check has passed, so a
    failure leaves no half-built dataset behind.
    """
    for k, p in s.required().items():
        log(f"  input {k}: {p}")
    mpp = inputs.read_mpp(s)
    log(f"mpp {mpp}")
    cells = inputs.load_cells(s)
    log(f"annotated cells: {len(cells):,}")
    mag, nuc = inputs.load_nuclei(s)
    table, source = inputs.load_nucleus_table(s, nuc)
    W, H = slide.size(s.pyr_tif)
    mask = inputs.load_label_mask(s)
    bins = inputs.load_bins(s, cells.crop_key)
    log(f"nuclei {len(nuc):,} | label table {len(table):,} ({source}) | image {W}x{H} | bins {len(bins.array_row):,}")

    prov, offset = inputs.provenance(s, cells, nuc, bins, mask.shape, (W, H))
    _report(prov)
    if match.failures(prov):
        raise StageError("provenance checks failed; nothing written")

    h = labels.build(cells.labels, s.levels, s.unassigned_pattern, s.unassigned_types)
    un = labels.unassigned_mask(cells.labels, h)
    log(f"levels: {[len(l) for l in h.levels]} | unassigned types {sorted(h.unassigned)} | "
        f"unassigned cells {un.sum():,}")

    totals = inputs.cell_totals(s)
    marker_expr = None
    if markers:
        types = set(cells.labels[s.levels[-1]].astype(str))
        genes = sorted({g for t in types if t in match.COLON_MARKERS for g in match.COLON_MARKERS[t]})
        marker_expr = (inputs.marker_expression(s, genes, cells.object_id)
                       if genes else None)
    checks, cell_keys, scores = match.run(cells, nuc, table, source, mask, bins, totals, (W, H),
                                          marker_expr, s.levels[-1])
    _report(checks)
    if match.failures(checks):
        raise StageError("matching checks failed; nothing written")
    del mask, bins

    kept_h = labels.restrict(h, cells.labels[~un])
    pal = colors.palette(kept_h.levels, kept_h.parent, kept_h.unassigned)
    leaf_diff, pair = colors.min_difference({k: v for k, v in pal.items() if k[0] == kept_h.n_levels - 1})
    palette_hex = {f"lv{k}::{n}": colors.to_hex(rgb) for (k, n), rgb in pal.items()}

    kept = ~un
    hier = labels.restrict(h, cells.labels[kept])
    from .spots import hierarchy_to_json
    _prepare_out(s, overwrite=True)
    (s.out_dir / "sim").mkdir(parents=True, exist_ok=True)
    meta = {
        "sample": s.name, "date": datetime.now().isoformat(timespec="seconds"),
        "code": {"repo": str(REPO), "commit": config.git_commit(REPO)},
        "inputs": {k: config.file_info(p) for k, p in {**s.required(), "nucleus_table": s.nucleus_table,
                                                         "full_image": s.full_image}.items()},
        "mpp": mpp, "crop_key": cells.crop_key, "crop_offset_xy": offset.tolist(),
        "levels": {f"lv{k}": c for k, c in enumerate(s.levels)},
        "unassigned": {"pattern": s.unassigned_pattern, "explicit_types": list(s.unassigned_types),
                       "cells_removed": int(un.sum()),
                       "types_removed": {f"lv{k}": [n for n in h.levels[k] if (k, n) in h.unassigned]
                                         for k in range(h.n_levels)},
                       "rule": "a cell unassigned at any level is removed from every level"},
        "nuclei_in_hovernet": len(nuc), "cells_annotated": len(cells), "nuclei_written": int(kept.sum()),
        "id_rule": "cell id = annotated obs name = bin2cell object_id = HoVerNet label; "
                   "HoVerNet key from the label table (label = key + 1)",
        "hierarchy": hierarchy_to_json(hier),
        "palette": palette_hex,
        "palette_min_leaf_difference": {"oklab_x100": leaf_diff, "pair": [list(p) for p in pair]},
        "checks": prov + checks,
        "marker_scores": scores,
    }
    new_to_old = dict(zip(cells.ids[kept], cell_keys[kept]))
    filtered = hovernet.select(nuc, new_to_old)
    hovernet.write(s.out_dir / "hovernet.json", filtered, mag, mpp["used"])
    xy = hovernet.centroids(filtered, list(filtered))
    pd.DataFrame({
        "cell_id": list(filtered),
        "hovernet_key": [e["hovernet_key"] for e in filtered.values()],
        "x": xy[:, 0], "y": xy[:, 1],
        "hovernet_type": [e["type"] for e in filtered.values()],
        "hovernet_type_prob": [e["type_prob"] for e in filtered.values()],
        **{f"lv{k}": cells.labels[c].astype(str).to_numpy()[kept] for k, c in enumerate(s.levels)},
    }).to_csv(s.out_dir / "sim" / "cell_types.csv", index=False)
    config.update_metadata(s.out_dir, "seg", meta)
    log(f"{int(kept.sum()):,} nuclei written; {int(un.sum()):,} unassigned cells removed")
