"""H-Optimus-0 tile embeddings for a bench_data dataset.

This stage is **new**. ``hoptimus_tile_embed_bench_<name>.pt`` sits in all eight reference
folders and is the feature file ``HEDeST/gridsearch/semi_sim.yaml`` reads for the STHELAR
datasets, but nothing in HEDeST-bench produced it -- it was run ad hoc against HEDeST's own
CLI. Here it is a stage like any other, so a regenerated ``bench_data`` is complete.

The work itself is HEDeST's (``hedest.features.hoptimus``), reached through
:class:`~hedest_simdata.hedest_bridge.Bridge`: whole H&E tiles at the model's native
0.5 um/px, patch tokens pooled per nucleus using the HoVer-Net contours. This module only
supplies the right slide, the right segmentation and the right mpp, and then checks what
came back.
"""
from __future__ import annotations

import time

from ..config import Layout
from ..hedest_bridge import Bridge
from . import paths

#: ViT-g/14 patch tokens, so pooled cell embeddings are 1536-dimensional.
EMBED_DIM = 1536


def extract(layout: Layout, bridge: Bridge, name: str, batch_size: int = 32,
            num_workers: int = 6) -> dict:
    """Writes ``hoptimus_tile_embed_bench_<name>.pt`` and reports what is in it.

    Runs on ``bench_data``'s own ``he.tiff`` and ``hovernet.json``, not on the source
    dataset's, so the embedding keys are the final, spot-filtered, reindexed cell ids --
    which is what makes them line up with ``spot_dict.json`` and ``image_dict.pt``. The
    ``post`` stage must therefore have run; it is declared as a prerequisite.
    """
    ds = paths.dataset(layout, name)
    for needed in (ds.he_out, ds.hovernet_json):
        if not needed.exists():
            raise FileNotFoundError(
                f"{name}: {needed} is missing; the post stage writes it, and the "
                "embedding keys must be the final reindexed cell ids.")

    t0 = time.time()
    emb = bridge.hoptimus_embeddings(
        slide=ds.he_out, segmentation=ds.hovernet_json, out=ds.hoptimus_embed,
        mpp=ds.mpp, batch_size=batch_size, num_workers=num_workers)
    dt = time.time() - t0

    dims = {tuple(v.shape) for v in emb.values()}
    if dims != {(EMBED_DIM,)}:
        raise ValueError(f"{name}: embeddings are {dims}, expected {{({EMBED_DIM},)}}")
    return {"dataset": name, "n_cells": len(emb), "dim": EMBED_DIM,
            "seconds": round(dt, 1), "cells_per_s": round(len(emb) / max(dt, 1e-6), 1),
            "out": ds.hoptimus_embed.name}
