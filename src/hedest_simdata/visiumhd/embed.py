"""H-Optimus-0 tile embeddings for one Visium HD sample.

``VisiumHD_analysis`` carried its own copy of the tile-inference loop, importing three
helpers from ``HEDeST/review/feature_extraction/infer_hoptimus_tile.py``. Two things make
that the wrong shape here:

* **that path no longer exists.** The code moved to ``hedest/features/hoptimus.py``, so
  the original module raises ``FileNotFoundError`` against a current HEDeST checkout --
  the Visium HD ``embed`` stage cannot run as it stands.
* **there is one H-Optimus implementation in this repository**, reached through
  :class:`~hedest_simdata.hedest_bridge.Bridge`, and all three families use it. A second
  loop producing *almost* the same embeddings is exactly the kind of difference that is
  never noticed and never explained.

So the loop is gone and this module supplies the inputs and checks the output. What it
keeps is the verification the original added, which is worth more than the loop: a random
sample of nuclei is re-embedded **on its own** and must come back the same, which is what
shows each embedding depends only on that nucleus's own tile and contour -- not on which
other cells happened to be batched with it.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

from ..hedest_bridge import Bridge
from .config import SEED, Sample, log, read_metadata, update_metadata

OUT_NAME = "hoptimus_tile_embed.pt"

#: ViT-g/14 patch tokens pooled per nucleus.
EMBED_DIM = 1536

#: How many nuclei to re-embed on their own as the independence check, and how closely the
#: two runs must agree. Not zero: the model runs in fp16 on GPU, so a different batch
#: composition changes the order of the reductions inside it and a few ULPs are expected.
#: 1e-4 is far below the scale of the embeddings and far above that noise.
SUBSET_N = 64
SUBSET_ATOL = 1e-4


def run(s: Sample, bridge: Bridge, batch_size: int = 32, num_workers: int = 6,
        subset_check: bool = True) -> dict:
    """Writes ``hoptimus_tile_embed.pt``, keyed by cell id, and records what it contains.

    **The re-keying is the whole subtlety of this stage.** HEDeST's extractor names its
    embeddings by *position* in the segmentation file it was handed --
    ``cell_ids = [str(i) for i in range(len(nuclei))]`` -- which is right for STHELAR and
    Xenium, whose segmentations are renumbered 0..n-1 before anything is keyed to them.
    Visium HD's ``hovernet.json`` is not: it keeps bin2cell's own cell ids, sparse integers
    running past the cell count (1..187,378 for 178,601 cells), because ``spot_dict.json``,
    ``cell_types.csv`` and the delivered embedding file are all keyed by them.

    So position is not the key here, and the delivered file proves it: its 178,601 keys are
    exactly ``hovernet.json``'s, and the spot dictionary's cells are a subset of them. Taking
    the bridge's output as-is produced 0..178,600 instead, a file in which **none** of the
    spot dictionary's cells can be looked up -- well-formed, the right size, and unusable.
    The original ``VisiumHD_analysis`` avoided this by carrying its own inference loop; the
    port to the shared bridge dropped the one line that mattered.
    """
    import json

    import torch

    out = s.out_dir / OUT_NAME
    seg = s.out_dir / "hovernet.json"
    if not seg.exists():
        raise FileNotFoundError(
            f"{seg} is missing; the seg stage writes it, and the embedding keys have to be "
            "the cell ids that stage assigned.")

    mpp = _mpp(s)
    # the bridge insists on writing its own output; keep that in _work so the delivered
    # folder only ever holds the re-keyed file the manifest declares
    work = (s.work_dir or out.parent) / "hoptimus"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    by_position = bridge.hoptimus_embeddings(
        slide=s.pyr_tif, segmentation=seg, out=work / "hoptimus_by_position.pt",
        mpp=mpp, batch_size=batch_size, num_workers=num_workers)
    dt = time.time() - t0

    # position -> cell id, in the order the segmentation file lists them, which is the
    # order the extractor numbered them in
    cell_ids = list(json.loads(seg.read_text())["nuc"])
    if len(cell_ids) != len(by_position):
        raise ValueError(
            f"{len(by_position):,} embeddings for {len(cell_ids):,} nuclei in {seg.name}; "
            "they are produced one per nucleus in file order, so these must agree")
    emb = {cid: by_position[str(i)] for i, cid in enumerate(cell_ids)}
    torch.save(emb, out)

    dims = {tuple(v.shape) for v in emb.values()}
    if dims != {(EMBED_DIM,)}:
        raise ValueError(f"embeddings are {dims}, expected {{({EMBED_DIM},)}}")
    log(f"embedded {len(emb):,} nuclei in {dt:.0f}s "
        f"({len(emb) / max(dt, 1e-6):.0f} nuclei/s), keyed by cell id "
        f"({cell_ids[0]}..{cell_ids[-1]})")

    # the keys have to be usable by the rest of the dataset, which is the thing that went
    # wrong; check it here rather than trusting the remap above
    spot_dict_path = s.out_dir / "spot_dict.json"
    if spot_dict_path.exists():
        in_spots = {str(c) for v in json.loads(spot_dict_path.read_text()).values()
                    for c in v}
        missing = in_spots - set(emb)
        if missing:
            raise ValueError(
                f"{len(missing):,} of the {len(in_spots):,} cells in spot_dict.json have no "
                f"embedding (e.g. {sorted(missing)[:3]}); the keys are not the cell ids")

    info = {"model": "hf-hub:bioptimus/H-optimus-0",
            "implementation": "hedest.features.hoptimus (imported, not copied)",
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "embed_dim": EMBED_DIM, "nuclei": len(emb), "mpp": mpp,
            "seconds": round(dt, 1), "out": OUT_NAME,
            "hedest": bridge.provenance()}

    if subset_check:
        info["subset_test"] = _subset_check(s, bridge, emb, mpp, batch_size, num_workers)
    update_metadata(s.out_dir, "embed", info)
    return info


def _mpp(s: Sample) -> float:
    """The resolution the seg stage recorded, which is spaceranger's."""
    meta = read_metadata(s.out_dir)
    try:
        return float(meta["seg"]["mpp"]["used"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "metadata.json has no seg.mpp.used; run the seg stage first rather than "
            "letting the resolution be guessed from the image.") from exc


def _subset_check(s: Sample, bridge: Bridge, full: dict, mpp: float,
                  batch_size: int, num_workers: int) -> dict:
    """Re-embed a random sample of nuclei alone; they must come back the same.

    Done by writing a segmentation file holding only those nuclei, so the second run sees
    a different tile grid and a different batching. Agreement therefore means the
    per-nucleus result is independent of both.

    ``full`` arrives keyed by **cell id**, as the deliverable is, so it indexes
    ``hovernet.json`` directly. It was keyed by position before the re-key in :func:`run`,
    which is how this function came to ask ``nuc["89524"]`` for a nucleus whose id that
    never was.
    """
    import torch

    from . import hovernet

    keys = sorted(full)
    if not keys:
        return {"skipped": "no nuclei"}
    rng = np.random.default_rng(SEED)
    pick = [keys[i] for i in rng.choice(len(keys), size=min(SUBSET_N, len(keys)),
                                        replace=False)]

    # `hovernet.load` returns the whole file; this module's `hovernet` has no two-value
    # `load_nuc`. The name is not invented -- `sthelar/hnio.py` has exactly that function
    # and three STHELAR modules call it -- which is why reaching for it here read as
    # correct. Because this is the stage's last step the mistake cost a full GPU pass to
    # find: 16 min for P5, with all 178,601 embeddings already written. Second bug of this
    # shape, after a spawned `plot_hovernet_types.py` that did not exist, so
    # `tests/test_env_spec.py` now checks these calls statically for nothing.
    data = hovernet.load(s.out_dir / "hovernet.json")
    nuc = data["nuc"]
    subset = {k: nuc[k] for k in pick}
    # in _work, not in the delivered folder: an earlier crash in exactly this function
    # would have left these behind next to the deliverables
    scratch = (s.work_dir or s.out_dir) / "hoptimus"
    scratch.mkdir(parents=True, exist_ok=True)
    tmp_seg = scratch / "_subset_check.json"
    tmp_out = scratch / "_subset_check.pt"
    try:
        hovernet.write(tmp_seg, subset, data.get("mag"), mpp)
        again = bridge.hoptimus_embeddings(slide=s.pyr_tif, segmentation=tmp_seg,
                                          out=tmp_out, mpp=mpp, batch_size=batch_size,
                                          num_workers=num_workers)
        # the standalone run numbers its cells by position in the file it was given, so
        # compare by position in `pick` rather than by key
        order = list(again.values())
        diffs = [float(torch.abs(torch.as_tensor(full[k], dtype=torch.float32)
                                 - torch.as_tensor(order[i], dtype=torch.float32)).max())
                 for i, k in enumerate(pick)]
    finally:
        tmp_seg.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)

    worst = max(diffs) if diffs else 0.0
    ok = worst <= SUBSET_ATOL
    log(f"subset check: {len(pick)} nuclei re-embedded alone, largest difference "
        f"{worst:.2e} ({'ok' if ok else 'TOO LARGE'})")
    if not ok:
        raise ValueError(
            f"re-embedding {len(pick)} nuclei on their own changed them by up to {worst:.2e} "
            f"(limit {SUBSET_ATOL}). A nucleus's embedding must depend only on its own tile "
            "and contour, so this means the tiling or the pooling is not per-cell.")
    return {"n": len(pick), "max_abs_difference": worst, "atol": SUBSET_ATOL}
