"""Compares a built data folder against a reference one, honouring each file's policy.

    python tests/compare.py --built /path/to/new --reference /path/to/old
    python tests/compare.py --built NEW --reference OLD --families sthelar --datasets skin_s4

This is the acceptance test: "the regenerated data matches what we already have" as a
check rather than an opinion. Almost nothing here is byte-stable -- a pyramidal TIFF depends
on the zlib build, a matplotlib PNG on the freetype version, a ``.pt`` on the order tensors
were stacked -- so comparing bytes everywhere would report noise as failure and bury a real
regression next to it. Each file is compared the way its content actually has to agree; see
:mod:`hedest_simdata.manifest` for the policies and for the one-ULP measurement that set
the float tolerance.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hedest_simdata import config      # noqa: E402
from hedest_simdata import manifest    # noqa: E402

RTOL = manifest.FLOAT_RTOL


class Mismatch(Exception):
    pass


def _set_report(mine, theirs, unit: str, agree=None) -> str:
    """Why two keyed collections differ, as a measurement rather than a verdict.

    "index differs (107433 vs 107429 rows)" is true and tells you nothing: it reads the
    same whether four rows were added or every row was renamed. The difference between
    those two is the whole question when a build is being accepted, so say which it is --
    how much of the reference is present, how much is extra, and whether the part they
    share actually agrees.
    """
    sa, sb = set(map(str, mine)), set(map(str, theirs))
    shared = sa & sb
    detail = ""
    if agree is not None and shared:
        try:
            keep = [k for k in map(str, theirs) if k in shared]
            detail = "; " + agree(keep)
        except Exception as exc:                       # a report must not raise
            detail = f"; (could not compare the shared part: {exc})"
    return (f"{len(sa):,} {unit} vs {len(sb):,}: {len(shared):,} of the reference's "
            f"{len(sb):,} present ({len(shared) / max(len(sb), 1):.4%}), "
            f"{len(sa - sb):,} extra, {len(sb - sa):,} missing{detail}")


# --------------------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------------------


def _floats_close(a, b, where: str) -> None:
    """Equal, with floats to :data:`RTOL` and missing values treated as equal to each other.

    The NaN handling is not a convenience. ``sim/cell_types.csv`` has a ``spot_id`` column
    that is empty for every cell outside a spot -- 129,469 of 178,601 rows on Visium HD
    P5 -- and those come back as NaN in an object column, where ``!=`` is true of every
    pair. A first version of this function reported all 129,469 as differences on a file
    that was in fact identical, which is exactly the kind of false alarm that makes a
    comparison tool worth less than no tool at all.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise Mismatch(f"{where}: shape {a.shape} vs {b.shape}")
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        if not np.allclose(a, b, rtol=RTOL, atol=0.0, equal_nan=True):
            d = np.nanmax(np.abs(a.astype(float) - b.astype(float)))
            raise Mismatch(f"{where}: largest difference {d:.3g} (rtol {RTOL:g})")
        return
    import pandas as pd
    sa, sb = pd.Series(a.ravel()), pd.Series(b.ravel())
    both_missing = sa.isna() & sb.isna()
    differ = (sa != sb) & ~both_missing
    if differ.any():
        n = int(differ.sum())
        example = [(x, y) for x, y in zip(sa[differ][:2], sb[differ][:2])]
        raise Mismatch(f"{where}: {n} of {a.size} values differ, e.g. {example}")


def cmp_bytes(new: Path, ref: Path) -> str:
    hn = hashlib.sha256(new.read_bytes()).hexdigest()
    hr = hashlib.sha256(ref.read_bytes()).hexdigest()
    if hn != hr:
        raise Mismatch(f"bytes differ ({new.stat().st_size:,} vs {ref.stat().st_size:,})")
    return "byte-identical"


def _json_compare(a, b, where: str = "") -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            only_a, only_b = sorted(set(a) - set(b))[:4], sorted(set(b) - set(a))[:4]
            raise Mismatch(f"{where or 'root'}: keys differ "
                           f"(+{len(set(a) - set(b))} {only_a}, "
                           f"-{len(set(b) - set(a))} {only_b})")
        for k in a:
            _json_compare(a[k], b[k], f"{where}.{k}" if where else str(k))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise Mismatch(f"{where}: length {len(a)} vs {len(b)}")
        if a and all(isinstance(x, (int, float)) for x in a + b):
            _floats_close(a, b, where)
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _json_compare(x, y, f"{where}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        _floats_close([a], [b], where)
    elif a != b:
        raise Mismatch(f"{where}: {a!r} vs {b!r}")


def cmp_json(new: Path, ref: Path) -> str:
    a, b = json.loads(new.read_text()), json.loads(ref.read_text())
    _json_compare(a, b)
    n = len(a.get("nuc", a)) if isinstance(a, dict) else len(a)
    return f"equal as data ({n:,} entries, floats to rtol {RTOL:g})"


def cmp_csv(new: Path, ref: Path) -> str:
    import pandas as pd
    a = pd.read_csv(new, index_col=0)
    b = pd.read_csv(ref, index_col=0)
    if list(a.columns) != list(b.columns):
        raise Mismatch(f"columns {list(a.columns)} vs {list(b.columns)}")
    if not a.index.equals(b.index):
        raise Mismatch(_set_report(a.index, b.index, "rows", lambda idx: _csv_agree(a, b, idx)))
    for col in a.columns:
        _floats_close(a[col].to_numpy(), b[col].to_numpy(), col)
    return f"equal ({a.shape[0]:,} x {a.shape[1]})"


def _csv_agree(a, b, shared) -> str:
    """How much of the shared part agrees, for a report about an index difference.

    Both frames are re-indexed as strings first. ``read_csv`` infers the index dtype, and
    two builds of the same file can land on int64 and object -- which made the first
    version of this report raise "None of [...] are in the [index]" and print that instead
    of the number it exists to print.
    """
    import numpy as np
    a = a.set_axis(a.index.astype(str))
    b = b.set_axis(b.index.astype(str))
    sa, sb = a.loc[shared], b.loc[shared]
    same = np.ones(len(shared), bool)
    for col in a.columns:
        va, vb = sa[col].to_numpy(), sb[col].to_numpy()
        if va.dtype.kind in "fc" or vb.dtype.kind in "fc":
            same &= np.isclose(va.astype(float), vb.astype(float), rtol=RTOL,
                               equal_nan=True)
        else:
            same &= (va.astype(str) == vb.astype(str))
    return f"{float(same.mean()):.4%} of the shared rows identical"


def cmp_h5ad(new: Path, ref: Path) -> str:
    import anndata as ad
    import scipy.sparse as sp
    a, b = ad.read_h5ad(new), ad.read_h5ad(ref)
    if not a.obs_names.equals(b.obs_names):
        raise Mismatch(_set_report(a.obs_names, b.obs_names, "obs"))
    if a.shape != b.shape:
        raise Mismatch(f"shape {a.shape} vs {b.shape}")
    if not a.var_names.equals(b.var_names):
        raise Mismatch("var_names differ")
    Xa = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    Xb = b.X.toarray() if sp.issparse(b.X) else np.asarray(b.X)
    _floats_close(Xa, Xb, "X")
    if set(a.layers) != set(b.layers):
        raise Mismatch(f"layers {sorted(a.layers)} vs {sorted(b.layers)}")
    for k in a.layers:
        La = a.layers[k].toarray() if sp.issparse(a.layers[k]) else np.asarray(a.layers[k])
        Lb = b.layers[k].toarray() if sp.issparse(b.layers[k]) else np.asarray(b.layers[k])
        _floats_close(La, Lb, f"layers[{k}]")
    if set(a.obsm) != set(b.obsm):
        raise Mismatch(f"obsm {sorted(a.obsm)} vs {sorted(b.obsm)}")
    for k in a.obsm:
        _floats_close(np.asarray(a.obsm[k]), np.asarray(b.obsm[k]), f"obsm[{k}]")
    shared = [c for c in a.obs.columns if c in b.obs.columns]
    for c in shared:
        if a.obs[c].dtype.kind in "ifb":
            _floats_close(a.obs[c].to_numpy(), b.obs[c].to_numpy(), f"obs[{c}]")
    return (f"equal ({a.n_obs:,} x {a.n_vars:,}, {len(a.layers)} layer(s), "
            f"{len(a.obsm)} obsm)")


def cmp_tensor(new: Path, ref: Path) -> str:
    import torch
    # weights_only=True: these files are {cell id: tensor} mappings, so the safe loader
    # handles them, and the permissive default now warns on every call.
    a = torch.load(new, map_location="cpu", weights_only=True)
    b = torch.load(ref, map_location="cpu", weights_only=True)
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise Mismatch("not a {key: tensor} mapping")
    if set(a) != set(b):
        shapes = {tuple(torch.as_tensor(v).shape) for v in a.values()}
        raise Mismatch(_set_report(a, b, "keys") + f"; all shapes {shapes}")
    worst, worst_key = 0.0, None
    for k in a:
        va, vb = torch.as_tensor(a[k]), torch.as_tensor(b[k])
        if va.shape != vb.shape:
            raise Mismatch(f"{k}: shape {tuple(va.shape)} vs {tuple(vb.shape)}")
        # The cheap test first, and it is the usual answer. `.float()` on both sides
        # allocates a fresh 4-byte copy of every element before subtracting, which for
        # `image_dict.pt` means turning 100k+ uint8 crops into float32 one at a time --
        # minutes of the comparison spent proving that identical tensors are identical.
        if va.dtype == vb.dtype and torch.equal(va, vb):
            continue
        d = float((va.float() - vb.float()).abs().max())
        if d > worst:
            worst, worst_key = d, k
    # Not compared to RTOL, for two reasons and the second one dominates. These come off a
    # GPU in fp16, so two runs differ by more than float noise and two GPU models by more
    # again -- but far bigger than that, these files are keyed by HoVerNet cell id, and
    # those ids permute between builds (see cmp_segment). So the vector under key "7" is a
    # different cell's vector, and the difference reported below is a measure of the
    # permutation, not of numerical drift. What has to agree is the keys and the shapes,
    # which is what a downstream run depends on; that the right vector sits under the right
    # id is checked within a build by the pipeline's own verify stage.
    return (f"same {len(a):,} keys and shapes; largest value difference {worst:.3g} "
            f"(at {worst_key})")


def cmp_tiff(new: Path, ref: Path) -> str:
    import tifffile as tf
    with tf.TiffFile(new) as fa, tf.TiffFile(ref) as fb:
        sa = [p.shape for p in fa.pages]
        sb = [p.shape for p in fb.pages]
        if sa != sb:
            raise Mismatch(f"pyramid {sa} vs {sb}")
        # a fixed sample of tiles, rather than the whole multi-GB level 0
        page_a, page_b = fa.pages[min(2, len(fa.pages) - 1)], fb.pages[min(2, len(fb.pages) - 1)]
        ia, ib = page_a.asarray(), page_b.asarray()
    if ia.shape != ib.shape:
        raise Mismatch(f"sampled level shape {ia.shape} vs {ib.shape}")
    diff = np.abs(ia.astype(np.int32) - ib.astype(np.int32))
    if diff.max() > 1:
        raise Mismatch(f"sampled level differs by up to {int(diff.max())} grey levels")
    return f"{len(sa)} levels {sa[0]} -> {sa[-1]}, sampled level within {int(diff.max())}"


#: A figure may differ by at most this fraction of its pixels, and its sides by at most
#: :data:`FIGURE_SIZE_SLACK` px. Both come from a measurement rather than from loosening
#: the test until it passed. Rebuilding ``skin_s4``'s 14 figures against the delivered
#: ones: every single one differs, by 0.27% to 0.96% of pixels, mean absolute channel
#: difference 0.34 to 1.31 out of 255 -- and in all 14 the differing pixels begin at the
#: legend column (x >= 506, or x >= 166 for the levels tree), with the data panel to its
#: left pixel-identical. That is text rasterisation -- a freetype or font version -- which
#: also moves a ``bbox_inches="tight"`` figure's width by a pixel or two, which is exactly
#: the 1-2 px seen on 8 of the 14.
FIGURE_MAX_DIFF_FRACTION = 0.02
FIGURE_SIZE_SLACK = 4
#: Per-channel difference below which two pixels count as the same, so that antialiasing
#: along a glyph edge is not read as a changed figure.
FIGURE_CHANNEL_TOL = 8


def cmp_figure(new: Path, ref: Path) -> str:
    """Compare two figures by content, tolerating only text rasterisation.

    The first version of this compared image *sizes* and nothing else, which is how it
    came to report five of these files as equal while they differed in half a percent of
    their pixels. It now measures the difference and reports what it measured.
    """
    import numpy as np
    from PIL import Image

    with Image.open(new) as a_img, Image.open(ref) as b_img:
        a = np.asarray(a_img.convert("RGB")).astype(np.int16)
        b = np.asarray(b_img.convert("RGB")).astype(np.int16)
    if abs(a.shape[1] - b.shape[1]) > FIGURE_SIZE_SLACK or \
            abs(a.shape[0] - b.shape[0]) > FIGURE_SIZE_SLACK:
        raise Mismatch(f"size {a.shape[1]}x{a.shape[0]} vs {b.shape[1]}x{b.shape[0]}, "
                       f"further apart than the {FIGURE_SIZE_SLACK} px a shifted text "
                       "extent explains")
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    diff = np.abs(a[:h, :w] - b[:h, :w]).max(axis=2) > FIGURE_CHANNEL_TOL
    frac = float(diff.mean())
    cols = np.where(diff.any(axis=0))[0]
    where = f", x {cols.min()}-{cols.max()} of {w}" if cols.size else ", identical"
    if frac > FIGURE_MAX_DIFF_FRACTION:
        raise Mismatch(f"{100 * frac:.2f}% of pixels differ, past the "
                       f"{100 * FIGURE_MAX_DIFF_FRACTION:g}% that a font difference "
                       f"explains{where}")
    return (f"{a.shape[1]}x{a.shape[0]} (ref {b.shape[1]}x{b.shape[0]}), "
            f"{100 * frac:.2f}% of pixels differ{where}")


def cmp_segment(new: Path, ref: Path) -> str:
    """A segmentation that reproduces exactly, compared as an unordered set of cells.

    Exactly, but in a different **order**: the 0..n-1 reindexing that follows a
    multi-worker pass depends on which tile's cells land first, so the ids permute between
    builds. A per-id comparison called ``skin_s4``'s segmentation ~31,000 differences,
    which is true and useless; by position it is the same cells at 0 px with the same
    types, and the id agreement (68.7%) is reported rather than required.

    **Only the DAPI ground truth uses this now**, and it earns it: measured across three
    slides and five levels, every one is the same cell set at 0 px with the same types and
    100% of ids preserved. ``hovernet.json`` used to use it too, on the belief that
    HoVerNet's detections were exact -- a belief that came from comparing a rebuild against
    a work directory that still held the delivered segmentation. On a clean tree HoVerNet
    moves a few cells per ten thousand, so it is a ``segment_amp`` file; see the note in
    ``manifest.py`` for the three measurements.
    """
    import numpy as np

    a = json.loads(new.read_text())
    b = json.loads(ref.read_text())
    na, nb = a.get("nuc", {}), b.get("nuc", {})
    if len(na) != len(nb):
        raise Mismatch(f"{len(na):,} cells vs {len(nb):,}")
    for key in ("mag", "mpp"):
        if key in a or key in b:
            va, vb = a.get(key), b.get(key)
            if va is not None and vb is not None:
                _floats_close([float(va)], [float(vb)], key)

    ka, kb = sorted(na, key=int), sorted(nb, key=int)
    ca = np.array([na[k]["centroid"] for k in ka], dtype=float)
    cb = np.array([nb[k]["centroid"] for k in kb], dtype=float)
    from scipy.spatial import cKDTree
    d_fwd, idx = cKDTree(cb).query(ca)
    d_rev, _ = cKDTree(ca).query(cb)
    if d_fwd.max() > 1e-6 or d_rev.max() > 1e-6:
        raise Mismatch(
            f"the cell sets differ: {int((d_fwd > 1e-6).sum()):,} of mine have no "
            f"reference cell at the same place (largest gap {d_fwd.max():.4g} px), "
            f"{int((d_rev > 1e-6).sum()):,} the other way")

    # the matched cells must also agree on type
    ta = np.array([int(na[k]["type"]) for k in ka])
    tb = np.array([int(nb[kb[i]]["type"]) for i in idx])
    if not (ta == tb).all():
        raise Mismatch(f"{int((ta != tb).sum()):,} matched cells have a different type")

    same_id = float(np.mean([ka[i] == kb[idx[i]] for i in range(len(ka))]))
    return (f"same {len(na):,} cells to 0 px and same types; "
            f"{same_id:.1%} carry the same id (the order is not reproducible)")


#: Bounds for :func:`cmp_segment_amp`, measured on ``skin_s4``'s rebuilt ``cellvit.json``
#: against the delivered one: 86,125 cells vs 86,119 (+0.007%), 99.65% of cells in each
#: direction sitting on a cell of the other at 0 px and 99.96% within 1 px, and 99.97% of
#: the coincident cells agreeing on type. The thresholds sit above that noise and well
#: below anything a real regression would do -- a wrong mpp, model or taxonomy, or a lost
#: spot-proximity filter, all move the cell count by percents, not by 0.007%.
AMP_COUNT_TOL = 0.001      # relative
AMP_MIN_COINCIDENT = 0.99  # within AMP_POS_TOL px, each direction
AMP_POS_TOL = 1.0          # px
AMP_MIN_TYPE_AGREE = 0.995


def cmp_segment_amp(new: Path, ref: Path) -> str:
    """A segmentation whose detections are only reproducible to within run-to-run noise.

    This is ``cellvit.json``, and it is the one deliverable here that genuinely cannot be
    reproduced exactly -- which is why it does not use the ``segment`` policy.

    The difference from HoVerNet is worth stating, because the two look alike and are not.
    HoVerNet's *detections* are exact and only their **order** varies, so ``segment``
    compares cell sets at 0 px and merely reports the id permutation. CellViT-SAM runs the
    published AMP checkpoint (``CellViT-SAM-H-x40-AMP.pth``) and merges cells across
    overlapping 1024 px patches in ray workers (``OverlapCellCleaner``), so the detections
    themselves move: borderline cells in fp16 fall on either side of threshold, and which
    duplicate of an edge cell survives depends on which worker finished first. Measured
    on ``skin_s4``, the numbers are in :data:`AMP_COUNT_TOL` and friends above.

    So this asserts the two runs found the same segmentation to within that noise, and
    reports the residual instead of hiding it.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    a = json.loads(new.read_text())
    b = json.loads(ref.read_text())
    na, nb = a.get("nuc", {}), b.get("nuc", {})
    if not na or not nb:
        raise Mismatch(f"{len(na):,} cells vs {len(nb):,}")
    rel = abs(len(na) - len(nb)) / len(nb)
    if rel > AMP_COUNT_TOL:
        raise Mismatch(f"{len(na):,} cells vs {len(nb):,} ({100 * rel:.3f}% apart, over "
                       f"the {100 * AMP_COUNT_TOL:g}% this backend's noise explains)")
    for key in ("mag", "mpp"):
        va, vb = a.get(key), b.get(key)
        if va is not None and vb is not None:
            _floats_close([float(va)], [float(vb)], key)

    ka, kb = sorted(na, key=int), sorted(nb, key=int)
    ca = np.array([na[k]["centroid"] for k in ka], dtype=float)
    cb = np.array([nb[k]["centroid"] for k in kb], dtype=float)
    d_fwd, idx = cKDTree(cb).query(ca)
    d_rev, _ = cKDTree(ca).query(cb)
    fwd = float((d_fwd <= AMP_POS_TOL).mean())
    rev = float((d_rev <= AMP_POS_TOL).mean())
    if min(fwd, rev) < AMP_MIN_COINCIDENT:
        raise Mismatch(f"only {100 * min(fwd, rev):.2f}% of cells have a counterpart "
                       f"within {AMP_POS_TOL:g} px (need {100 * AMP_MIN_COINCIDENT:g}%)")

    exact = d_fwd == 0
    ta = np.array([int(na[k]["type"]) for k in ka])
    tb = np.array([int(nb[kb[i]]["type"]) for i in idx])
    agree = float((ta[exact] == tb[exact]).mean()) if exact.any() else 0.0
    if agree < AMP_MIN_TYPE_AGREE:
        raise Mismatch(f"only {100 * agree:.2f}% of the coincident cells share a type "
                       f"(need {100 * AMP_MIN_TYPE_AGREE:g}%)")
    return (f"{len(na):,} cells vs {len(nb):,}; {100 * fwd:.2f}%/{100 * rev:.2f}% within "
            f"{AMP_POS_TOL:g} px, {100 * float(exact.mean()):.2f}% at 0 px, "
            f"{100 * agree:.2f}% of those share a type (AMP + multi-worker merge)")


#: Bounds for :func:`cmp_celltable`, measured on P2 and P5 against their delivered files.
#: The row counts are 178,601 / 178,601 for P5 and 280,600 / 282,462 for P2 -- the latter
#: short by **exactly** the 1,862 annotated cells P2's fresh segmentation did not reproduce,
#: a number that comes out of the re-key independently (see
#: ``hedest_simdata.visiumhd.cell_labels``), so the 0.66% gap is accounted for rather than
#: tolerated. Within 1 px, 99.98% of P5's rows and 99.33% of P2's find their counterpart,
#: and on those rows every level label agrees -- 100.000000%, no exceptions, which is why
#: the labels get no tolerance at all.
TABLE_COUNT_TOL = 0.01     # relative; P2 measures 0.66% of it, P5 exactly 0
TABLE_POS_TOL = 1.0        # px, as AMP_POS_TOL: sub-pixel contour drift moves a centroid
TABLE_MIN_COINCIDENT = 0.99
#: ``hovernet_type`` is HoVerNet's own PanNuke call, not a deliverable label, and a nucleus
#: whose class probability sits near a boundary can flip: 1 of P5's and 27 of P2's do.
TABLE_MIN_HN_TYPE_AGREE = 0.999


#: What ``metadata.json`` records that is **data** rather than provenance, and therefore has
#: to agree between two builds of the same dataset. Everything not listed is deliberately
#: skipped: ``date``, ``inputs`` (absolute paths), ``checks`` (this pipeline runs a different
#: set -- it dropped a soft file-mtime check that failed on the delivered P5 itself), the
#: embedding timings and device name, and ``marker_scores`` (recomputed, generous
#: thresholds). Comparing the whole file deeply fails on every one of those for reasons that
#: say nothing about the data, which would bury the ones that matter -- the palette in
#: particular is a real payload, read by HEDeST through this file.
METADATA_PATHS = (
    "seg.mpp.used", "seg.crop_key", "seg.crop_offset_xy", "seg.levels", "seg.id_rule",
    "seg.nuclei_in_hovernet", "seg.cells_annotated", "seg.nuclei_written",
    "seg.hierarchy", "seg.palette", "seg.palette_min_leaf_difference",
    "seg.unassigned.pattern", "seg.unassigned.cells_removed", "seg.unassigned.types_removed",
    "spots.grid", "spots.spots", "spots.nuclei", "spots.nuclei_in_spots",
    "spots.bins_total", "spots.bins_in_spots", "spots.genes", "spots.types_per_level",
    "embed.embed_dim", "embed.nuclei", "verify.passed",
)


def cmp_metadata(new: Path, ref: Path) -> str:
    """``metadata.json``: the declared data paths, not the whole file."""
    MISSING = object()

    def get(d, path):
        for k in path.split("."):
            if not isinstance(d, dict) or k not in d:
                return MISSING
            d = d[k]
        return d

    a, b = json.loads(new.read_text()), json.loads(ref.read_text())
    if set(a) != set(b):
        raise Mismatch(_set_report(a, b, "stages"))
    bad, checked, absent = [], 0, 0
    for path in METADATA_PATHS:
        va, vb = get(a, path), get(b, path)
        if va is MISSING and vb is MISSING:
            absent += 1
            continue
        if va is MISSING or vb is MISSING:
            bad.append(f"{path} (present in only one)")
            continue
        checked += 1
        try:
            _json_compare(va, vb, path)
        except Mismatch as exc:
            bad.append(str(exc))
    if bad:
        raise Mismatch(f"{len(bad)} of {checked + len(bad)} data fields differ: "
                       + "; ".join(bad[:4])
                       + (f" (+{len(bad) - 4} more)" if len(bad) > 4 else ""))
    return (f"the same {sorted(a)} stages and all {checked} data fields equal"
            + (f" ({absent} not recorded by either)" if absent else ""))


def cmp_celltable(new: Path, ref: Path) -> str:
    """``cell_types.csv``: rows aligned by centroid, then the level labels compared.

    The index is this build's own cell numbering. bin2cell derives it from the nucleus
    label image and a re-run renumbers -- only 1.0% of P5's ids and 0.09% of P2's came back
    the same -- so the rows arrive in a different order and a different id space. Compared
    row by row, P5 looked 26% different; compared by position, every one of its 178,567
    coincident cells agrees on all three levels.

    So this is ``segment_amp`` for a table: match cells by where they are, then demand the
    labels agree exactly, and report the coverage rather than assume it.
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree

    a = pd.read_csv(new)
    b = pd.read_csv(ref)
    if list(a.columns) != list(b.columns):
        raise Mismatch(f"columns differ: {list(a.columns)} vs {list(b.columns)}")
    for col in ("x", "y"):
        if col not in a.columns:
            raise Mismatch(f"no {col!r} column, so the rows cannot be aligned by position")
    rel = abs(len(a) - len(b)) / max(len(b), 1)
    if rel > TABLE_COUNT_TOL:
        raise Mismatch(f"{len(a):,} rows vs {len(b):,} ({100 * rel:.2f}% apart, over the "
                       f"{100 * TABLE_COUNT_TOL:g}% a segmentation re-run explains)")

    axy, bxy = a[["x", "y"]].to_numpy(float), b[["x", "y"]].to_numpy(float)
    d_fwd, idx = cKDTree(bxy).query(axy)
    back = cKDTree(axy).query(bxy)[1]
    # mutual, so one cell cannot stand in for two -- the same requirement the re-key makes
    ok = (d_fwd <= TABLE_POS_TOL) & (back[idx] == np.arange(len(axy)))
    fwd = float(ok.mean())
    rev = float(np.mean(cKDTree(axy).query(bxy)[0] <= TABLE_POS_TOL))
    if fwd < TABLE_MIN_COINCIDENT or rev < TABLE_MIN_COINCIDENT:
        raise Mismatch(f"only {100 * fwd:.2f}%/{100 * rev:.2f}% of rows pair up within "
                       f"{TABLE_POS_TOL:g} px (floor {100 * TABLE_MIN_COINCIDENT:g}%)")

    levels = [c for c in a.columns if c.startswith("lv")]
    if not levels:
        raise Mismatch("no lv* columns, so there is nothing to compare")
    ai, bi = np.arange(len(axy))[ok], idx[ok]
    for col in levels + ["spot_id"]:
        if col not in a.columns:
            continue
        same = (a[col].astype(str).to_numpy()[ai] == b[col].astype(str).to_numpy()[bi])
        if not same.all():
            raise Mismatch(f"{int((~same).sum()):,} of {len(ai):,} paired cells carry a "
                           f"different {col!r}")
    hn = 1.0
    if "hovernet_type" in a.columns:
        hn = float(np.mean(a["hovernet_type"].to_numpy()[ai]
                           == b["hovernet_type"].to_numpy()[bi]))
        if hn < TABLE_MIN_HN_TYPE_AGREE:
            raise Mismatch(f"only {100 * hn:.4f}% of paired cells share a "
                           f"'hovernet_type' (floor "
                           f"{100 * TABLE_MIN_HN_TYPE_AGREE:g}%)")
    return (f"{len(a):,} rows vs {len(b):,}; {100 * fwd:.2f}% pair up within "
            f"{TABLE_POS_TOL:g} px and all of those agree on {levels} and the spot; "
            f"{100 * hn:.4f}% share a hovernet_type (the index is a build's own)")


#: The segmentations :func:`_nucleus_centroids` reads to turn a nucleus id into a position,
#: in the order it tries them. Both Xenium slides write the first beside the files keyed by
#: those ids; Visium HD writes the second, whose ids are bin2cell's and permute just as
#: badly -- P5's embeddings cover exactly the nuclei its ``hovernet.json`` holds, and the
#: two builds' nuclei coincide 100% within 1 px, yet only 95.4% of the delivered file's
#: keys appear in the rebuild, because the id space is sparse (1..187,378 for 178,601
#: cells) and two independent numberings of it overlap by chance. Comparing those keys as
#: sets measures that chance overlap and nothing else.
SEG_NAMES = ("pannuke_fast_mask_lvl3_annotated.json", "hovernet.json")
#: A nucleus is "the same nucleus" within this distance. HoVerNet's own run-to-run
#: variation moved 31 of lung's 141,257 nuclei by up to 0.5704 px, so the tolerance has to
#: clear that; a nucleus is ~10 px across, so 1 px cannot confuse two of them.
NUCLEUS_POS_TOL = 1.0


def _nucleus_centroids(folder: Path):
    """``nucleus id -> (x, y)`` from the segmentation beside the file being compared."""
    import numpy as np

    seg = next((folder / n for n in SEG_NAMES if (folder / n).exists()), None)
    if seg is None:
        raise Mismatch(f"none of {list(SEG_NAMES)} is beside this file, so its nucleus "
                       "ids cannot be turned into positions")
    nuc = json.loads(seg.read_text())["nuc"]
    keys = list(nuc)
    return keys, np.asarray([nuc[k]["centroid"] for k in keys], dtype=float)


def _align_by_nucleus(ctx) -> tuple[dict, str]:
    """``new nucleus id -> reference nucleus id``, matched on position.

    The ids in these files are a build's own: HoVerNet emits its cells in an order that
    depends on worker scheduling and they are then renumbered 0..n-1, so the nucleus called
    "7" in one build is a different cell in the next. While the embeddings covered every
    nucleus this was invisible -- the key set was 0..n-1 either way -- but the delivered
    files hold only the ground-truth nuclei, so the key *set* became permutation-dependent
    too, and a set comparison reported 22% of lung's rows as missing when none are.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    (new_dir, ref_dir) = ctx
    ka, ca = _nucleus_centroids(new_dir)
    kb, cb = _nucleus_centroids(ref_dir)
    d, idx = cKDTree(cb).query(ca)
    ok = d <= NUCLEUS_POS_TOL
    mapping = {ka[i]: kb[idx[i]] for i in np.flatnonzero(ok)}
    note = (f"{len(mapping):,} of {len(ka):,} nuclei paired within "
            f"{NUCLEUS_POS_TOL:g} px")
    return mapping, note


def cmp_gt(new: Path, ref: Path, ctx) -> str:
    """``sim_<slide>_gt.csv``: rows paired through the segmentation, then compared.

    One row per callable nucleus, indexed by that nucleus's id -- which permutes (see
    :func:`_align_by_nucleus`). So the ids are translated through the two segmentations'
    positions first, and only then are the rows required to agree.
    """
    import numpy as np
    import pandas as pd

    a = pd.read_csv(new, index_col=0)
    b = pd.read_csv(ref, index_col=0)
    if list(a.columns) != list(b.columns):
        raise Mismatch(f"columns {list(a.columns)} vs {list(b.columns)}")
    a = a.set_axis(a.index.astype(str))
    b = b.set_axis(b.index.astype(str))
    mapping, note = _align_by_nucleus(ctx)

    mine = [k for k in a.index if mapping.get(k) in b.index]
    theirs = [mapping[k] for k in mine]
    cover_a = len(mine) / max(len(a), 1)
    cover_b = len(set(theirs)) / max(len(b), 1)
    if cover_a < 0.99 or cover_b < 0.99:
        raise Mismatch(f"{len(a):,} rows vs {len(b):,}; only {cover_a:.4%} of mine and "
                       f"{cover_b:.4%} of the reference's pair up by position ({note})")
    sa = a.loc[mine].to_numpy(float)
    sb = b.loc[theirs].to_numpy(float)
    same = np.isclose(sa, sb, rtol=RTOL, equal_nan=True).all(axis=1)
    if not same.all():
        worst = float(np.abs(sa - sb).max())
        raise Mismatch(f"{int((~same).sum()):,} of {len(mine):,} paired rows differ "
                       f"(largest value gap {worst:.4g}); {note}")
    return (f"{len(a):,} rows vs {len(b):,}; {len(mine):,} paired by position and every "
            f"one identical ({note}; the index is a build's own)")


def cmp_tensor_gt(new: Path, ref: Path, ctx) -> str:
    """``hoptimus_tile_embed_<code>.pt``: the same alignment, then keys and shapes.

    The values are not compared, for the reason :func:`cmp_tensor` gives: fp16 on a GPU.
    What has to agree is that the file covers the same cells with the same shapes.
    """
    import torch

    a = torch.load(new, map_location="cpu", weights_only=True)
    b = torch.load(ref, map_location="cpu", weights_only=True)
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise Mismatch("not a {key: tensor} mapping")
    mapping, note = _align_by_nucleus(ctx)
    paired = [k for k in a if mapping.get(k) in b]
    cover_a = len(paired) / max(len(a), 1)
    cover_b = len({mapping[k] for k in paired}) / max(len(b), 1)
    if cover_a < 0.99 or cover_b < 0.99:
        raise Mismatch(f"{len(a):,} keys vs {len(b):,}; only {cover_a:.4%} of mine and "
                       f"{cover_b:.4%} of the reference's pair up by position ({note})")
    bad = [k for k in paired
           if tuple(torch.as_tensor(a[k]).shape) != tuple(torch.as_tensor(b[mapping[k]]).shape)]
    if bad:
        raise Mismatch(f"{len(bad):,} paired cells have a different shape, e.g. {bad[:3]}")
    shapes = {tuple(torch.as_tensor(v).shape) for v in a.values()}
    return (f"{len(a):,} keys vs {len(b):,}; {len(paired):,} paired by position, all "
            f"{shapes}; values not compared (fp16 on a GPU)")


def cmp_xenium_segment(new: Path, ref: Path, ctx) -> str:
    """``pannuke_fast_mask_lvl3_annotated.json``: cells by position, cell types exactly.

    Not ``segment``: that policy requires the cell sets to coincide to 1e-6 px, which holds
    for STHELAR but not here -- HoVerNet's tiled pass over a regenerated pyramid moves 31 of
    lung's 141,257 nuclei by up to 0.5704 px.

    Not ``segment_amp`` either, and this is the interesting part. Its 99.5% type-agreement
    floor would fail this file at 98.2%, for a difference that is not a disagreement about
    any cell's type. The codes above ``len(cell_types)`` are not cell types: ``catchall``
    means "matched a Xenium cell whose type this dataset drops" and ``unmatched`` means "no
    Xenium cell matched", and lung's rebuild puts 2,554 nuclei in the first where the
    reference put them in the second. Both say "this nucleus has no ground truth", neither
    enters the ground truth, the proportions or the spots, and HEDeST reads them only to
    build a fallback palette for an overlay (``analysis/postseg.py``). The cause is in the
    dataset, not the code: the delivered archive's ``cell_boundaries`` carries the retired
    id space, so the reference's nucleus -> cell -> type hop ran there.

    So: the *callable* codes must agree cell for cell, and the split between the two
    non-callable codes is measured and reported. A real regression -- a wrong alignment, a
    lost drop-type list, a shifted palette order -- moves the callable codes, which this
    refuses to tolerate at all.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    sys.path.insert(0, str(REPO / "src"))
    from hedest_simdata import config

    a = json.loads(new.read_text())["nuc"]
    b = json.loads(ref.read_text())["nuc"]
    rel = abs(len(a) - len(b)) / max(len(b), 1)
    if rel > AMP_COUNT_TOL:
        raise Mismatch(f"{len(a):,} cells vs {len(b):,} ({100 * rel:.3f}% apart, over the "
                       f"{100 * AMP_COUNT_TOL:g}% a segmentation re-run explains)")
    ka, kb = list(a), list(b)
    ca = np.asarray([a[k]["centroid"] for k in ka], float)
    cb = np.asarray([b[k]["centroid"] for k in kb], float)
    d, idx = cKDTree(cb).query(ca)
    rev = float(np.mean(cKDTree(ca).query(cb)[0] <= NUCLEUS_POS_TOL))
    ok = d <= NUCLEUS_POS_TOL
    fwd = float(ok.mean())
    if fwd < AMP_MIN_COINCIDENT or rev < AMP_MIN_COINCIDENT:
        raise Mismatch(f"only {100 * fwd:.2f}%/{100 * rev:.2f}% of cells sit within "
                       f"{NUCLEUS_POS_TOL:g} px of one of the other's "
                       f"(floor {100 * AMP_MIN_COINCIDENT:g}%)")

    ta = np.asarray([int(a[k]["type"]) for k in ka])[ok]
    tb = np.asarray([int(b[kb[i]]["type"]) for i in idx[ok]])
    # the number of real cell types is the smallest code that means "no ground truth"
    n_types = min(len(sl.cell_types) for sl in config.XENIUM_SLIDES.values())
    mine_call, theirs_call = ta < n_types, tb < n_types
    either = mine_call | theirs_call
    both = mine_call & theirs_call
    flipped = int((ta[both] != tb[both]).sum())          # both callable, different type
    one_sided = int((mine_call != theirs_call).sum())    # callable in one build only
    agree = 1.0 - (flipped + one_sided) / max(int(either.sum()), 1)
    if agree < AMP_MIN_TYPE_AGREE:
        raise Mismatch(
            f"only {100 * agree:.3f}% of the {int(either.sum()):,} cells callable in "
            f"either build agree (floor {100 * AMP_MIN_TYPE_AGREE:g}%): {flipped:,} carry "
            f"a different cell type, {one_sided:,} are callable in one build only")
    swapped = int((ta != tb).sum()) - flipped - one_sided
    return (f"{len(a):,} cells vs {len(b):,}; {100 * fwd:.2f}%/{100 * rev:.2f}% coincide "
            f"within {NUCLEUS_POS_TOL:g} px, {100 * float((d[ok] == 0).mean()):.2f}% at "
            f"0 px; cell types agree on {100 * agree:.3f}% of the {int(either.sum()):,} "
            f"callable cells ({flipped:,} a different type, {one_sided:,} callable in one "
            f"build only); {swapped:,} more differ only in which non-callable code they "
            f"carry")


def cmp_spotdict(new: Path, ref: Path) -> str:
    """``spot -> [cell ids]``, compared on what survives a cell-id permutation.

    The cell ids inside are one build's own (see :func:`cmp_segment`), so the spot set and
    the number of cells in each spot are what can be compared; the identity of those cells
    is checked instead by the pipeline's own ``verify``, which confirms within a build that
    every id here exists in that build's segmentation.
    """
    a = json.loads(new.read_text())
    b = json.loads(ref.read_text())
    if set(a) != set(b):
        def _sizes(keep):
            n = sum(1 for k in keep if len(a[k]) == len(b[k]))
            held = sum(len(a[k]) for k in keep)
            return (f"{n:,} of those {len(keep):,} hold the same number of cells, "
                    f"{held:,} cells in total")
        extra = sorted(set(a) - set(b), key=lambda k: -len(a[k]))[:3]
        raise Mismatch(_set_report(a, b, "spots", _sizes)
                       + (f"; largest extra spots hold {[len(a[k]) for k in extra]} cells"
                          if extra else ""))
    bad = {k: (len(a[k]), len(b[k])) for k in a if len(a[k]) != len(b[k])}
    if bad:
        raise Mismatch(f"{len(bad)} of {len(a)} spots hold a different number of cells, "
                       f"e.g. {dict(list(bad.items())[:3])}")
    total = sum(len(v) for v in a.values())
    return (f"same {len(a):,} spots holding the same {total:,} cells each; the ids inside "
            "are a build's own")


POLICIES = {"bytes": cmp_bytes, "json": cmp_json, "csv": cmp_csv, "h5ad": cmp_h5ad,
            "tensor": cmp_tensor, "tiff": cmp_tiff, "figure": cmp_figure,
            "segment": cmp_segment, "segment_amp": cmp_segment_amp,
            "celltable": cmp_celltable, "spotdict": cmp_spotdict,
            "gt": cmp_gt, "tensor_gt": cmp_tensor_gt,
            "xenium_segment": cmp_xenium_segment, "metadata": cmp_metadata}


# --------------------------------------------------------------------------------------


def targets(built: Path, reference: Path, families, datasets):
    """``(label, relative dir, [FileSpec])`` for everything to compare."""
    out = []
    if "sthelar" in families:
        for name in config.BENCH_NAMES:
            if datasets and name not in datasets:
                continue
            rel = Path("STHELAR") / "bench_data" / name
            lv = reference / rel / "sim" / "levels.json"
            n = int(json.loads(lv.read_text())["n_levels"]) if lv.exists() else 5
            out.append((f"sthelar:{name}", rel, manifest.sthelar_files(name, n)))
    if "visiumhd" in families:
        for p in config.VISIUMHD_PATIENTS:
            if datasets and p not in datasets:
                continue
            rel = Path(config.visiumhd_name(p)) / "pseudovisium"
            out.append((f"visiumhd:{p}", rel, manifest.visiumhd_files(p)))
    if "xenium" in families:
        for k, spec in config.XENIUM_SLIDES.items():
            if datasets and k not in datasets:
                continue
            rel = Path(spec.name) / spec.sim_subdir
            out.append((f"xenium:{k}", rel, manifest.xenium_files(k)))
    return out


#: Differences that were measured and explained, so a rebuild can be checked for *new*
#: ones. See the file's own ``_README``.
EXPECTED_PATH = REPO / "tests" / "expected_residuals.json"


def load_expected(path: Path = EXPECTED_PATH) -> dict[str, str]:
    """``'<family>:<dataset>/<path>' -> why``, patterns allowed, ``_*`` keys ignored."""
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text()).items()
            if not k.startswith("_")}


def expected_reason(expected: dict[str, str], key: str) -> str | None:
    """The reason this difference is known, or None if it is new.

    Exact keys win over patterns, so a dataset can be called out individually even when its
    family is covered -- otherwise adding a broad pattern would quietly swallow a specific
    entry someone wrote for a reason.
    """
    if key in expected:
        return expected[key]
    best = None
    for pat, why in expected.items():
        if "*" not in pat:
            continue
        if fnmatch.fnmatch(key, pat) and (best is None or len(pat) > len(best[0])):
            best = (pat, why)
    return best[1] if best else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--built", required=True)
    ap.add_argument("--reference", required=True,
                    help="the data root to compare against")
    ap.add_argument("--families", nargs="+", default=["sthelar", "visiumhd", "xenium"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--quiet", action="store_true", help="only print problems")
    ap.add_argument("--strict", action="store_true",
                    help="treat the known residuals in tests/expected_residuals.json as "
                         "failures too")
    args = ap.parse_args(argv)

    built, reference = Path(args.built), Path(args.reference)
    totals = {"equal": 0, "differs": 0, "known": 0, "missing": 0, "no-reference": 0}
    problems: list[str] = []
    expected = {} if args.strict else load_expected()
    matched: set[str] = set()
    #: the datasets this invocation looked at, so the stale report below can tell "this
    #: entry is fixed" from "--families/--datasets did not include it this time"
    seen: set[str] = set()

    for label, rel, specs in targets(built, reference, set(args.families), args.datasets):
        print(f"\n=== {label}  ({rel})")
        seen.add(label)
        for spec in specs:
            n, r = built / rel / spec.path, reference / rel / spec.path
            if not n.exists():
                totals["missing"] += 1
                problems.append(f"{label}/{spec.path}: not built")
                print(f"  MISSING   {spec.path}")
                continue
            if not r.exists():
                totals["no-reference"] += 1
                if not args.quiet:
                    print(f"  (no ref)  {spec.path}")
                continue
            try:
                fn = POLICIES[spec.policy]
                # a policy that needs a sibling file (the ground truth needs the
                # segmentation, to turn permuting nucleus ids back into positions) asks
                # for the two folders by taking a third parameter
                detail = (fn(n, r, (built / rel, reference / rel))
                          if len(inspect.signature(fn).parameters) >= 3 else fn(n, r))
            except Mismatch as exc:
                key = f"{label}/{spec.path}"
                why = expected_reason(expected, key)
                if why is None:
                    totals["differs"] += 1
                    problems.append(f"{key}: {exc}")
                    print(f"  DIFFERS   {spec.path:<46} {exc}")
                else:
                    totals["known"] += 1
                    matched.add(key)
                    if not args.quiet:
                        print(f"  known     {spec.path:<46} {exc}")
            except Exception as exc:                        # unreadable is a result too
                totals["differs"] += 1
                problems.append(f"{label}/{spec.path}: {type(exc).__name__}: {exc}")
                print(f"  ERROR     {spec.path:<46} {type(exc).__name__}: {exc}")
            else:
                totals["equal"] += 1
                if not args.quiet:
                    print(f"  equal     {spec.path:<46} [{spec.policy}] {detail}")

    print("\n" + "=" * 70)
    print(f"equal {totals['equal']}   known residuals {totals['known']}   "
          f"differs {totals['differs']}   not built {totals['missing']}   "
          f"no reference {totals['no-reference']}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems[:40]:
            print(f"  {p}")
    if expected:
        # an entry that no longer matches anything is either fixed or mis-keyed, and either
        # way should not sit in the file being believed. Only entries whose dataset was
        # actually compared are judged: the families run as separate jobs, and reporting
        # every other family's entries as stale in each one made the report worthless --
        # it listed 7 to 9 "stale" entries per run, all of them live in a sibling job.
        candidates = [k for k in expected
                      if any(fnmatch.fnmatch(lbl, k.split("/", 1)[0]) for lbl in seen)]
        stale = [k for k in candidates if not any(
            k == m or ("*" in k and fnmatch.fnmatch(m, k)) for m in matched)]
        if stale:
            print(f"\n{len(stale)} of the {len(candidates)} entry/entries in "
                  f"{EXPECTED_PATH.name} covering the datasets compared here matched "
                  "nothing (fixed, or mis-keyed):")
            for k in stale:
                print(f"  {k}")
        skipped = len(expected) - len(candidates)
        if skipped and not args.quiet:
            print(f"\n({skipped} entry/entries name datasets this run did not compare.)")
    if not args.strict and totals["known"]:
        print(f"\n{totals['known']} known residual(s) were not counted as failures; "
              f"--strict counts them.")
    return 1 if (totals["differs"] or totals["missing"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
