"""Stage the H&E and drive HoVerNet mask + WSI inference for one dataset.

Reuses the HEDeST HoVerNet code unchanged:
  * ``external/hovernet/run_mask.py``  -> Otsu tissue mask (level 3);
  * ``external/hovernet/run_infer.py`` -> WSI segmentation JSON (reindexed
    0..n-1) + a ``image_dict.pt`` of 64px/20um cell crops.

``run_infer.py`` requires the input dir to contain exactly one file and names
its outputs after that file, so we stage a single-file input dir. To recover the
~46px patch-context strip at the crop edge (HoVerNet fast mode cannot predict
within (256-164)/2 px of the image border), the staged tiff is a TEMPORARY
reflect-padded copy of ``he.tiff`` (``PAD_PX`` on each side); after inference the
coordinates are shifted back, padding cells dropped, and the padded tiff deleted
(:func:`_finalize_padded`). We pass the dataset's own ``mpp`` so the 20um cell
windows are physically correct; ``size_px=64 size_um=20``.

**The cell ids this produces are one build's own.** HoVerNet's detections reproduce
exactly -- a rebuild of ``skin_s4`` found the identical 98,678 cells, each at a distance of
0 px from a reference cell, with the same types -- but the order it emits them in does not,
because its WSI post-processing runs across several workers. The 0..n-1 reindexing in
:func:`_finalize_padded` therefore permutes: only 68.7% of cells kept their id. Everything
in one build shares that id space and ``verify`` checks it, but anything outside the build
that is keyed by these ids -- notably HEDeST's ``STHELAR/matches/gt_<name>_<level>.json``,
which the gridsearch reads -- has to be rebuilt after a regeneration.

Everything lands under ``_work/sthelar/hovernet/<name>/``. In HEDeST-bench this was
``<DATA_ROOT>/hier_sim/<name>/_staging``, named after a dataset family that no longer
exists; the paths and the weights are now resolved by
:class:`~hedest_simdata.hedest_bridge.Bridge` instead of being constants in this file.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Layout
from ..hedest_bridge import Bridge
from ..hedest_bridge import MASK_LEVEL
from . import paths

# Reflect-pad the H&E by this many px before inference. HoVerNet fast mode
# predicts a 164px output tile from a 256px input, so every output pixel needs
# (256-164)/2 = 46px of surrounding context -> the model cannot segment within
# ~46px of the true image edge (the residual edge strip). Padding gives true-edge
# pixels that context; coordinates are shifted back and out-of-pad cells dropped
# afterwards, so no real cell is lost at the crop boundary before matching.
PAD_PX = 64


def _staging_paths(ds: paths.Dataset):
    st = ds.staging_dir
    wsi_dir = st / "wsi"          # single-file input dir for run_infer
    mask_dir = st / "mask"
    seg_dir = st                  # seg json written straight into _staging
    wsi = wsi_dir / f"{ds.name}.tiff"
    mask = mask_dir / f"{ds.name}.png"
    return st, wsi_dir, mask_dir, seg_dir, wsi, mask


def stage_inputs(layout: Layout, name: str) -> None:
    """Write a TEMPORARY reflect-padded copy of he.tiff as the inference input.

    The padded tiff (single-level tiled BigTIFF, openslide-readable) lives only in
    ``_staging/wsi/`` and is deleted after inference (:func:`_finalize_padded`).
    Everything HoVerNet sees is in the padded frame; downstream sees crop-local.
    """
    import numpy as np
    import openslide
    import PIL.Image
    import tifffile
    PIL.Image.MAX_IMAGE_PIXELS = None

    ds = paths.dataset(layout, name)
    st, wsi_dir, mask_dir, seg_dir, wsi, mask = _staging_paths(ds)
    wsi_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    slide = openslide.open_slide(str(ds.he_tiff))
    W, H = slide.level_dimensions[0]
    img = np.asarray(slide.read_region((0, 0), 0, (W, H)).convert("RGB"))
    padded = np.pad(img, ((PAD_PX, PAD_PX), (PAD_PX, PAD_PX), (0, 0)), mode="reflect")
    del img
    if wsi.exists() or wsi.is_symlink():
        wsi.unlink()
    tifffile.imwrite(str(wsi), padded, tile=(512, 512), photometric="rgb",
                     compression="deflate", bigtiff=True)
    del padded


def run_mask(layout: Layout, name: str) -> None:
    """Otsu tissue mask WITHOUT border clearing.

    HEDeST's ``make_auto_mask`` applies ``clear_border(prop=30)``, which zeroes a
    1/30 border (~195um top / ~170um sides here) -> HoVerNet never segments that
    band and we lose every edge cell (defeating the 12um margin extension). Our
    crops are already tight to tissue+margin, so we keep the whole tissue region,
    edges included. Downsampled by stride (cheap, no full-res float resize).
    """
    import cv2
    import numpy as np
    import openslide
    import PIL.Image
    PIL.Image.MAX_IMAGE_PIXELS = None      # crops can exceed PIL's default cap
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu
    from skimage.morphology import (closing, opening, remove_small_holes,
                                    remove_small_objects, square)

    ds = paths.dataset(layout, name)
    _, _, _, _, wsi, mask = _staging_paths(ds)
    mask.parent.mkdir(parents=True, exist_ok=True)

    slide = openslide.open_slide(str(wsi))
    W, H = slide.level_dimensions[0]
    im = np.asarray(slide.read_region((0, 0), 0, (W, H)).convert("RGB"))
    step = 2 ** MASK_LEVEL
    gray = rgb2gray(im[::step, ::step])
    del im
    sel = (gray > 0.05) & (gray < 0.95)
    thr = float(threshold_otsu(gray[sel])) if sel.any() else 0.8
    m = (gray < thr) & (gray > 0.05)          # tissue = darker than background
    m = opening(closing(m, square(8)), square(8))
    m = remove_small_objects(m, min_size=64)
    m = remove_small_holes(m, area_threshold=256)
    out = (np.stack([m] * 3, axis=-1).astype(np.uint8)) * 255
    cv2.imwrite(str(mask), out)


def run_infer(layout: Layout, bridge: Bridge, name: str, gpu: str = "0") -> None:
    """HoVerNet WSI inference on the staged, padded H&E, then back to crop coordinates."""
    ds = paths.dataset(layout, name)
    st, wsi_dir, mask_dir, seg_dir, wsi, mask = _staging_paths(ds)
    image_dict = st / "image_dict_raw.pt"
    # run_infer.py SKIPS segmentation if any *.json already exists in output_dir,
    # so a stale seg JSON from a prior run would be silently reused (wrong mask,
    # missing edge cells). Remove prior outputs to force a fresh segmentation.
    for stale in (seg_dir / f"{name}.json", seg_dir / f"{name}.geojson", image_dict):
        if stale.exists():
            stale.unlink()
    bridge.hovernet_infer(wsi_dir=wsi_dir, mask_dir=mask_dir, out_dir=seg_dir,
                          mpp=ds.mpp, cache_dir=st / "cache", image_dict=image_dict,
                          gpu=gpu)
    _finalize_padded(layout, name)


def _finalize_padded(layout: Layout, name: str) -> None:
    """Shift seg back to crop-local coords, drop out-of-pad cells, reindex.

    HoVerNet ran in the padded frame; every centroid/bbox/contour is offset by
    ``PAD_PX``. We subtract it, keep only cells whose centroid lands inside the
    original ``[0,W]x[0,H]`` (drops the reflected padding cells), reindex 0..m-1,
    and filter the raw ``image_dict`` to match (its tiles were extracted in the
    padded frame, so true-edge cells already have complete windows). Then the
    temporary padded he.tiff is deleted.
    """
    import json

    import openslide
    import torch

    ds = paths.dataset(layout, name)
    st, _, _, seg_dir, wsi, _ = _staging_paths(ds)
    W, H = openslide.open_slide(str(ds.he_tiff)).level_dimensions[0]
    P = PAD_PX

    seg = seg_dir / f"{name}.json"
    with open(seg) as f:
        data = json.load(f)
    nuc = data["nuc"]
    img_path = st / "image_dict_raw.pt"
    images = torch.load(img_path, weights_only=True)

    new_nuc: dict = {}
    new_images: dict = {}
    k = 0
    for old_id, e in nuc.items():          # keys "0".."n-1", same order as images
        cx, cy = e["centroid"]
        ox, oy = cx - P, cy - P
        if not (0 <= ox < W and 0 <= oy < H):
            continue                       # cell lives in the reflected padding
        e2 = dict(e)
        e2["centroid"] = [ox, oy]
        e2["contour"] = [[px - P, py - P] for px, py in e["contour"]]
        bb = e["bbox"]
        e2["bbox"] = [[bb[0][0] - P, bb[0][1] - P], [bb[1][0] - P, bb[1][1] - P]]
        new_nuc[str(k)] = e2
        if old_id in images:
            new_images[str(k)] = images[old_id]
        k += 1

    with open(seg, "w") as f:
        json.dump({"mag": data.get("mag"), "mpp": data.get("mpp"), "nuc": new_nuc}, f)
    torch.save(new_images, img_path)
    # padded frame geojson is now wrong-frame; drop it. Delete the temp padded tiff.
    (seg_dir / f"{name}.geojson").unlink(missing_ok=True)
    if wsi.exists():
        wsi.unlink()
