"""
HoVerNet nucleus segmentation for the Visium HD bin2cell pipeline
==================================================================

Drop-in replacement for the StarDist segmentation step of
``visiumhd_to_cell.py``. Instead of segmenting a downscaled (0.5 mpp) render of
the H&E, this runs HoVerNet at the *full* resolution of the source ``.btf``,
restricted to the capture area.

Why the crop matters: the colon slides are ~4 Gpx while the Visium capture area
covers only ~15% of that. Cropping first turns an intractable full-slide run
into ~600 Mpx, and is also required for a second reason -- HoVerNet reads slides
through OpenSlide, which does not accept ``.btf`` and needs a real pyramid
(the source has a single level).

Pipeline (all geometry in the *cropped full-resolution* H&E frame):

  1. ``crop_capture_area``  bin2cell decides the crop box (``b2c.get_crop``), we
     do a windowed read of the ``.btf`` and write a tiled, pyramidal ``.tif``
     whose XResolution encodes the true mpp so OpenSlide reports 40x.
  2. ``run_hovernet``       shells out to the ``hovernet`` conda env (Python 3.8,
     separate from bin2cell's 3.10) for tissue mask + WSI inference.
  3. ``rasterize_json``     fills HoVerNet's polygons into an int32 label array
     and saves it as the same sparse ``.npz`` that ``b2c.stardist()`` writes, so
     ``b2c.insert_labels()`` consumes it unchanged.

The label array is at the *source* mpp, so downstream ``insert_labels`` must be
called with ``mpp=<microns_per_pixel from scalefactors>`` (scale factor 1.0),
not the 0.5 used by the StarDist path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..hedest_bridge import Bridge
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import scipy.sparse
import tifffile as tf
import zarr

import bin2cell as b2c

# --- external HoVerNet install ----------------------------------------------
# These four used to be hard-coded here. They are the only site-specific things in the
# Visium HD segmentation path, and they now come from the bridge, which reads them from the
# command line or the environment -- see hedest_simdata.hedest_bridge.
#   HOVERNET_REPO  -> bridge.hovernet_repo
#   HOVERNET_ENV   -> bridge.hovernet_env
#   HOVERNET_MODEL -> --hovernet-weights / $HOVERNET_WEIGHTS
#   CONDA_SH       -> --conda-sh / $SIMDATA_CONDA_SH

# PanNuke classes emitted by the `fast` type model (see hovernet/type_info.json).
PANNUKE_TYPES = {
    0: "nolabe",
    1: "neopla",
    2: "inflam",
    3: "connec",
    4: "necros",
    5: "no-neo",
}

TILE = 512          # tile size of the written pyramid
PYRAMID_LEVELS = 5  # level 0 + 4 reductions; mask generation needs level 3


# ---------------------------------------------------------------------------
# 1. crop + pyramidal TIFF
# ---------------------------------------------------------------------------

def _pyramid_write(img: np.ndarray, out_path: Path, mpp: float) -> None:
    """Write ``img`` as a tiled, multi-level (generic) pyramidal TIFF.

    OpenSlide's generic-tiff driver wants each level as its own top-level IFD
    (reduced-image subfiletype), all tiled -- not SubIFDs. The resolution tags
    carry the true mpp, which is how ``wsi_handler.find_slide_magnification``
    recovers 40x for these slides (0.274 / 0.25 rounds to 1).
    """
    # pixels per centimeter
    res = 1.0 / (mpp * 1e-4)
    opts = dict(
        tile=(TILE, TILE),
        photometric="rgb",
        compression="deflate",
        resolution=(res, res),
        resolutionunit="CENTIMETER",
    )

    with tf.TiffWriter(out_path, bigtiff=True) as writer:
        writer.write(img, **opts)
        level = img
        for _ in range(PYRAMID_LEVELS - 1):
            h, w = level.shape[:2]
            if min(h, w) < TILE:
                break
            level = cv2.resize(
                level, (w // 2, h // 2), interpolation=cv2.INTER_AREA
            )
            writer.write(level, subfiletype=1, **opts)


def crop_capture_area(
    adata,
    btf_path: str,
    out_tif: Path,
    buffer: int = 150,
    spatial_cropped_key: Optional[str] = None,
    preview_mpp: float = 0.5,
    row_chunk: int = 4096,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int]]:
    """Crop the source H&E to the capture area and write a pyramidal TIFF.

    Mirrors what ``b2c.scaled_he_image(crop=True, buffer=...)`` does to the
    object -- it sets ``obsm[spatial_cropped_key]`` and stores a preview image
    plus its scale factor -- but reads the ``.btf`` through a windowed zarr
    instead of loading all ~4 Gpx into RAM.

    Returns ``(crop_coords, crop_shape)`` with crop_coords as PIL-style
    ``(left, upper, right, lower)`` in full-resolution pixels.
    """
    library = list(adata.uns["spatial"].keys())[0]
    mpp_source = adata.uns["spatial"][library]["scalefactors"]["microns_per_pixel"]

    # Let bin2cell define the box so the frame matches insert_labels exactly.
    left, upper, right, lower = b2c.get_crop(
        adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=buffer
    )

    store = tf.imread(btf_path, aszarr=True)
    try:
        z = zarr.open(store, mode="r")
        # a pyramidal source exposes a group; these .btf files are a plain array
        arr = z[0] if isinstance(z, zarr.hierarchy.Group) else z
        img_h, img_w = arr.shape[:2]

        # clip exactly the way numpy slicing would on the full array
        left, upper = int(max(left, 0)), int(max(upper, 0))
        right, lower = int(min(right, img_w)), int(min(lower, img_h))
        h, w = lower - upper, right - left

        print(f"  source image : {img_h} x {img_w}")
        print(f"  capture crop : rows {upper}:{lower}  cols {left}:{right}")
        print(f"  crop shape   : {h} x {w}  ({h * w / 1e6:.0f} Mpx, "
              f"{100 * h * w / (img_h * img_w):.1f}% of slide)")

        crop = np.empty((h, w, 3), dtype=np.uint8)
        for y0 in range(0, h, row_chunk):
            y1 = min(y0 + row_chunk, h)
            crop[y0:y1] = arr[upper + y0:upper + y1, left:right, :3]
    finally:
        store.close()

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    print(f"  writing pyramidal TIFF -> {out_tif}")
    _pyramid_write(crop, out_tif, mpp_source)

    # Same bookkeeping scaled_he_image() would have done.
    if spatial_cropped_key is None:
        spatial_cropped_key = f"spatial_cropped_{buffer}_buffer"
    adata.obsm[spatial_cropped_key] = adata.obsm["spatial"].copy()
    adata.obsm[spatial_cropped_key][:, 0] -= left
    adata.obsm[spatial_cropped_key][:, 1] -= upper

    # A small render kept only so the existing QC plots keep working; storing
    # the full-resolution crop in .uns would add ~1.8 GB to the h5ad.
    scalef = mpp_source / preview_mpp
    preview = cv2.resize(
        crop, (int(w * scalef), int(h * scalef)), interpolation=cv2.INTER_AREA
    )
    img_key = f"{preview_mpp}_mpp_{buffer}_buffer"
    adata.uns["spatial"][library]["images"][img_key] = preview
    adata.uns["spatial"][library]["scalefactors"][f"tissue_{img_key}_scalef"] = scalef
    print(f"  preview image key: {img_key}  {preview.shape}")

    del crop
    return (left, upper, right, lower), (h, w)


# ---------------------------------------------------------------------------
# 2. external HoVerNet inference
# ---------------------------------------------------------------------------

def _conda_run(bridge: "Bridge", args: list, cwd: Path) -> None:
    """Run a command inside the hovernet conda env.

    ``run_infer.py`` imports ``external.hovernet.*``, which only resolves with the HEDeST
    root on ``sys.path`` -- running the script puts its own directory there, not the
    working directory, so it has to go through ``PYTHONPATH``. The bridge does that.
    """
    print(f"  $ {' '.join(str(a) for a in args)}", flush=True)
    bridge.run_in_hovernet_env(args, cwd=cwd)


def run_hovernet(
    bridge: "Bridge",
    tif_path: Path,
    out_dir: Path,
    mask_dir: Path,
    cache_dir: Path,
    gpu: str = "0",
    batch_size: int = 16,
    nr_inference_workers: int = 8,
    nr_post_proc_workers: int = 16,
    mask_level: int = 3,
) -> Path:
    """Tissue mask + HoVerNet WSI inference. Returns the nuclei JSON path."""
    # Everything below runs with cwd=bridge.hovernet_repo, so every path handed to a
    # subprocess has to be absolute -- a relative one would resolve against the
    # HoVerNet checkout instead of the caller's working directory.
    tif_path = Path(tif_path).resolve()
    out_dir = Path(out_dir).resolve()
    mask_dir = Path(mask_dir).resolve()
    cache_dir = Path(cache_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mask_png = mask_dir / f"{tif_path.stem}.png"
    if not mask_png.exists():
        print("\n  [1/2] tissue mask")
        _conda_run(
            bridge,
            ["python3", "-u", "run_mask.py", tif_path, mask_png,
             "--mask-level", mask_level],
            cwd=bridge.hovernet_repo,
        )
        # run_mask.py saves via cv2.imwrite, which returns False rather than
        # raising if the path is unwritable -- so check rather than trust it.
        if not mask_png.exists():
            raise FileNotFoundError(
                f"run_mask.py reported success but wrote no mask at {mask_png}"
            )
    else:
        print(f"\n  [1/2] tissue mask exists, reusing {mask_png}")

    print("\n  [2/2] HoVerNet WSI inference")
    _conda_run(
        bridge,
        ["python3", "-u", "run_infer.py",
         f"--gpu={gpu}",
         "--nr_types=6",
         f"--type_info_path={bridge.hovernet_repo / 'type_info.json'}",
         f"--batch_size={batch_size}",
         "--model_mode=fast",
         f"--model_path={bridge.hovernet_weights_path()}",
         f"--nr_inference_workers={nr_inference_workers}",
         f"--nr_post_proc_workers={nr_post_proc_workers}",
         "wsi",
         f"--input_dir={tif_path.parent}",
         f"--output_dir={out_dir}",
         "--skip_image_dict",
         f"--input_mask_dir={mask_dir}",
         f"--cache_path={cache_dir}",
         "--proc_mag=40"],
        cwd=bridge.hovernet_repo,
    )

    json_path = out_dir / "json" / f"{tif_path.stem}.json"
    if not json_path.exists():
        alt = out_dir / f"{tif_path.stem}.json"
        if alt.exists():
            json_path = alt
        else:
            raise FileNotFoundError(
                f"HoVerNet produced no JSON at {json_path} or {alt}"
            )
    return json_path


# ---------------------------------------------------------------------------
# 3. JSON -> sparse label array
# ---------------------------------------------------------------------------

def rasterize_json(
    json_path: Path,
    shape: Tuple[int, int],
    out_npz: Path,
    types_csv: Optional[Path] = None,
) -> Dict[str, int]:
    """Fill HoVerNet nucleus contours into an int32 label array.

    Produces exactly what ``b2c.stardist()`` writes -- a CSR sparse int32 matrix
    of shape ``(H, W)`` where 0 is background and each nucleus has a unique
    label from 1..n -- so ``b2c.insert_labels()`` reads it unchanged.

    Labels are assigned in JSON order. If ``types_csv`` is given, the PanNuke
    class of every label is written alongside, which lets downstream code keep
    HoVerNet's typing without re-reading the JSON.
    """
    with open(json_path) as fh:
        payload = json.load(fh)
    nuc = payload.get("nuc", payload)
    print(f"  {len(nuc)} nuclei in {json_path.name}")

    labels = np.zeros(shape, dtype=np.int32)
    rows = []
    skipped = 0

    for i, (key, entry) in enumerate(nuc.items(), start=1):
        contour = entry.get("contour")
        if not contour or len(contour) < 3:
            skipped += 1
            continue
        # HoVerNet contours are [x, y]; cv2.fillPoly wants that order too.
        poly = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(labels, [poly], int(i))
        if types_csv is not None:
            cx, cy = entry.get("centroid", (np.nan, np.nan))
            rows.append(
                (i, key, int(entry.get("type", 0)),
                 float(entry.get("type_prob", np.nan)), float(cx), float(cy))
            )

    if skipped:
        print(f"  skipped {skipped} nuclei with degenerate contours")

    # A polygon drawn over an earlier one steals its pixels; a nucleus fully
    # covered that way would vanish, so report it rather than fail silently.
    present = np.unique(labels)
    n_present = len(present) - (1 if present[0] == 0 else 0)
    lost = (len(nuc) - skipped) - n_present
    if lost > 0:
        print(f"  note: {lost} nuclei fully overlapped by later ones")

    coverage = 100.0 * np.count_nonzero(labels) / labels.size
    print(f"  label coverage: {coverage:.1f}% of cropped area")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    scipy.sparse.save_npz(out_npz, scipy.sparse.csr_matrix(labels))
    print(f"  saved {out_npz}")

    if types_csv is not None and rows:
        import pandas as pd
        pd.DataFrame(
            rows, columns=["label", "hovernet_id", "type", "type_prob", "x", "y"]
        ).to_csv(types_csv, index=False)
        print(f"  saved {types_csv}")

    return {
        "n_json": len(nuc),
        "n_skipped": skipped,
        "n_labels": int(n_present),
        "coverage_pct": float(coverage),
    }
