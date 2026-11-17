"""Diagnostic plots for one fixed 2000x2000 crop of every bench_data dataset.

* ``00_slide_overview_hovernet.png`` / ``00_slide_overview_dapi.png`` -- the whole
  slide (crop) downsampled + faded, every cell drawn as a point (HoVerNet cells by
  raw HoVerNet class, DAPI cells by finest-level/label2 type) with the spot circles
  overlaid, so cell coverage across the whole slide is visible at a glance;
* ``00_hovernet_seg.png``      -- H&E crop, HoVerNet cells filled by their raw
  HoVerNet class colour (external/hovernet/type_info.json) + spot circles;
* ``01_dapi_seg_level<k>.png`` -- same crop, DAPI ground-truth cells filled by
  their level-<k> cell-type colour + spot circles;
* ``02_prop_level<k>.png``     -- same crop, each spot circle a pie chart of the
  level-<k> cell-type proportions;
* ``03_levels_tree.png``       -- the level hierarchy as a left-to-right diagram
  (boxes linked by arrows parent -> child), showing what each level contains.

Cell-type colours come from :mod:`bench.colors`, which looks every category up by
name in the shared hand-written table, so a type reads the same across levels.
The raw HoVerNet-class colours of the two plots above are separate and unrelated.
"""
from __future__ import annotations

import json

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, FancyArrow, Patch, Rectangle, Wedge

from ..config import Layout
from ..hedest_bridge import Bridge
from . import hnio
from . import bench_hier, colors, paths

WINDOW = 2000
OVERVIEW_MAX_DIM = 3000     # px for the downsampled whole-slide background


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def pick_window(ds: paths.Dataset, size: int = WINDOW) -> tuple[int, int, int, int]:
    """A ``size x size`` H&E window centred on the densest spot region."""
    adata = ad.read_h5ad(ds.adata_out)
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    slide = openslide.open_slide(str(ds.he_out))
    W, H = slide.dimensions
    g = 250
    nx, ny = int(np.ceil(W / g)), int(np.ceil(H / g))
    cnt = np.zeros((ny, nx))
    gx = np.clip((xy[:, 0] // g).astype(int), 0, nx - 1)
    gy = np.clip((xy[:, 1] // g).astype(int), 0, ny - 1)
    np.add.at(cnt, (gy, gx), 1)
    integ = np.pad(np.cumsum(np.cumsum(cnt, 0), 1), ((1, 0), (1, 0)))
    k = size // g
    best, best_ij = -1, (0, 0)
    for i in range(0, max(1, ny - k + 1)):
        for j in range(0, max(1, nx - k + 1)):
            s = (integ[i + k, j + k] - integ[i, j + k]
                 - integ[i + k, j] + integ[i, j])
            if s > best:
                best, best_ij = s, (i, j)
    y0 = min(best_ij[0] * g, max(0, H - size))
    x0 = min(best_ij[1] * g, max(0, W - size))
    return int(x0), int(y0), int(x0 + size), int(y0 + size)


def read_he(ds: paths.Dataset, window) -> np.ndarray:
    x0, y0, x1, y1 = window
    slide = openslide.open_slide(str(ds.he_out))
    return np.asarray(slide.read_region((x0, y0), 0, (x1 - x0, y1 - y0)).convert("RGB"))


def _cells_in_window(nuc: dict, window):
    x0, y0, x1, y1 = window
    out = []
    for cid, e in nuc.items():
        cx, cy = e["centroid"]
        if x0 <= cx < x1 and y0 <= cy < y1:
            c = hnio.contour_xy(e).astype(float) - [x0, y0]
            if len(c) >= 3:
                out.append((cid, e, c))
    return out


def _spots_in_window(ds: paths.Dataset, window):
    x0, y0, x1, y1 = window
    adata = ad.read_h5ad(ds.adata_out)
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    rad = adata.uns["spatial"][ds.adata_name]["scalefactors"]["spot_diameter_fullres"] / 2
    ids = list(adata.obs.index)
    out = []
    for sid, (sx, sy) in zip(ids, xy):
        if x0 - rad <= sx < x1 + rad and y0 - rad <= sy < y1 + rad:
            out.append((sid, sx - x0, sy - y0))
    return out, rad


# ---------------------------------------------------------------------------
# whole-slide overview: every cell as a point + spot circles
# ---------------------------------------------------------------------------
def _read_he_overview(ds: paths.Dataset, max_dim: int = OVERVIEW_MAX_DIM):
    """Downsampled whole-slide RGB via the tiff pyramid + full-res (Wf, Hf)."""
    import tifffile
    with tifffile.TiffFile(str(ds.he_out)) as tf:
        s0 = tf.series[0]
        levels = list(s0.levels) if getattr(s0, "levels", None) else [s0]
        Hf, Wf = levels[0].shape[:2]
        pick = levels[-1]
        for lv in levels:                       # finest level that fits max_dim
            h, w = lv.shape[:2]
            if max(h, w) <= max_dim:
                pick = lv
                break
        arr = pick.asarray()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr, (Wf, Hf)


def _slide_overview(ds, he, wh, xy, cols, legend, title, path):
    """Faded whole-slide H&E + all cells as coloured points + spot circles."""
    Wf, Hf = wh
    adata = ad.read_h5ad(ds.adata_out)
    sxy = np.asarray(adata.obsm["spatial"], dtype=float)
    rad = adata.uns["spatial"][ds.adata_name]["scalefactors"]["spot_diameter_fullres"] / 2
    long_in = 16.0
    figsize = (long_in * Wf / Hf, long_in) if Hf >= Wf else (long_in, long_in * Hf / Wf)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(he, extent=[0, Wf, Hf, 0], origin="upper", alpha=0.55)
    ax.scatter(xy[:, 0], xy[:, 1], s=1.2, c=cols, linewidths=0, rasterized=True)
    for sx, sy in sxy:
        ax.add_patch(Circle((sx, sy), rad, fill=False, ec="black", lw=0.4, alpha=0.55))
    ax.set_xlim(0, Wf); ax.set_ylim(Hf, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="k", label=l) for l, c in legend],
              loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return len(xy), len(sxy)


# ---------------------------------------------------------------------------
# segmentation overlays
# ---------------------------------------------------------------------------
def _seg_plot(he, cells, colormap, spots, rad, legend, title, path):
    """``cells`` = [(id, entry, contour_xy)]; ``colormap`` maps entry->RGB."""
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(he, origin="upper")
    polys, facel, edgel = [], [], []
    for _cid, e, c in cells:
        col = colormap(e)
        polys.append(c)
        facel.append((*col, 0.5))
        edgel.append((*col, 0.9))
    if polys:
        ax.add_collection(PolyCollection(polys, facecolors=facel,
                                         edgecolors=edgel, linewidths=0.25))
    for _, sx, sy in spots:
        ax.add_patch(Circle((sx, sy), rad, fill=False, ec="black", lw=1.1, alpha=0.65))
    ax.set_xlim(0, he.shape[1]); ax.set_ylim(he.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="k", label=l) for l, c in legend],
              loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _prop_plot(he, spots, rad, props, cats, pal, title, path):
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(he, origin="upper", alpha=0.55)
    pr = rad * 0.9
    for sid, sx, sy in spots:
        if sid not in props.index:
            continue
        fr = props.loc[sid, cats].to_numpy(dtype=float)
        if fr.sum() <= 0:
            continue
        ang = 90.0
        for c, f in zip(cats, fr):
            if f <= 0:
                continue
            ext = 360.0 * f
            ax.add_patch(Wedge((sx, sy), pr, ang, ang + ext,
                               facecolor=pal[c], edgecolor="white", lw=0.3))
            ang += ext
        ax.add_patch(Circle((sx, sy), pr, fill=False, ec="black", lw=0.6, alpha=0.5))
    ax.set_xlim(0, he.shape[1]); ax.set_ylim(he.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11)
    ax.legend(handles=[Patch(facecolor=pal[c], edgecolor="k", label=c) for c in cats],
              loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# level hierarchy diagram
# ---------------------------------------------------------------------------
def _tree_diagram(bridge, name, path):
    levels = bench_hier.HIER[name]
    nlev = len(levels)
    lc = colors.leaf_colors(bridge, name)
    maxrows = max(len(lv) for lv in levels)
    dx = 3.4
    box_w, box_h = 1.7, 0.66
    fig_h = max(4.0, maxrows * 0.8)
    fig_w = max(6.0, nlev * dx)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    pos: list[dict] = []
    for li, lv in enumerate(levels):
        cats = list(lv.keys())
        x = li * dx
        ys = np.linspace(maxrows - 1, 0, len(cats)) if len(cats) > 1 else [(maxrows - 1) / 2]
        p = {}
        for cat, y in zip(cats, ys):
            cols = [lc[m] for m in lv[cat]]
            col = tuple(sum(c[k] for c in cols) / len(cols) for k in range(3))
            ax.add_patch(Rectangle((x - box_w / 2, y - box_h / 2), box_w, box_h,
                                    facecolor=(*col, 0.85), edgecolor="black", lw=1.0))
            fs = 8 if len(cat) <= 14 else (7 if len(cat) <= 20 else 6)
            ax.text(x, y, cat, ha="center", va="center", fontsize=fs)
            p[cat] = (x, y)
        pos.append(p)
        tag = ""
        if li == bench_hier.ORIGINAL_LEVEL:
            tag = "\n(= original)"
        elif li == bench_hier.label2_level(name):
            tag = "\n(= label2)"
        ax.text(x, maxrows + 0.1, f"level{li}{tag}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    for li in range(nlev - 1):
        coarse, fine = levels[li], levels[li + 1]
        for c_cat, c_members in fine.items():
            cset = set(c_members)
            for p_cat, p_members in coarse.items():
                if cset <= set(p_members):
                    (x0, y0), (x1, y1) = pos[li][p_cat], pos[li + 1][c_cat]
                    ax.add_patch(FancyArrow(
                        x0 + box_w / 2, y0, (x1 - box_w / 2) - (x0 + box_w / 2), y1 - y0,
                        width=0.002, head_width=0.10, head_length=0.12,
                        length_includes_head=True, color="0.45", alpha=0.8))
                    break
    ax.set_xlim(-box_w, (nlev - 1) * dx + box_w); ax.set_ylim(-1, maxrows + 1.0)
    ax.axis("off")
    ax.set_title(f"{name} -- {nlev} annotation levels", fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def make_all_plots(layout: Layout, bridge: Bridge, name: str) -> dict:
    ds = paths.dataset(layout, name)
    ds.plots_dir.mkdir(parents=True, exist_ok=True)
    window = pick_window(ds)
    he = read_he(ds, window)
    spots, rad = _spots_in_window(ds, window)

    # 00 -- whole-slide overviews: every cell a point + spot circles.
    # HoVerNet (raw class) and DAPI (finest-level/label2) are different cell sets,
    # so abundances/locations differ -- that is expected and fine.
    _, hn_nuc = hnio.load_nuc(ds.hovernet_json)
    he_ov, wh = _read_he_overview(ds)
    hn_ids, hn_xy = hnio.centroids(hn_nuc)
    hn_present = sorted({int(hn_nuc[i]["type"]) for i in hn_ids})
    n_hn, n_spots = _slide_overview(
        ds, he_ov, wh, hn_xy,
        [colors.HOVERNET_TYPE_COLORS[int(hn_nuc[i]["type"])] for i in hn_ids],
        [(f"{t} {hnio.HOVERNET_TYPE_NAMES.get(t, t)}", colors.HOVERNET_TYPE_COLORS[t])
         for t in hn_present],
        f"{name}: {len(hn_ids)} HoVerNet cells (raw types)",
        ds.plots_dir / "00_slide_overview_hovernet.png")

    lastlv = bench_hier.label2_level(name)
    _, dp_all = hnio.load_nuc(ds.level_dir(lastlv) / "dapi_seg.json")
    lc = colors.leaf_colors(bridge, name)
    dp_ids, dp_xy = hnio.centroids(dp_all)
    n_dp, _ = _slide_overview(
        ds, he_ov, wh, dp_xy,
        [lc[dp_all[i]["type_name"]] for i in dp_ids],
        [(l, lc[l]) for l in bench_hier.leaves(name)],
        f"{name}: {len(dp_ids)} DAPI cells (label2)",
        ds.plots_dir / "00_slide_overview_dapi.png")

    # 00 -- raw HoVerNet-type segmentation (2000x2000 crop)
    hn_cells = _cells_in_window(hn_nuc, window)
    hn_cells = _cells_in_window(hn_nuc, window)
    present = sorted({int(e["type"]) for _, e, _ in hn_cells})
    hn_legend = [(f"{t} {hnio.HOVERNET_TYPE_NAMES.get(t, t)}", colors.HOVERNET_TYPE_COLORS[t])
                 for t in present]
    _seg_plot(he, hn_cells, lambda e: colors.HOVERNET_TYPE_COLORS[int(e["type"])],
              spots, rad, hn_legend, f"{name} | HoVerNet raw types",
              ds.plots_dir / "00_hovernet_seg.png")

    # 01 / 02 -- per-level DAPI seg + proportion pies
    for lv in range(bench_hier.n_levels(name)):
        cats = bench_hier.categories(name, lv)
        pal = colors.level_palette(bridge, name, lv)
        _, dp_nuc = hnio.load_nuc(ds.level_dir(lv) / "dapi_seg.json")
        dp_cells = _cells_in_window(dp_nuc, window)
        _seg_plot(he, dp_cells, lambda e: pal[e["type_name"]], spots, rad,
                  [(c, pal[c]) for c in cats], f"{name} DAPI seg | level{lv}",
                  ds.plots_dir / f"01_dapi_seg_level{lv}.png")
        props = pd.read_csv(ds.level_dir(lv) / "proportions.csv", index_col="spot_id")
        _prop_plot(he, spots, rad, props, cats, pal,
                   f"{name} proportions | level{lv}",
                   ds.plots_dir / f"02_prop_level{lv}.png")

    _tree_diagram(bridge, name, ds.plots_dir / "03_levels_tree.png")
    return {"dataset": name, "window": window,
            "n_hovernet_cells": n_hn, "n_dapi_cells": n_dp, "n_spots_total": n_spots,
            "n_hovernet_cells_window": len(hn_cells), "n_spots_window": len(spots),
            "n_levels": bench_hier.n_levels(name)}
