"""Verification figures, in ``<out-dir>/plots/``.

00_levels_tree.png            annotation hierarchy, with cell counts
01_overview_hovernet.png      whole slide: every nucleus by HoVerNet class, spots
01_overview_lv<k>.png         whole slide: every nucleus by level-k type, spots
02_window_hovernet.png        densest 2000 px window: nuclear outlines by class
02_window_lv<k>.png           same window: outlines by level-k type, spots
03_props_lv<k>.png            same window: each spot as a pie of its proportions
04_dominant_lv<k>.png         whole slide: each spot filled with its dominant type
05_gallery_lv<k>.png          random nucleus crops (20 um) per level-k type
06_spot_qc.png                nuclei, bins and counts per spot
07_matching.png               matching checks against their controls + marker scores

Colours come from ``metadata.json``. Every figure names its types in a legend.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import EllipseCollection, PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Patch, Wedge

from . import hovernet, slide, tree

from .config import SEED, Sample, log, read_metadata
from .labels import counts as label_counts
from .spots import hierarchy_from_json, read_cell_table

WINDOW = 2000
OVERVIEW_MAX_DIM = 3000
GALLERY_COLS = 16
GALLERY_UM = 20.0
GALLERY_PX = 64
INK, MUTED = "#1f1f1f", "#5a5a5a"


def _palette(seg: dict, level: int) -> dict[str, tuple]:
    return {name: to_rgb(hexc) for key, hexc in seg["palette"].items()
            for lv, name in [key.split("::", 1)] if int(lv[2:]) == level}


def _legend(ax, items, title=None, ncol=1):
    handles = [Patch(facecolor=c, edgecolor="#333333", label=l) for l, c in items]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
              frameon=False, title=title, title_fontsize=8.5, ncol=ncol)


def _dense_window(xy: np.ndarray, W: int, H: int, size: int = WINDOW) -> tuple[int, int]:
    g = 250
    nx, ny = int(np.ceil(W / g)), int(np.ceil(H / g))
    cnt = np.zeros((ny, nx))
    np.add.at(cnt, (np.clip((xy[:, 1] // g).astype(int), 0, ny - 1),
                    np.clip((xy[:, 0] // g).astype(int), 0, nx - 1)), 1)
    integ = np.pad(cnt.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    k = size // g
    best, bij = -1.0, (0, 0)
    for i in range(max(1, ny - k + 1)):
        for j in range(max(1, nx - k + 1)):
            v = integ[i + k, j + k] - integ[i, j + k] - integ[i + k, j] + integ[i, j]
            if v > best:
                best, bij = v, (i, j)
    return min(bij[1] * g, max(0, W - size)), min(bij[0] * g, max(0, H - size))


def _overview(bg, scale, xy, cols, spots_xy, radius, legend, title, path, spot_fill=None):
    Hs, Ws = bg.shape[:2]
    W, H = Ws / scale, Hs / scale
    fig, ax = plt.subplots(figsize=(14 * W / max(W, H) + 3, 14 * H / max(W, H)))
    ax.imshow(bg, extent=[0, W, H, 0], alpha=0.45 if spot_fill is None else 0.35)
    if xy is not None:
        ax.scatter(xy[:, 0], xy[:, 1], s=0.6, c=cols, linewidths=0, rasterized=True)
    d = np.full(len(spots_xy), 2 * radius)
    if spot_fill is None:
        ax.add_collection(EllipseCollection(d, d, np.zeros(len(d)), units="xy", offsets=spots_xy,
                                            offset_transform=ax.transData, facecolors="none",
                                            edgecolors="black", linewidths=0.35, alpha=0.6))
    else:
        ax.add_collection(EllipseCollection(d, d, np.zeros(len(d)), units="xy", offsets=spots_xy,
                                            offset_transform=ax.transData, facecolors=spot_fill,
                                            edgecolors="none"))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12, color=INK)
    _legend(ax, legend)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _window(img, x0, y0, polys, cols, spots, radius, legend, title, path):
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(img)
    if polys:
        ax.add_collection(PolyCollection([p - [x0, y0] for p in polys],
                                         facecolors=[(*c, 0.45) for c in cols],
                                         edgecolors=[(*c, 0.95) for c in cols], linewidths=0.4))
    for sx, sy in spots:
        ax.add_patch(Circle((sx - x0, sy - y0), radius, fill=False, ec="black", lw=1.0, alpha=0.7))
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, color=INK)
    _legend(ax, legend)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _pies(img, x0, y0, spot_ids, spots, radius, props, cats, pal, title, path):
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(img, alpha=0.55)
    r = radius * 0.95
    for sid, (sx, sy) in zip(spot_ids, spots):
        fr = props.loc[sid, cats].to_numpy(dtype=float)
        ang = 90.0
        for c, f in zip(cats, fr):
            if f <= 0:
                continue
            ax.add_patch(Wedge((sx - x0, sy - y0), r, ang, ang + 360 * f, facecolor=pal[c],
                               edgecolor="white", lw=0.4))
            ang += 360 * f
        ax.add_patch(Circle((sx - x0, sy - y0), r, fill=False, ec="black", lw=0.6, alpha=0.6))
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, color=INK)
    _legend(ax, [(c, pal[c]) for c in cats])
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _gallery(sl, cells, col, cats, pal, mpp, title, path):
    rng = np.random.default_rng(SEED)
    size_px = int(round(GALLERY_UM / mpp))
    rows = []
    # Sampled from a spatial order, not the table's own. The seed makes the *positions*
    # chosen deterministic, but the rows those positions name follow the cell numbering,
    # which bin2cell reassigns on every run (only 1.0% of P5's ids and 0.09% of P2's came
    # back the same), so two builds of the identical dataset drew different nuclei here --
    # 49-64% of the pixels apart -- while agreeing on every cell type. Sorting by centroid
    # first makes the choice a property of the cells rather than of their labels.
    cells = cells.sort_values(["y", "x"], kind="stable")
    for c in cats:
        sub = cells[cells[col] == c]
        n = min(GALLERY_COLS, len(sub))
        pick = sub.iloc[np.sort(rng.choice(len(sub), n, replace=False))] if n else sub
        rows.append((c, len(sub), slide.crops(sl, pick[["x", "y"]].to_numpy(), size_px, GALLERY_PX)))
    gap, gut = 2, 230
    cw = GALLERY_PX + gap
    Wc, Hc = GALLERY_COLS * cw, len(rows) * cw
    canvas = np.full((Hc, Wc, 3), 255, np.uint8)
    for i, (_, _, crops) in enumerate(rows):
        for j, im in enumerate(crops):
            canvas[i * cw:i * cw + GALLERY_PX, j * cw:j * cw + GALLERY_PX] = im
    fig = plt.figure(figsize=((Wc + gut) / 100, Hc / 100 + 0.7), dpi=100)
    ax = fig.add_axes([gut / (Wc + gut), 0, Wc / (Wc + gut), Hc / (Hc + 70)])
    ax.imshow(canvas, interpolation="nearest"); ax.axis("off")
    for i, (c, n, _) in enumerate(rows):
        y = (Hc - (i + 0.5) * cw) / (Hc + 70)      # image occupies the bottom Hc of Hc + 70 px
        fig.add_artist(plt.Rectangle((0.01, y - 0.012), 0.012, 0.024, transform=fig.transFigure,
                                     facecolor=pal[c], edgecolor="#333333", lw=0.5))
        fig.text(0.03, y + 0.004, c, fontsize=8.5, color=INK, va="center", fontweight="bold")
        fig.text(0.03, y - 0.011, f"{n:,} nuclei", fontsize=7, color=MUTED, va="center")
    fig.suptitle(title, fontsize=11, color=INK, y=0.999)
    fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)


def _spot_qc(pv, bg, scale, radius, title, path):
    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2.3], hspace=0.22, wspace=0.25, top=0.93, bottom=0.02)
    for i, (key, lab, kind) in enumerate([("n_cells", "annotated nuclei per spot", "int"),
                                          ("n_bins", "2 um bins per spot", "int10"),
                                          ("total_counts", "destriped counts per spot", "log")]):
        ax = fig.add_subplot(gs[0, i])
        v = pv.obs[key].to_numpy(dtype=float)
        if kind == "int":            # one bar per integer, no empty-bin combs
            bins = np.arange(-0.5, v.max() + 1.5, max(1, int(np.ceil(v.max() / 80))))
        elif kind == "int10":
            bins = np.arange(0, v.max() + 11, 10)
        else:
            bins = np.logspace(np.log10(max(v.min(), 1)), np.log10(v.max()), 50)
        ax.hist(v, bins=bins, color="#3b6fb6", edgecolor="white", lw=0.3)
        if kind == "log":
            ax.set_xscale("log")
        ax.axvline(np.median(v), color=INK, lw=1, ls="--")
        ax.set_title(f"{lab}\nmedian {np.median(v):,.0f}", fontsize=10, color=INK)
        ax.set_ylabel("spots", fontsize=9, color=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8, colors=MUTED)
    ax = fig.add_subplot(gs[1, :])
    Hs, Ws = bg.shape[:2]
    ax.imshow(bg, extent=[0, Ws / scale, Hs / scale, 0], alpha=0.3)
    xy = pv.obsm["spatial"]
    d = np.full(len(xy), 2 * radius)
    col = EllipseCollection(d, d, np.zeros(len(d)), units="xy", offsets=xy, offset_transform=ax.transData,
                            array=np.log10(pv.obs["total_counts"].to_numpy() + 1), cmap="viridis")
    ax.add_collection(col)
    ax.set_xlim(0, Ws / scale); ax.set_ylim(Hs / scale, 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_anchor("C")
    cb = fig.colorbar(col, cax=ax.inset_axes([1.015, 0.15, 0.018, 0.7]))
    cb.set_label("log10(destriped counts + 1)", fontsize=9)
    ax.set_title("destriped counts per spot", fontsize=10, color=INK)
    fig.suptitle(title, fontsize=12, color=INK, y=0.975)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _matching(seg, title, path):
    checks = {c["id"]: c for c in seg["checks"]}
    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.5)
    ax = fig.add_subplot(gs[0, 0])
    rows = [("C3", "label under nucleus = cell id"), ("C4", "nucleus inside its own cell"),
            ("C5", "cell counts = bin2cell counts for that id")]
    y = np.arange(len(rows))[::-1]
    for yy, (cid, lab) in zip(y, rows):
        c = checks[cid]
        ax.barh(yy + 0.18, c["value"], height=0.34, color="#3b6fb6")
        ax.barh(yy - 0.18, c["control"], height=0.34, color="#b8b8b8")
        ax.text(c["value"] + 0.01, yy + 0.18, f"{100 * c['value']:.2f}%", va="center", fontsize=9, color=INK)
        ax.text(c["control"] + 0.01, yy - 0.18, f"{100 * c['control']:.2f}%", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(y, [f"{cid}  {lab}" for cid, lab in rows], fontsize=9)
    ax.set_xlim(0, 1.22); ax.set_xlabel("fraction of cells", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(color="#3b6fb6", label="true ids"),
                       Patch(color="#b8b8b8", label="control: id + 1")],
              loc="lower right", fontsize=8, frameon=False)
    lines = [f"{'PASS' if c['passed'] else ('FAIL' if c['hard'] else 'WARN')}  {c['id']}  {c['name']}"
             + ("" if c["hard"] else "  (reported, not a stop)") for c in seg["checks"]]
    ax.text(0.0, -0.22, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=7.5,
            family="monospace", color=INK)
    ax.set_title("nucleus <-> cell matching", fontsize=11, color=INK)

    ax = fig.add_subplot(gs[0, 1])
    ms = seg.get("marker_scores") or {}
    if ms:
        M = np.asarray(ms["matrix"])
        lim = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(ms["types"])), ms["types"], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(ms["types"])), ms["types"], fontsize=8)
        for i in range(len(ms["types"])):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
        ax.set_xlabel("marker gene set", fontsize=9); ax.set_ylabel("annotated type", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="z-scored mean expression")
        c6 = checks.get("C6", {})
        ax.set_title(f"marker check (own set highest: {c6.get('value')}; shuffled labels: "
                     f"{str(c6.get('control', '')).split(' ')[0]})", fontsize=10, color=INK)
    else:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def run(s: Sample) -> None:
    pdir = s.out_dir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    seg = read_metadata(s.out_dir)["seg"]
    h = hierarchy_from_json(seg["hierarchy"])
    mpp = seg["mpp"]["used"]
    cells = read_cell_table(s.out_dir)
    nuc = hovernet.load(s.out_dir / "hovernet.json")["nuc"]
    pv = ad.read_h5ad(s.out_dir / "pseudovisium.h5ad")
    radius = s.spot_diameter_um / mpp / 2
    xy = cells[["x", "y"]].to_numpy(dtype=float)
    spot_xy = pv.obsm["spatial"]
    tag = s.name
    log(f"plots: {len(cells):,} nuclei, {pv.n_obs:,} spots")

    pal = {k: _palette(seg, k) for k in range(h.n_levels)}
    cols = {(k, n): pal[k][n] for k in range(h.n_levels) for n in h.levels[k]}
    labs = cells[[f"lv{k}" for k in range(h.n_levels)]].rename(
        columns={f"lv{k}": c for k, c in enumerate(h.columns)})
    tree.draw(h.levels, h.parent, cols, label_counts(h, labs),
              [f"lv{k}\n{c} · {len(h.levels[k])} types" for k, c in enumerate(h.columns)],
              f"{tag} · annotation levels", pdir / "00_levels_tree.png", hatched=h.unassigned)

    with slide.open_slide(s.pyr_tif) as sl:
        W, H = sl.dimensions
        bg, scale = slide.read_level(sl, slide.level_for(sl, OVERVIEW_MAX_DIM), s.pyr_tif)
        hn_type = cells["hovernet_type"].to_numpy(dtype=int)
        present = sorted(set(hn_type))
        _overview(bg, scale, xy, [hovernet.CLASS_COLORS[t] for t in hn_type], spot_xy, radius,
                  [(f"{t} {hovernet.CLASS_NAMES[t]}", hovernet.CLASS_COLORS[t]) for t in present],
                  f"{tag} · {len(cells):,} nuclei by HoVerNet class", pdir / "01_overview_hovernet.png")
        for k in range(h.n_levels):
            lab = cells[f"lv{k}"].to_numpy()
            _overview(bg, scale, xy, [pal[k][v] for v in lab], spot_xy, radius,
                      [(n, pal[k][n]) for n in h.levels[k]],
                      f"{tag} · nuclei by lv{k} ({h.columns[k]})", pdir / f"01_overview_lv{k}.png")

        x0, y0 = _dense_window(xy, W, H)
        img = slide.read_window(sl, x0, y0, WINDOW, WINDOW)
        win = (xy[:, 0] >= x0) & (xy[:, 0] < x0 + WINDOW) & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + WINDOW)
        wcells = cells[win]
        polys = [hovernet.contour_xy(nuc[c]).astype(float) for c in wcells["cell_id"]]
        sw = ((spot_xy[:, 0] > x0 - radius) & (spot_xy[:, 0] < x0 + WINDOW + radius)
              & (spot_xy[:, 1] > y0 - radius) & (spot_xy[:, 1] < y0 + WINDOW + radius))
        wspots, wids = spot_xy[sw], pv.obs_names[sw]
        wt = wcells["hovernet_type"].to_numpy(dtype=int)
        _window(img, x0, y0, polys, [hovernet.CLASS_COLORS[t] for t in wt], wspots, radius,
                [(f"{t} {hovernet.CLASS_NAMES[t]}", hovernet.CLASS_COLORS[t]) for t in sorted(set(wt))],
                f"{tag} · window x={x0}, y={y0} · HoVerNet class", pdir / "02_window_hovernet.png")
        for k in range(h.n_levels):
            lab = wcells[f"lv{k}"].to_numpy()
            _window(img, x0, y0, polys, [pal[k][v] for v in lab], wspots, radius,
                    [(n, pal[k][n]) for n in h.levels[k] if n in set(lab)],
                    f"{tag} · window x={x0}, y={y0} · lv{k} ({h.columns[k]})", pdir / f"02_window_lv{k}.png")
            props = pd.read_csv(s.out_dir / "sim" / f"proportions_lv{k}.csv", index_col="spot_id")
            _pies(img, x0, y0, wids, wspots, radius, props, h.levels[k], pal[k],
                  f"{tag} · spot proportions · lv{k} ({h.columns[k]})", pdir / f"03_props_lv{k}.png")
            dom = pv.obs[f"dominant_lv{k}"].astype(str).to_numpy()
            _overview(bg, scale, None, None, spot_xy, radius,
                      [(n, pal[k][n]) for n in h.levels[k]],
                      f"{tag} · dominant type per spot · lv{k} ({h.columns[k]})",
                      pdir / f"04_dominant_lv{k}.png", spot_fill=[pal[k][d] for d in dom])
            _gallery(sl, cells, f"lv{k}", h.levels[k], pal[k], mpp,
                     f"{tag} · nucleus crops ({GALLERY_UM:.0f} um) · lv{k} ({h.columns[k]})",
                     pdir / f"05_gallery_lv{k}.png")
    _spot_qc(pv, bg, scale, radius, f"{tag} · spot QC", pdir / "06_spot_qc.png")
    _matching(seg, f"{tag} · matching checks", pdir / "07_matching.png")
    log(f"plots: {len(list(pdir.glob('*.png')))} figures")
