"""Left-to-right diagram of an annotation hierarchy.

One column per level (coarse on the left). Finest-level categories are spaced
evenly top to bottom; every coarser category sits at the mean height of its
children, so the diagram reads as a tree. Boxes are filled with the category
colour and show the cell count; arrows go from parent to child. Categories listed
in ``hatched`` (e.g. unassigned ones) get a hatch pattern so they do not rely on
colour alone.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from . import colors as _colors

BOX_W, BOX_H, DX, DY = 2.3, 0.64, 3.4, 0.82


def draw(levels: list[list[str]], parent: list[dict[str, str]], colors: dict,
         counts: dict, headers: list[str], title: str, path,
         hatched: set[tuple[int, str]] = frozenset()) -> None:
    last = len(levels) - 1
    y: dict[tuple[int, str], float] = {}
    n_leaf = len(levels[last])
    for i, name in enumerate(levels[last]):
        y[(last, name)] = (n_leaf - 1 - i) * DY
    for level in range(last - 1, -1, -1):
        for name in levels[level]:
            kids = [y[(level + 1, c)] for c in levels[level + 1] if parent[level + 1][c] == name]
            y[(level, name)] = float(np.mean(kids)) if kids else 0.0

    top = (n_leaf - 1) * DY
    fig, ax = plt.subplots(figsize=(len(levels) * DX * 0.95 + 0.8, max(3.2, n_leaf * DY * 0.78 + 1.6)))
    for level, names in enumerate(levels):
        x = level * DX
        ax.text(x, top + 1.0, headers[level], ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        for name in names:
            rgb = colors[(level, name)]
            yy = y[(level, name)]
            shape = dict(boxstyle="round,pad=0.02,rounding_size=0.12")
            ax.add_patch(FancyBboxPatch((x - BOX_W / 2, yy - BOX_H / 2), BOX_W, BOX_H,
                                        facecolor=rgb, edgecolor="#333333", lw=0.9, **shape))
            is_hatched = (level, name) in hatched
            if is_hatched:   # light hatch on top, outline drawn by the patch above
                ax.add_patch(FancyBboxPatch((x - BOX_W / 2, yy - BOX_H / 2), BOX_W, BOX_H,
                                            fill=False, hatch="////", edgecolor="white",
                                            lw=0, alpha=0.7, **shape))
            tc = _colors.text_color(rgb)
            backing = dict(boxstyle="round,pad=0.12", facecolor=rgb, edgecolor="none") if is_hatched else None
            ax.text(x, yy + 0.09, name, ha="center", va="center", fontsize=8.2 if len(name) <= 22 else 7,
                    color=tc, fontweight="bold", bbox=backing)
            ax.text(x, yy - 0.17, f"n = {counts.get((level, name), 0):,}", ha="center", va="center",
                    fontsize=7, color=tc, bbox=backing)
            if level > 0:
                px, py = (level - 1) * DX + BOX_W / 2, y[(level - 1, parent[level][name])]
                ax.annotate("", xy=(x - BOX_W / 2, yy), xytext=(px, py),
                            arrowprops=dict(arrowstyle="-|>", color="#8a8a8a", lw=0.9,
                                            shrinkA=0, shrinkB=1, mutation_scale=9))
    ax.set_xlim(-BOX_W / 2 - 0.3, last * DX + BOX_W / 2 + 0.3)
    ax.set_ylim(-BOX_H, top + 1.9)
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=4)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
