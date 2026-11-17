"""Hierarchical cell-type palette: every type is a shade of its top-level family.

The hierarchy is given level by level (coarse to fine): ``levels[k]`` lists the
categories of level ``k`` and ``parent[k]`` maps each of them to its category at
level ``k - 1``. Colours are keyed by ``(level, name)`` because a name may repeat
across levels with different members (e.g. ``Stromal`` at two levels).

A category named in :data:`TABLE` takes the colour written there; only the rest are
chosen by the search described below. The table is the colour code of the wider
project (see its own comment) and covers every cell type the three Visium HD colon
samples use, so in practice the search now runs only for a sample whose annotation
brings a type the table has never seen.

How the search places the rest:

* Each top-level category is a *family* with a hue band: epithelial types blue,
  immune green, stromal orange; any other family takes a free hue, and an
  unassigned family is grey.
* The finest-level types of a family (its *leaves*) get shades inside that band.
  Shades are chosen jointly for all families by farthest-point search in OKLab,
  which maximises the smallest perceptual difference between any two leaves; the
  shades of a family are then handed out in depth-first order sorted by hue, so
  siblings get neighbouring hues.
* An unassigned leaf inside a coloured family is a pale, greyish tint of it, and
  the leaves of an unassigned family are greys. Both are fixed before the search,
  so the chosen shades keep away from them -- as are the colours the table named.
* Any coarser category the table does not name is the mean colour of its assigned
  leaves, so it reads as their parent.

Everything is deterministic.
"""
from __future__ import annotations

import colorsys
import re
from itertools import combinations

import numpy as np

RGB = tuple[float, float, float]

#: The colour code of the project, written out for the Visium HD colon cell types.
#:
#: Hue says which compartment a cell type belongs to and lightness says which member of it. The
#: load is on lightness because it is the channel that survives colour blindness: under red-green
#: blindness green, gold and red collapse onto one yellowish axis, so a family has to be a ladder
#: of lightness rather than a cluster of shades. Hue interleaving -- placing a lightness
#: neighbour in a distant hue -- was tried here and measured *worse* (10.5 against 15.3), for
#: that same reason.
#:
#:     epithelium   red for the tumour, oranges for the intestinal lineage
#:     lymphoid     one green ladder, B dark / plasma mid / T light
#:     myeloid      one yellow ladder, neutrophil dark / macrophage gold / mast pale
#:     structural   one blue ladder, stroma dark, vessels light
#:     perivascular pink, smooth muscle alone
#:     neural       purple, enteric neurons alone
#:
#: A type that exists elsewhere in the project keeps the colour it has there, so a cell type is
#: one colour across the benchmark, the two case studies, the Xenium simulations and these.
#: Smooth muscle is the one deliberate exception: it takes a darker rung of the pink chosen for
#: breast PVL (the same lineage) rather than that pink itself, because `#D70475` and the
#: neutrophil yellow measure 4.3 apart under protanopia -- effectively the same colour.
#:
#: Contrast only has to hold inside one figure, and a figure is one (sample, level), so what is
#: measured is the worst pair of each of the nine vocabularies, not of the table as a whole.
#: That worst pair is **16.0** (tumour epithelial / neutrophil), in CIELAB over normal,
#: deuteranopic and protanopic vision; the palette this replaces measured 6.4 to 10.8 on the
#: three lv2 figures. Two colon types never share a figure and so share a rung: Paneth cell (P1)
#: and Intestinal Epithelial Other (P5) are each their sample's *second* intestinal type.
#: Keyed by name alone, not by ``(level, name)``: a name means the same thing at every level it
#: appears at, and a parent deliberately carries the colour of its main child, so that ``Stromal``
#: at level 1 and ``Fibroblast`` at level 2 are one blue. No two of those ever share a figure.
TABLE = {
    # --- epithelium: red for the tumour lineage, oranges for the intestinal one
    "Epithelial": "#A33A2B",                   # level 0, carried by the tumour
    "Tumor Epithelial": "#A33A2B",
    "Tumor Epithelial Inflamed": "#701C33",    # the dark rung of the tumour red
    "Intestinal Epithelial": "#FF9429",        # carried by Goblet, its main type
    "Goblet": "#FF9429",
    "Paneth cell": "#C95318",                  # P1's second intestinal type
    "Intestinal Epithelial Other": "#C95318",  # P5's second intestinal type, the same rung
    # --- immune: the green ladder for the lymphoid types, the yellow one for the myeloid
    "Immune": "#20973C", "Lymphoid": "#20973C",  # both carried by the plasma cell
    "B cell": "#024B13", "Plasma cell": "#20973C", "T cell / NK cell": "#67E49D",
    "Myeloid": "#EEBB63",                      # carried by the macrophage
    "Neutrophil": "#BA9B36", "Macrophage": "#EEBB63", "Mast cell": "#FFEDB2",
    # --- structural: one blue ladder, stroma dark and vessels light
    "Stromal": "#04429F", "Fibroblast": "#04429F",
    "Vascular": "#51ABFB", "Endothelial Vascular": "#51ABFB", "Lymphatic Vascular": "#AFC9FD",
    # --- the two compartments of their own
    "Smooth Muscle": "#AF2372",                # perivascular
    "Neuronal": "#7A2EBD",                     # neural, levels 1 and 2 alike
}

# (keyword pattern, hue centre, half band) -- hues in HLS turns
_FAMILIES = [
    (re.compile(r"epithel", re.I), 0.60, 0.07),                      # blue
    (re.compile(r"immun|lymph|myeloid|leuko", re.I), 0.33, 0.07),    # green
    (re.compile(r"strom|struct|mesench|fibro", re.I), 0.085, 0.05),  # orange
]
_FREE_HUES = [0.83, 0.50, 0.95, 0.17, 0.45, 0.25, 0.70]
_FREE_HALF_BAND = 0.05
_L_RANGE, _S_RANGE = (0.33, 0.72), (0.50, 0.85)
_BASE_L, _BASE_S = 0.50, 0.70


def _oklab(rgb: RGB) -> np.ndarray:
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return 100.0 * np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                             1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                             0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s])


def _hue_offset(h: float, centre: float) -> float:
    return (h - centre + 0.5) % 1.0 - 0.5


def _family_bands(families: list[str], grey: set[str]) -> dict[str, tuple[float, float] | None]:
    bands: dict[str, tuple[float, float] | None] = {}
    used: list[float] = []
    for fam in families:
        if fam in grey:
            bands[fam] = None
            continue
        for pat, hue, half in _FAMILIES:
            if pat.search(fam) and hue not in used:
                bands[fam] = (hue, half)
                used.append(hue)
                break
    for fam in families:
        if fam in bands:
            continue
        hue = next(h for h in _FREE_HUES if all(abs(_hue_offset(h, u)) > 0.12 for u in used))
        bands[fam] = (hue, _FREE_HALF_BAND)
        used.append(hue)
    return bands


def _candidates(hue: float, half: float) -> tuple[list[tuple], np.ndarray]:
    hls = [((hue + dh) % 1.0, L, S)
           for dh in np.linspace(-half, half, 11)
           for L in np.linspace(*_L_RANGE, 12)
           for S in np.linspace(*_S_RANGE, 4)]
    lab = np.array([_oklab(colorsys.hls_to_rgb(*c)) for c in hls])
    return hls, lab


def palette(levels: list[list[str]], parent: list[dict[str, str]],
            unassigned: set[tuple[int, str]] = frozenset(),
            table: dict[str, str] | None = None) -> dict[tuple[int, str], RGB]:
    """``(level, name) -> RGB`` for every category of every level.

    Args:
        levels: The categories of each level, coarse to fine.
        parent: ``parent[k][name]`` is ``name``'s category at level ``k - 1``.
        unassigned: The ``(level, name)`` pairs to draw as greys or pale tints.
        table: Name to hex colour, taking precedence over the search; :data:`TABLE` by
            default, and ``{}`` to search for everything the way this did before the
            table existed.
    """
    table = TABLE if table is None else table
    last = len(levels) - 1

    def ancestor(level: int, name: str, target: int) -> str:
        while level > target:
            name = parent[level][name]
            level -= 1
        return name

    families = list(levels[0])
    bands = _family_bands(families, {n for lv, n in unassigned if lv == 0})
    odd_leaves = {n for lv, n in unassigned if lv == last}
    leaves = {f: [l for l in levels[last] if ancestor(last, l, 0) == f] for f in families}

    leaf_rgb: dict[str, RGB] = {}
    fixed: list[np.ndarray] = []
    # fixed colours: greys for unassigned families, pale tints for unassigned leaves
    for f in families:
        normal = [l for l in leaves[f] if l not in odd_leaves]
        odd = [l for l in leaves[f] if l in odd_leaves]
        if bands[f] is None:
            for i, l in enumerate(normal + odd):
                n = len(normal) + len(odd)
                light = 0.62 if n == 1 else 0.42 + 0.34 * i / (n - 1)
                leaf_rgb[l] = colorsys.hls_to_rgb(0.0, light, 0.0)
        else:
            for j, l in enumerate(odd):
                leaf_rgb[l] = colorsys.hls_to_rgb(bands[f][0], 0.74 - 0.08 * j, 0.16)
    # the named leaves, which the search then has to keep away from like any other fixed colour
    named = {l for l in levels[last] if l in table and l not in odd_leaves}
    for l in named:
        leaf_rgb[l] = from_hex(table[l])
    fixed = [_oklab(c) for c in leaf_rgb.values()]

    # joint farthest-point search for the assigned leaves of coloured families the table misses
    todo = {f: [l for l in leaves[f] if l not in odd_leaves and l not in named]
            for f in families if bands[f] is not None}
    todo = {f: ls for f, ls in todo.items() if ls}
    cand = {f: _candidates(*bands[f]) for f in todo}
    picked: dict[str, list[int]] = {f: [] for f in todo}
    chosen: list[np.ndarray] = list(fixed)
    for f in todo:                                   # family base colour first
        hls, lab = cand[f]
        j = int(np.argmin([abs(_hue_offset(h, bands[f][0])) + abs(L - _BASE_L) + abs(S - _BASE_S)
                           for h, L, S in hls]))
        picked[f].append(j)
        chosen.append(lab[j])
    while any(len(picked[f]) < len(todo[f]) for f in todo):
        for f in todo:
            if len(picked[f]) >= len(todo[f]):
                continue
            hls, lab = cand[f]
            dist = np.min(np.linalg.norm(lab[:, None, :] - np.asarray(chosen)[None], axis=2), axis=1)
            dist[picked[f]] = -1.0
            j = int(np.argmax(dist))
            picked[f].append(j)
            chosen.append(lab[j])
    for f in todo:
        hls, _ = cand[f]
        order = sorted(picked[f], key=lambda j: _hue_offset(hls[j][0], bands[f][0]))
        for leaf, j in zip(todo[f], order):
            leaf_rgb[leaf] = colorsys.hls_to_rgb(*hls[j])

    colors: dict[tuple[int, str], RGB] = {(last, l): leaf_rgb[l] for l in levels[last]}
    for level in range(last - 1, -1, -1):
        for name in levels[level]:
            if name in table and (level, name) not in unassigned:
                colors[(level, name)] = from_hex(table[name])
                continue
            members = [l for l in levels[last] if ancestor(last, l, level) == name]
            keep = [l for l in members if l not in odd_leaves] or members
            cols = np.array([leaf_rgb[l] for l in keep])
            colors[(level, name)] = tuple(float(v) for v in cols.mean(axis=0))
    return colors


def to_hex(rgb: RGB) -> str:
    return "#%02x%02x%02x" % tuple(int(round(255 * min(max(v, 0.0), 1.0))) for v in rgb)


def from_hex(hexc: str) -> RGB:
    h = hexc.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def text_color(rgb: RGB) -> str:
    """Black or white, whichever reads better on ``rgb``."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "black" if lum > 0.28 else "white"


def min_difference(colors: dict) -> tuple[float, tuple]:
    """Smallest perceptual difference (OKLab distance x100) between two colours."""
    best = (np.inf, ())
    for (ka, a), (kb, b) in combinations(colors.items(), 2):
        d = float(np.linalg.norm(_oklab(a) - _oklab(b)))
        if d < best[0]:
            best = (d, (ka, kb))
    return best
