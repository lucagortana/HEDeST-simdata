"""Flat multi-level cell-type hierarchies per dataset (from ``hierarchy.md``).

Unlike :mod:`hedest_bench.hier` there is **no tree strategy** here (no compact /
open): every dataset just has a sequence of annotation *levels*, coarse -> fine,
so we can benchmark at several cell-type resolutions. Levels are **0-indexed**
(``level0`` == hierarchy.md "Level 1"). Two levels are anchored to the existing
STHELAR annotations:

* ``ORIGINAL_LEVEL`` (index 2, hierarchy.md "Level 3") == the coarse *original*
  annotation (``proportions_original.csv`` column set);
* the **last** level == the fine *label2* annotation (``proportions_label2.csv``).

Each dataset is encoded twice, redundantly, so the two cross-check each other at
import time:

* ``LEVEL_ORDER[name]``  -- per level, the ordered category names exactly as
  hierarchy.md lists them (this fixes the integer cell-type code / column order);
* ``LEAF_PATH[name]``    -- per finest-grained ``label2`` leaf, the category it
  belongs to at every level (``[catL0, catL1, ...]``).

From these we build ``HIER[name]`` = ``[level0, level1, ...]`` with each level an
ordered ``{category: [leaf, ...]}`` mapping, and validate: every level is a
partition of the same leaf set, consecutive levels strictly refine, the finest
level is the identity, and each level's category set equals ``LEVEL_ORDER``.

Leaf names are the label2 names **as they appear in the data** (post per-crop
merge); hierarchy.md's ``"T/NK"`` is written ``"T_NK"`` to match.
"""
from __future__ import annotations

# hierarchy.md "Level 3" (0-based index 2) is the coarse "original" annotation;
# the last level is the fine "label2" annotation. True for every dataset.
ORIGINAL_LEVEL = 2

_LUNG_EPI = ["Lung_epithelial_cell", "Lung_alveolar_typeI_epithelial_cell",
             "Lung_alveolar_typeII_epithelial_cell", "Lung_ciliated_epithelial_cell"]

# name -> per level, ordered category names (hierarchy.md order, coarse -> fine).
LEVEL_ORDER: dict[str, list[list[str]]] = {
    "breast_s6": [
        ["Immune", "Structural", "Epithelial"],
        ["Lymphoid", "Myeloid", "Stromal", "Blood_vessel", "Epithelial"],
        ["B_Plasma", "T_NK", "Myeloid", "Fibroblast_Myofibroblast", "Blood_vessel", "Epithelial"],
        ["B_Plasma", "T_NK", "Myeloid", "CAF", "Myofibroblast", "Blood_vessel", "Epithelial"],
        ["B", "T", "Monocyte/Macrophage/DC", "Mast", "CAF", "Myofibroblast", "Endothelial", "Breast_epithelial_cell"],
    ],
    "cervix_s0_0": [
        ["Immune", "Structural", "Epithelial"],
        ["Immune", "Stromal", "Blood_vessel", "Epithelial"],
        ["T_NK", "Myeloid", "Fibroblast_Myofibroblast", "Blood_vessel", "Epithelial"],
        ["T_NK", "Myeloid", "Fibroblast", "Smooth_muscle", "Blood_vessel", "Epithelial"],
        ["T", "Monocyte/Macrophage", "Fibroblast", "Smooth_muscle", "Endothelial", "Secretory_epithelial_cell", "Squamous_epithelial_cell"],
    ],
    "cervix_s0_1": [
        ["Immune", "Structural", "Epithelial"],
        ["Immune", "Stromal", "Blood_vessel", "Epithelial"],
        ["T_NK", "Myeloid", "Fibroblast_Myofibroblast", "Blood_vessel", "Epithelial"],
        ["T", "Monocyte/Macrophage", "Fibroblast", "Smooth_muscle", "Endothelial", "Epithelial"],
    ],
    "lung_s3": [
        ["Immune", "Structural", "Epithelial"],
        ["Lymphoid", "Myeloid", "Stromal", "Blood_vessel", "Epithelial"],
        ["B_Plasma", "T_NK", "Myeloid", "Fibroblast_Myofibroblast", "Blood_vessel", "Epithelial"],
        ["B", "Plasma", "T", "Monocyte/Macrophage", "Mast", "CAF", "Smooth_muscle", "Endothelial", "Epithelial"],
        ["B", "Plasma", "T", "Monocyte/Macrophage", "Mast", "CAF", "Smooth_muscle", "Endothelial", *_LUNG_EPI],
    ],
    "lymph_node_s0": [
        ["Immune", "Structural"],
        ["Lymphoid", "Myeloid", "Stromal", "Blood_vessel"],
        ["B_Plasma", "T_NK", "Myeloid", "Fibroblast_Myofibroblast", "Blood_vessel"],
        ["B", "Plasma", "T_NK", "Myeloid", "Fibroblast", "Endothelial"],
        ["B", "Plasma", "T_NK", "Monocyte/Macrophage", "DC", "Mast", "Fibroblast", "Endothelial"],
    ],
    "ovary_s1": [
        ["Immune", "Structural", "Epithelial"],
        ["Immune", "Stromal", "Blood_vessel", "Epithelial"],
        ["T_NK", "Myeloid", "Fibroblast_Myofibroblast", "Blood_vessel", "Epithelial"],
        ["T_NK", "Myeloid", "CAF", "Smooth_muscle", "Blood_vessel", "Epithelial"],
        ["T_CD8", "Monocyte/Macrophage", "CAF", "Smooth_muscle", "Angiogenic_endothelial", "Epithelial", "Proliferating_epithelial"],
    ],
    "prostate_s0": [
        ["Immune", "Structural", "Epithelial"],
        ["Immune", "Stromal", "Blood_vessel", "Epithelial"],
        ["T_NK", "Myeloid", "Fibroblast_Myofibroblast", "Blood_vessel", "Epithelial"],
        ["T_CD8", "Monocyte/Macrophage", "Mast", "CAF", "Smooth_muscle", "Blood_vessel", "Epithelial"],
        ["T_CD8", "Monocyte/Macrophage", "Mast", "CAF", "Smooth_muscle", "Endothelial", "Pericyte", "Epithelial", "Luminal_epithelial_cell"],
    ],
    "skin_s4": [
        ["Immune", "Structural", "Melanocyte", "Epithelial"],
        ["Immune", "Stromal", "Blood_vessel", "Melanocyte", "Epithelial"],
        ["B_Plasma", "T_NK", "Myeloid", "Fibroblast_Myofibroblast", "Blood_vessel", "Melanocyte", "Epithelial"],
        ["B_Plasma", "T_NK", "Myeloid", "CAF", "Fibroblast/Myofibroblast", "Blood_vessel", "Melanocyte", "Epithelial"],
        ["Plasma", "T_CD8", "Monocyte/Macrophage", "CAF", "Fibroblast/Myofibroblast", "Endothelial", "Melanocyte", "High_proliferating_melanocyte", "Keratinocyte"],
    ],
}

# name -> {leaf: [category at level0, level1, ...]}. len == number of levels.
LEAF_PATH: dict[str, dict[str, list[str]]] = {
    "breast_s6": {
        "B":                      ["Immune", "Lymphoid", "B_Plasma", "B_Plasma", "B"],
        "T":                      ["Immune", "Lymphoid", "T_NK", "T_NK", "T"],
        "Monocyte/Macrophage/DC": ["Immune", "Myeloid", "Myeloid", "Myeloid", "Monocyte/Macrophage/DC"],
        "Mast":                   ["Immune", "Myeloid", "Myeloid", "Myeloid", "Mast"],
        "CAF":                    ["Structural", "Stromal", "Fibroblast_Myofibroblast", "CAF", "CAF"],
        "Myofibroblast":          ["Structural", "Stromal", "Fibroblast_Myofibroblast", "Myofibroblast", "Myofibroblast"],
        "Endothelial":            ["Structural", "Blood_vessel", "Blood_vessel", "Blood_vessel", "Endothelial"],
        "Breast_epithelial_cell": ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Breast_epithelial_cell"],
    },
    "cervix_s0_0": {
        "T":                        ["Immune", "Immune", "T_NK", "T_NK", "T"],
        "Monocyte/Macrophage":      ["Immune", "Immune", "Myeloid", "Myeloid", "Monocyte/Macrophage"],
        "Fibroblast":               ["Structural", "Stromal", "Fibroblast_Myofibroblast", "Fibroblast", "Fibroblast"],
        "Smooth_muscle":            ["Structural", "Blood_vessel", "Blood_vessel", "Smooth_muscle", "Smooth_muscle"],
        "Endothelial":              ["Structural", "Blood_vessel", "Blood_vessel", "Blood_vessel", "Endothelial"],
        "Secretory_epithelial_cell": ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Secretory_epithelial_cell"],
        "Squamous_epithelial_cell":  ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Squamous_epithelial_cell"],
    },
    "cervix_s0_1": {
        "T":                   ["Immune", "Immune", "T_NK", "T"],
        "Monocyte/Macrophage": ["Immune", "Immune", "Myeloid", "Monocyte/Macrophage"],
        "Fibroblast":          ["Structural", "Stromal", "Fibroblast_Myofibroblast", "Fibroblast"],
        "Smooth_muscle":       ["Structural", "Blood_vessel", "Blood_vessel", "Smooth_muscle"],
        "Endothelial":         ["Structural", "Blood_vessel", "Blood_vessel", "Endothelial"],
        "Epithelial":          ["Epithelial", "Epithelial", "Epithelial", "Epithelial"],
    },
    "lung_s3": {
        "B":                   ["Immune", "Lymphoid", "B_Plasma", "B", "B"],
        "Plasma":              ["Immune", "Lymphoid", "B_Plasma", "Plasma", "Plasma"],
        "T":                   ["Immune", "Lymphoid", "T_NK", "T", "T"],
        "Monocyte/Macrophage": ["Immune", "Myeloid", "Myeloid", "Monocyte/Macrophage", "Monocyte/Macrophage"],
        "Mast":                ["Immune", "Myeloid", "Myeloid", "Mast", "Mast"],
        "CAF":                 ["Structural", "Stromal", "Fibroblast_Myofibroblast", "CAF", "CAF"],
        "Smooth_muscle":       ["Structural", "Blood_vessel", "Blood_vessel", "Smooth_muscle", "Smooth_muscle"],
        "Endothelial":         ["Structural", "Blood_vessel", "Blood_vessel", "Endothelial", "Endothelial"],
        "Lung_epithelial_cell":                 ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Lung_epithelial_cell"],
        "Lung_alveolar_typeI_epithelial_cell":  ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Lung_alveolar_typeI_epithelial_cell"],
        "Lung_alveolar_typeII_epithelial_cell": ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Lung_alveolar_typeII_epithelial_cell"],
        "Lung_ciliated_epithelial_cell":        ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Lung_ciliated_epithelial_cell"],
    },
    "lymph_node_s0": {
        "B":                   ["Immune", "Lymphoid", "B_Plasma", "B", "B"],
        "Plasma":              ["Immune", "Lymphoid", "B_Plasma", "Plasma", "Plasma"],
        "T_NK":                ["Immune", "Lymphoid", "T_NK", "T_NK", "T_NK"],
        "Monocyte/Macrophage": ["Immune", "Myeloid", "Myeloid", "Myeloid", "Monocyte/Macrophage"],
        "DC":                  ["Immune", "Myeloid", "Myeloid", "Myeloid", "DC"],
        "Mast":                ["Immune", "Myeloid", "Myeloid", "Myeloid", "Mast"],
        "Fibroblast":          ["Structural", "Stromal", "Fibroblast_Myofibroblast", "Fibroblast", "Fibroblast"],
        "Endothelial":         ["Structural", "Blood_vessel", "Blood_vessel", "Endothelial", "Endothelial"],
    },
    "ovary_s1": {
        "T_CD8":                  ["Immune", "Immune", "T_NK", "T_NK", "T_CD8"],
        "Monocyte/Macrophage":    ["Immune", "Immune", "Myeloid", "Myeloid", "Monocyte/Macrophage"],
        "CAF":                    ["Structural", "Stromal", "Fibroblast_Myofibroblast", "CAF", "CAF"],
        "Smooth_muscle":          ["Structural", "Blood_vessel", "Blood_vessel", "Smooth_muscle", "Smooth_muscle"],
        "Angiogenic_endothelial": ["Structural", "Blood_vessel", "Blood_vessel", "Blood_vessel", "Angiogenic_endothelial"],
        "Epithelial":             ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Epithelial"],
        "Proliferating_epithelial": ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Proliferating_epithelial"],
    },
    "prostate_s0": {
        "T_CD8":                 ["Immune", "Immune", "T_NK", "T_CD8", "T_CD8"],
        "Monocyte/Macrophage":   ["Immune", "Immune", "Myeloid", "Monocyte/Macrophage", "Monocyte/Macrophage"],
        "Mast":                  ["Immune", "Immune", "Myeloid", "Mast", "Mast"],
        "CAF":                   ["Structural", "Stromal", "Fibroblast_Myofibroblast", "CAF", "CAF"],
        "Smooth_muscle":         ["Structural", "Blood_vessel", "Blood_vessel", "Smooth_muscle", "Smooth_muscle"],
        "Endothelial":           ["Structural", "Blood_vessel", "Blood_vessel", "Blood_vessel", "Endothelial"],
        "Pericyte":              ["Structural", "Blood_vessel", "Blood_vessel", "Blood_vessel", "Pericyte"],
        "Epithelial":            ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Epithelial"],
        "Luminal_epithelial_cell": ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Luminal_epithelial_cell"],
    },
    "skin_s4": {
        "Plasma":                  ["Immune", "Immune", "B_Plasma", "B_Plasma", "Plasma"],
        "T_CD8":                   ["Immune", "Immune", "T_NK", "T_NK", "T_CD8"],
        "Monocyte/Macrophage":     ["Immune", "Immune", "Myeloid", "Myeloid", "Monocyte/Macrophage"],
        "CAF":                     ["Structural", "Stromal", "Fibroblast_Myofibroblast", "CAF", "CAF"],
        "Fibroblast/Myofibroblast": ["Structural", "Stromal", "Fibroblast_Myofibroblast", "Fibroblast/Myofibroblast", "Fibroblast/Myofibroblast"],
        "Endothelial":             ["Structural", "Blood_vessel", "Blood_vessel", "Blood_vessel", "Endothelial"],
        "Melanocyte":              ["Melanocyte", "Melanocyte", "Melanocyte", "Melanocyte", "Melanocyte"],
        "High_proliferating_melanocyte": ["Melanocyte", "Melanocyte", "Melanocyte", "Melanocyte", "High_proliferating_melanocyte"],
        "Keratinocyte":            ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Keratinocyte"],
    },
}

NAMES = list(LEVEL_ORDER.keys())


def _build_hier() -> dict[str, list[dict[str, list[str]]]]:
    """LEAF_PATH + LEVEL_ORDER -> ordered ``[{category: [leaf, ...]}, ...]``."""
    hier: dict[str, list[dict[str, list[str]]]] = {}
    for name in NAMES:
        orders = LEVEL_ORDER[name]
        paths = LEAF_PATH[name]
        # finest-level leaf order is taken from LEVEL_ORDER's last level.
        leaf_order = orders[-1]
        levels: list[dict[str, list[str]]] = []
        for k, order in enumerate(orders):
            level: dict[str, list[str]] = {cat: [] for cat in order}
            for leaf in leaf_order:
                level[paths[leaf][k]].append(leaf)
            levels.append(level)
        hier[name] = levels
    return hier


HIER = _build_hier()


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------
def leaves(name: str) -> list[str]:
    """Canonical ordered label2 leaf names (== finest level order)."""
    return list(HIER[name][-1].keys())


def n_levels(name: str) -> int:
    return len(HIER[name])


def label2_level(name: str) -> int:
    return n_levels(name) - 1


def categories(name: str, level: int) -> list[str]:
    """Ordered category names at ``level`` (defines the integer type code)."""
    return list(HIER[name][level].keys())


def leaf_to_category(name: str, level: int) -> dict[str, str]:
    """Map every leaf label2 name to its category at ``level``."""
    out: dict[str, str] = {}
    for cat, members in HIER[name][level].items():
        for leaf in members:
            out[leaf] = cat
    return out


# ---------------------------------------------------------------------------
# validation (runs at import)
# ---------------------------------------------------------------------------
def _validate() -> None:
    for name in NAMES:
        orders = LEVEL_ORDER[name]
        paths = LEAF_PATH[name]
        nlev = len(orders)
        # every leaf path has one entry per level
        for leaf, p in paths.items():
            assert len(p) == nlev, f"{name}: leaf {leaf!r} path len {len(p)} != {nlev}"
        leaf_set = set(paths)
        levels = HIER[name]
        for lv, (level, order) in enumerate(zip(levels, orders)):
            # category set matches the declared order (cross-check of the two encodings)
            assert list(level.keys()) == order, \
                f"{name}/L{lv}: category order {list(level.keys())} != declared {order}"
            assert len(order) == len(set(order)), f"{name}/L{lv}: duplicated category"
            members = [m for ms in level.values() for m in ms]
            assert len(members) == len(set(members)), f"{name}/L{lv}: duplicated leaf"
            assert set(members) == leaf_set, \
                f"{name}/L{lv}: leaf set mismatch {set(members) ^ leaf_set}"
            assert all(level[c] for c in order), f"{name}/L{lv}: empty category"
        # finest level is the identity (one leaf per category, category == leaf)
        last = levels[-1]
        assert all(len(v) == 1 and k == v[0] for k, v in last.items()), \
            f"{name}: finest level is not the identity"
        # strict refinement: each level's categories nest inside the previous level's
        for lv in range(nlev - 1):
            coarse = [set(v) for v in levels[lv].values()]
            for fset in (set(v) for v in levels[lv + 1].values()):
                assert any(fset <= cset for cset in coarse), \
                    f"{name}: L{lv+1} category {fset} not nested in any L{lv} category"


_validate()
