"""Annotation levels -> hierarchy, category order, unassigned cells.

Levels are obs columns given coarse to fine (``lv1_annot lv2_annot lv3_annot``);
level ``k`` of the outputs is the ``k``-th column. They must form a strict tree:
every category of a level has exactly one parent in the level above.

Category order (proportion columns, legends, tree): families in the order
epithelial, immune, stromal, others alphabetically, unassigned last; then, depth
first, children alphabetically with unassigned children last.

A category counts as *unassigned* when its name matches the unassigned pattern
(case-insensitive), or when it is named explicitly as ``lv<k>:Name`` (exact, for
one level) or ``Name`` (exact, every level where it occurs) -- for catch-all
categories whose name says nothing, such as a leaf called ``Epithelial``. Every
cell unassigned at any level is dropped from every level.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

_FAMILY_RANK = [re.compile(p, re.I) for p in (r"epithel", r"immun|lymph|myeloid", r"strom|struct|mesench")]


class LabelError(RuntimeError):
    pass


@dataclass
class Hierarchy:
    columns: list[str]
    levels: list[list[str]]              # categories per level, canonical order
    parent: list[dict[str, str]]         # parent[k][child] -> category at k-1 (parent[0] empty)
    unassigned: set[tuple[int, str]]

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def ancestor(self, level: int, name: str, target: int) -> str:
        while level > target:
            name = self.parent[level][name]
            level -= 1
        return name

    def children(self, level: int, name: str) -> list[str]:
        return [c for c in self.levels[level + 1] if self.parent[level + 1][c] == name]

    def to_json(self) -> list[dict]:
        return [{"level": k, "column": self.columns[k], "n_types": len(self.levels[k]),
                 "cell_types": self.levels[k],
                 "unassigned_types": [n for n in self.levels[k] if (k, n) in self.unassigned],
                 "parent": self.parent[k] if k else None}
                for k in range(self.n_levels)]


def _rank(name: str, is_unassigned: bool) -> tuple:
    if is_unassigned:
        return (9, name.lower())
    for i, pat in enumerate(_FAMILY_RANK):
        if pat.search(name):
            return (i, name.lower())
    return (5, name.lower())


def build(labels: pd.DataFrame, columns: list[str], pattern: str,
          explicit: list[str] = ()) -> Hierarchy:
    """Validate the level columns of ``labels`` and derive the hierarchy."""
    missing = [c for c in columns if c not in labels.columns]
    if missing:
        raise LabelError(f"level columns not found: {missing}")
    na = labels[columns].isna().sum()
    if na.any():
        raise LabelError(f"missing labels: {na[na > 0].to_dict()}")
    pat = re.compile(pattern, re.I)
    lab = labels[columns].astype(str)

    parent: list[dict[str, str]] = [{}]
    for k in range(1, len(columns)):
        pairs = lab[[columns[k - 1], columns[k]]].drop_duplicates()
        multi = pairs.groupby(columns[k])[columns[k - 1]].nunique()
        multi = multi[multi > 1]
        if len(multi):
            raise LabelError(f"not a strict tree: {columns[k]} categories with several parents in "
                             f"{columns[k - 1]}: {multi.to_dict()}")
        parent.append(dict(zip(pairs[columns[k]], pairs[columns[k - 1]])))

    present = [set(lab[c].unique()) for c in columns]
    unassigned = {(k, n) for k in range(len(columns)) for n in present[k] if pat.search(n)}
    for entry in explicit or ():
        level, _, name = entry.rpartition(":")
        if level:                                        # "lv2:Epithelial"
            if not re.fullmatch(r"lv\d+", level):
                raise LabelError(f"--unassigned-types entry {entry!r}: expected 'lv<k>:Name' or 'Name'")
            k = int(level[2:])
            if k >= len(columns) or name not in present[k]:
                raise LabelError(f"--unassigned-types entry {entry!r}: no category {name!r} at lv{k}")
            unassigned.add((k, name))
        else:                                            # "Name" -> every level holding it
            hits = {(k, name) for k in range(len(columns)) if name in present[k]}
            if not hits:
                raise LabelError(f"--unassigned-types entry {entry!r}: no category with that name")
            unassigned |= hits

    order: list[list[str]] = [[] for _ in columns]

    def visit(k: int, name: str) -> None:
        order[k].append(name)
        if k + 1 < len(columns):
            kids = [c for c in present[k + 1] if parent[k + 1][c] == name]
            # alphabetical, unassigned children last
            for c in sorted(kids, key=lambda c: ((k + 1, c) in unassigned, c.lower())):
                visit(k + 1, c)

    for fam in sorted(present[0], key=lambda n: _rank(n, (0, n) in unassigned)):
        visit(0, fam)
    return Hierarchy(list(columns), order, parent, unassigned)


def unassigned_mask(labels: pd.DataFrame, h: Hierarchy) -> np.ndarray:
    """True for cells whose label is unassigned at any level."""
    mask = np.zeros(len(labels), dtype=bool)
    for k, col in enumerate(h.columns):
        bad = [n for n in h.levels[k] if (k, n) in h.unassigned]
        mask |= labels[col].astype(str).isin(bad).to_numpy()
    return mask


def restrict(h: Hierarchy, labels: pd.DataFrame) -> Hierarchy:
    """The hierarchy of a subset of cells: categories left empty are dropped."""
    present = [set(labels[c].astype(str).unique()) for c in h.columns]
    levels = [[n for n in h.levels[k] if n in present[k]] for k in range(h.n_levels)]
    parent = [{c: p for c, p in h.parent[k].items() if c in present[k]} for k in range(h.n_levels)]
    unassigned = {(k, n) for k, n in h.unassigned if n in present[k]}
    return Hierarchy(h.columns, levels, parent, unassigned)


def counts(h: Hierarchy, labels: pd.DataFrame) -> dict[tuple[int, str], int]:
    out = {}
    for k, col in enumerate(h.columns):
        vc = labels[col].astype(str).value_counts()
        for n in h.levels[k]:
            out[(k, n)] = int(vc.get(n, 0))
    return out
