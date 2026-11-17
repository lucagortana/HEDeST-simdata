"""Cell-type colours for the bench_data figures, taken from HEDeST rather than copied.

The STHELAR colour table is hand-written and already exists in three places --
``hedest_bench.celltype_colors``, ``HEDeST/simulations/semi_simulations/STHELAR/
palette.py`` and ``HEDeST/external/HistoCell/bench.py`` -- duplicated so that neither
consumer has to depend on the other repository, and kept honest by
``hedest_bench.check_palette_sync``.

This repo does not add a fourth copy. It already requires HEDeST (for HoVerNet, CellViT
and H-Optimus-0), the dependency direction permits it, and the table is loaded from
HEDeST's copy by file path. Verified before adopting it: the three existing copies agree
on all 43 cell types as of 2026-11-05, so reading HEDeST's gives the colours the published
figures already have.

Colours are keyed by category *name*, so one cell type is the same colour at every level
and in every other figure in the project; a coarse category is looked up like any other
rather than averaged from its leaves, which is what an earlier generative scheme did and
why it produced muddy in-between parents.

``HOVERNET_TYPE_COLORS`` is deliberately separate: it colours raw HoVer-Net classes, which
are not cell types, and is unaffected by the palette.
"""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from ..hedest_bridge import Bridge
from ..hedest_bridge import BridgeError
from . import bench_hier

#: Where the table lives inside a HEDeST checkout.
PALETTE_REL = Path("simulations") / "semi_simulations" / "STHELAR" / "palette.py"


@lru_cache(maxsize=4)
def _load(hedest: Path) -> ModuleType:
    """Imports HEDeST's palette module by path, without putting it on ``sys.path``."""
    src = hedest / PALETTE_REL
    if not src.exists():
        raise BridgeError(
            f"the STHELAR colour table was not found at {src}. It is HEDeST's copy of the "
            "hand-written palette; this repo reads it instead of keeping a fourth copy.")
    spec = importlib.util.spec_from_file_location("_hedest_sthelar_palette", src)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise BridgeError(f"cannot load {src}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _table(bridge: Bridge) -> ModuleType:
    return _load(bridge.hedest)


def leaf_colors(bridge: Bridge, name: str) -> dict[str, tuple[float, float, float]]:
    """label2 leaf -> RGB (0..1)."""
    pal = _table(bridge)
    leaves = [leaf for members in bench_hier.HIER[name][0].values() for leaf in members]
    missing = pal.unknown(leaves)
    if missing:
        raise KeyError(f"{name}: cell types absent from the colour table: {missing}")
    return {leaf: pal.rgb(leaf) for leaf in leaves}


def level_palette(bridge: Bridge, name: str, level: int
                  ) -> dict[str, tuple[float, float, float]]:
    """category (at ``level``) -> RGB."""
    pal = _table(bridge)
    cats = list(bench_hier.HIER[name][level])
    missing = pal.unknown(cats)
    if missing:
        raise KeyError(
            f"{name} level{level}: cell types absent from the colour table: {missing}")
    return {cat: pal.rgb(cat) for cat in cats}


def n_cell_types(bridge: Bridge) -> int:
    """Size of the table, so a run can record which palette it drew with."""
    return len(_table(bridge).CELLTYPE_HEX)


# HoVerNet raw-class colours, straight from external/hovernet/type_info.json
# (int class -> RGB 0..1), for the raw HoVerNet-type segmentation plot.
HOVERNET_TYPE_COLORS: dict[int, tuple[float, float, float]] = {
    0: (0.0, 0.0, 0.0),          # nolabe
    1: (1.0, 0.0, 0.0),          # neopla
    2: (0.0, 1.0, 0.0),          # inflam
    3: (0.0, 0.0, 1.0),          # connec
    4: (1.0, 1.0, 0.0),          # necros
    5: (1.0, 165 / 255, 0.0),    # no-neo
}
