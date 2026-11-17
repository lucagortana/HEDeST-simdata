"""Spot lattices. There are three, and they are deliberately not merged.

Each family's existing data was built with its own lattice, and this repo has to reproduce
that data, so a tidier single implementation would silently move every spot. The
differences, measured against the three original implementations:

**Honeycomb** (STHELAR, Visium HD) -- centres ``spacing`` apart along a row, row pitch
``spacing * sqrt(3)/2``, odd-indexed rows shifted half a spacing, so all six neighbours of
a spot are equidistant. This is real Visium v2 geometry: 55 um circles, 100 um apart.
Both implementations shift odd row *indices*; they differ only in where the index origin
sits, which is :func:`honeycomb`'s ``margin_rings``:

    STHELAR    origin (x0, y0); the row at y0 is index 0, so it is NOT shifted
               (hedest_bench.pseudovisium.build_grid, margin_rings=0)
    Visium HD  origin (x0 - spacing, y0 - pitch); the row at y0 is index 1, so it IS
               shifted (VisiumHD_analysis.utils.grid.build, margin_rings=1)

So the stagger parity of the first occupied row differs between the two families, but as a
*consequence* of the origin, not as an independent choice -- which is why it is not a
separate argument. Checked against both: the measured row pitch is 86.6025 um in each
(= 100 * sqrt(3)/2, versus 100.0 for a brick), the even/odd rows are displaced by exactly
50.00 um, and the lowest ``array_row`` is 0 for STHELAR and 1 for Visium HD. Visium HD's
extra ring also adds spots that end up empty and are dropped; it is kept for exactness,
since it is what sets the ``array_row``/``array_col`` the datasets carry.

**Brick** (Xenium) -- staggered rows, but ``y`` steps by the **full spacing**, not
``spacing * sqrt(3)/2``. So it is not a honeycomb: with lung's 283/440 geometry the in-row
neighbour is 440 away and the staggered neighbour 491, where a honeycomb would put both at
440. The spot-to-pitch ratio is different too: 0.64 on lung and 0.70 on breast, against
Visium's 55/100 = 0.55, so the capture circles cover proportionally more of the slide.
Both come from ``make_spot_grid`` in the Xenium notebooks' ``utils/spatial_tools.py``, and
both are reproduced as they are -- the Xenium datasets are what they are, and bending them
onto a Visium honeycomb would change the ground truth the gridsearch was run against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

#: Which row carries the half-spacing shift.
ODD_ROWS = "odd"      # rows 1, 3, 5... are shifted (both honeycomb implementations)


@dataclass(frozen=True)
class SpotGrid:
    """A spot lattice in the units it was built in (microns, or H&E pixels for Xenium)."""

    centers: np.ndarray        # (S, 2)
    array_row: np.ndarray      # (S,) int
    array_col: np.ndarray      # (S,) int
    spacing: float
    diameter: float
    kind: str                  # "honeycomb" | "brick"

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    def __len__(self) -> int:
        return len(self.centers)

    def neighbour_distances(self) -> tuple[float, float]:
        """``(in-row, staggered)`` centre-to-centre distance -- equal iff honeycomb."""
        pitch = self.row_pitch
        return self.spacing, float(np.hypot(self.spacing / 2.0, pitch))

    @property
    def row_pitch(self) -> float:
        return (self.spacing * np.sqrt(3.0) / 2.0 if self.kind == "honeycomb"
                else self.spacing)


#: ``margin_rings`` for each family, which is the whole of the difference between the two
#: original honeycomb implementations. Changing one of these moves every spot of that
#: family and renumbers ``array_row``/``array_col``, so they are named, not passed inline.
MARGIN_RINGS = {"sthelar": 0, "visiumhd": 1}


def honeycomb(points: np.ndarray, spacing: float = 100.0, diameter: float = 55.0,
              *, margin_rings: int = 0) -> SpotGrid:
    """Visium v2 honeycomb covering the bounding box of ``points``.

    Args:
        points: ``(N, 2)`` coordinates the lattice must cover, in the lattice's units.
        spacing: centre-to-centre distance within a row.
        diameter: capture-circle diameter; must be smaller than ``spacing``, since the
            assignment in :func:`assign` relies on the circles not overlapping.
        margin_rings: how many rows and columns to add around the bounding box, which
            also sets the index origin: the row sitting at ``y0`` gets index
            ``margin_rings``, and odd indices are the shifted ones. Use
            :data:`MARGIN_RINGS` -- 0 for STHELAR, 1 for Visium HD.
    """
    if diameter >= spacing:
        raise ValueError(f"diameter {diameter} must be smaller than spacing {spacing}")
    pitch = spacing * np.sqrt(3.0) / 2.0
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)

    ox = x0 - margin_rings * spacing
    oy = y0 - margin_rings * pitch
    n_rows = int(np.ceil((y1 - y0) / pitch)) + 1 + 2 * margin_rings
    n_cols = int(np.ceil((x1 - x0) / spacing)) + 1 + 2 * margin_rings

    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    cx = ox + cols * spacing + np.where(rows % 2 == 1, spacing / 2.0, 0.0)
    cy = oy + rows * pitch
    return SpotGrid(np.c_[cx, cy], rows, cols, float(spacing), float(diameter),
                    "honeycomb")


def brick(x_min: float, x_max: float, y_min: float, y_max: float,
          diameter: float, spacing: float) -> SpotGrid:
    """The Xenium lattice: staggered rows, but ``y`` steps by the full ``spacing``.

    A faithful port of ``make_spot_grid``. The loop bounds are kept exactly as they
    were -- a spot is emitted only while its whole circle fits inside the box, so the
    last row and column are dropped rather than clipped, and the explicit signature
    (bounds rather than points) is kept because the notebooks passed the element extent,
    not the cell centroids.
    """
    radius = diameter / 2.0
    centers, rows, cols = [], [], []
    y = y_min + radius
    row = 0
    while y + radius <= y_max:
        x = x_min + (radius if row % 2 == 0 else spacing / 2.0 + radius)
        col = 0
        while x + radius <= x_max:
            centers.append((x, y))
            rows.append(row)
            cols.append(col)
            x += spacing
            col += 1
        y += spacing
        row += 1
    if not centers:
        raise ValueError("the lattice is empty: check the bounds against diameter/spacing")
    return SpotGrid(np.asarray(centers, dtype=np.float64), np.asarray(rows),
                    np.asarray(cols), float(spacing), float(diameter), "brick")


def assign(points: np.ndarray, grid: SpotGrid) -> np.ndarray:
    """Index of the spot whose circle contains each point, ``-1`` if none.

    The nearest centre is the only candidate because the circles never overlap, which is
    what :func:`honeycomb` enforces and what the Xenium geometries satisfy as well
    (283 < 440, 350 < 500).
    """
    dist, idx = cKDTree(grid.centers).query(points, k=1)
    idx = idx.astype(np.int64)
    idx[dist > grid.radius] = -1
    return idx


def check_reproduces(grid: SpotGrid, reference_centers: np.ndarray,
                     atol: float = 1e-6) -> dict:
    """Does ``grid`` contain every spot of an existing dataset, at the same place?

    Used by the ``verify`` stages and by ``tests/inventory.py``: a lattice rewrite is the
    one change that would move every spot while every file still looks well-formed, so it
    is checked against the reference rather than assumed.

    Returns a report with the number of reference spots matched and the largest
    displacement. ``matched == len(reference_centers)`` is the passing condition; a
    rebuilt lattice may legitimately hold extra spots, since empty ones are dropped later.
    """
    tree = cKDTree(grid.centers)
    dist, _ = tree.query(reference_centers, k=1)
    return {"reference_spots": int(len(reference_centers)),
            "matched": int((dist <= atol).sum()),
            "max_displacement": float(dist.max()) if len(dist) else 0.0,
            "grid_spots": len(grid)}
