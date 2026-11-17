"""The lattice refactor must be behaviour-preserving, so it is tested against verbatim
copies of the three original implementations.

Merging three spot grids into one module is the single change in this repo that could move
every spot while every output file still looks perfectly well-formed, and no downstream
check would notice. So the originals are reproduced here, character for character from

    HEDeST-bench/src/hedest_bench/pseudovisium.py::build_grid
    VisiumHD_analysis/utils/grid.py::build
    HEDeST/simulations/semi_simulations/Xenium/dataset_construction/utils/
        spatial_tools.py::make_spot_grid

and the ported versions must agree with them exactly -- same centres, same row and column
indices, same order.

    python tests/test_lattice.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hedest_simdata.common import lattice  # noqa: E402

# --------------------------------------------------------------------------------------
# Verbatim originals
# --------------------------------------------------------------------------------------


def original_sthelar(points, spacing=100.0, diameter=55.0):
    """hedest_bench.pseudovisium.build_grid, with the MicronToHE mapping dropped."""
    radius = diameter / 2.0
    row_pitch = spacing * np.sqrt(3.0) / 2.0
    x0, y0 = points[:, 0].min(), points[:, 1].min()
    x1, y1 = points[:, 0].max(), points[:, 1].max()
    centers, rows, cols = [], [], []
    n_rows = int(np.ceil((y1 - y0) / row_pitch)) + 1
    for r in range(n_rows):
        cy = y0 + r * row_pitch
        x_off = (spacing / 2.0) if (r % 2) else 0.0
        n_cols = int(np.ceil((x1 - x0) / spacing)) + 1
        for c in range(n_cols):
            cx = x0 + x_off + c * spacing
            centers.append((cx, cy))
            rows.append(r)
            cols.append(c)
    del radius
    return (np.asarray(centers, dtype=np.float64), np.asarray(rows), np.asarray(cols))


def original_visiumhd(points, spacing_um=100.0, diameter_um=55.0):
    """VisiumHD_analysis.utils.grid.build."""
    pitch = spacing_um * np.sqrt(3.0) / 2.0
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    n_rows = int(np.ceil((y1 - y0) / pitch)) + 3
    n_cols = int(np.ceil((x1 - x0) / spacing_um)) + 3
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    cx = x0 - spacing_um + cols * spacing_um + np.where(rows % 2 == 1, spacing_um / 2.0, 0.0)
    cy = y0 - pitch + rows * pitch
    del diameter_um
    return np.c_[cx, cy], rows, cols


def original_xenium(x_min, x_max, y_min, y_max, diameter, spacing):
    """spatial_tools.make_spot_grid, returning centres instead of shapely circles."""
    radius = diameter / 2
    centers, rows, cols = [], [], []
    y = y_min + radius
    row_index = 0
    while y + radius <= y_max:
        x = x_min + (radius if row_index % 2 == 0 else (spacing / 2) + radius)
        c = 0
        while x + radius <= x_max:
            centers.append((x, y))
            rows.append(row_index)
            cols.append(c)
            x += spacing
            c += 1
        y += spacing
        row_index += 1
    return np.asarray(centers, dtype=np.float64), np.asarray(rows), np.asarray(cols)


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


#: Tolerance on a spot centre, in the lattice's units (um for the honeycombs).
#:
#: Not zero, and deliberately so. The originals accumulate ``x0 + x_off + c * spacing``
#: as Python scalars; the port evaluates ``ox + cols * spacing + offset`` vectorised, a
#: different association order, so the two differ by one float64 rounding step -- measured
#: at most 4.6e-13 um over the trials below, i.e. 0.5 femtometres on a 100 um pitch.
#:
#: What that can affect is one thing only: whether a cell centroid lands inside a 27.5 um
#: capture circle, in ``assign``. A displacement d flips a cell only if its centroid sits
#: within d of the circle boundary, so the expected number of flips is about
#: ``n_cells * d / radius`` ~ 1e6 * 4.6e-13 / 27.5 ~ 2e-8 across every dataset in this
#: repo. :func:`test_assignment_is_unchanged` checks it directly rather than trusting the
#: estimate.
CENTRE_ATOL = 1e-9


def same(got, want) -> tuple[bool, str]:
    gc, gr, gcol = got
    wc, wr, wcol = want
    if gc.shape != wc.shape:
        return False, f"{gc.shape} vs {wc.shape}"
    d = np.abs(gc - wc).max() if len(gc) else 0.0
    ok = d <= CENTRE_ATOL and np.array_equal(gr, wr) and np.array_equal(gcol, wcol)
    return ok, f"n={len(gc)}, max|dcentre|={d:.3g}"


def test_honeycomb_matches_originals() -> None:
    rng = np.random.default_rng(0)
    for trial in range(5):
        pts = rng.uniform([0, 0], [4000, 3000], size=(500, 2)) + rng.uniform(0, 137)
        for family, orig in (("sthelar", original_sthelar), ("visiumhd", original_visiumhd)):
            g = lattice.honeycomb(pts, margin_rings=lattice.MARGIN_RINGS[family])
            ok, detail = same((g.centers, g.array_row, g.array_col), orig(pts))
            check(f"honeycomb[{family}] == original (trial {trial})", ok, detail)


def test_brick_matches_original() -> None:
    from hedest_simdata import config
    for key, slide in config.XENIUM_SLIDES.items():
        box = (0.0, 9000.0, 0.0, 7000.0)
        g = lattice.brick(*box, diameter=slide.diameter, spacing=slide.spacing)
        ok, detail = same((g.centers, g.array_row, g.array_col),
                          original_xenium(*box, slide.diameter, slide.spacing))
        check(f"brick[{key}] == original", ok, detail)


def test_geometry_properties() -> None:
    """The structural claims the module docstring makes."""
    rng = np.random.default_rng(1)
    pts = rng.uniform([0, 0], [5000, 5000], size=(400, 2))

    g = lattice.honeycomb(pts, margin_rings=0)
    in_row, staggered = g.neighbour_distances()
    check("honeycomb neighbours are equidistant", abs(in_row - staggered) < 1e-9,
          f"{in_row:.4f} / {staggered:.4f}")
    check("honeycomb row pitch is spacing*sqrt(3)/2",
          abs(g.row_pitch - 100 * np.sqrt(3) / 2) < 1e-9, f"{g.row_pitch:.4f}")

    # the parity consequence, which is what the two margin_rings values really change
    for family, expected_first_row in (("sthelar", 0), ("visiumhd", 1)):
        g = lattice.honeycomb(pts, margin_rings=lattice.MARGIN_RINGS[family])
        y0 = pts[:, 1].min()
        at_y0 = np.isclose(g.centers[:, 1], y0)
        rows_at_y0 = np.unique(g.array_row[at_y0])
        check(f"{family}: the row at y0 has index {expected_first_row}",
              rows_at_y0.tolist() == [expected_first_row], f"got {rows_at_y0.tolist()}")

    b = lattice.brick(0.0, 9000.0, 0.0, 7000.0, diameter=283.0, spacing=440.0)
    in_row, staggered = b.neighbour_distances()
    check("brick neighbours are NOT equidistant", staggered > in_row + 1,
          f"in-row {in_row:.1f}, staggered {staggered:.1f}")
    check("brick row pitch is the full spacing", abs(b.row_pitch - 440.0) < 1e-9)


def test_assign_is_containment() -> None:
    rng = np.random.default_rng(2)
    pts = rng.uniform([0, 0], [2000, 2000], size=(2000, 2))
    g = lattice.honeycomb(pts, margin_rings=0)
    idx = lattice.assign(pts, g)
    inside = idx >= 0
    d = np.linalg.norm(pts[inside] - g.centers[idx[inside]], axis=1)
    check("every assigned point is inside its circle", bool((d <= g.radius + 1e-12).all()),
          f"{inside.sum()}/{len(pts)} assigned, max d={d.max():.2f} <= {g.radius}")
    # gaps are real: 55/100 circles cover pi*27.5^2 / (100*86.6025) = 27.4 % of the plane
    frac = inside.mean()
    check("coverage is the honeycomb's 27.4 %", 0.24 < frac < 0.31, f"{frac:.1%}")


def test_assignment_is_unchanged() -> None:
    """The only thing the float64 rounding difference could change: which spot a cell
    joins. Checked on a quarter of a million points per family, at the coordinate
    magnitudes the real slides use (tens of thousands of microns, where float64
    resolution is ~1e-11, so the rounding step is at its largest)."""
    rng = np.random.default_rng(3)
    for family, orig in (("sthelar", original_sthelar), ("visiumhd", original_visiumhd)):
        pts = rng.uniform([12000, 8000], [42000, 38000], size=(250_000, 2))
        mine = lattice.honeycomb(pts, margin_rings=lattice.MARGIN_RINGS[family])
        wc, wr, wcol = orig(pts)
        theirs = lattice.SpotGrid(wc, wr, wcol, 100.0, 55.0, "honeycomb")
        a, b = lattice.assign(pts, mine), lattice.assign(pts, theirs)
        n_diff = int((a != b).sum())
        check(f"{family}: no cell changes spot", n_diff == 0,
              f"{n_diff}/{len(pts)} differ, {int((a >= 0).sum())} assigned")


def main() -> int:
    for fn in (test_honeycomb_matches_originals, test_brick_matches_original,
               test_geometry_properties, test_assign_is_containment,
               test_assignment_is_unchanged):
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + ("all lattice checks passed" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
