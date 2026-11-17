"""Pseudo-Visium v2 grid: staggered spot lattice + per-spot aggregation.

The grid is built in micron space (the natural tissue frame where cells live),
spots collect cells by nucleus centroid, and spot centers are transformed to
H&E pixels for storage / plotting. Aggregation is cell-based: a spot's
transcriptome is the sum of its member cells' Xenium counts and its
proportions are those cells' cell-type fractions, so the two are mutually
consistent. "Other" cells are excluded before this step.

The lattice itself now comes from :mod:`hedest_simdata.common.lattice`, which the three
families share; ``SpotGrid`` here stays, because it also carries the H&E-pixel view of the
same spots, which is family-specific. ``tests/test_lattice.py`` checks the shared builder
against a verbatim copy of the one this module used to contain.
"""
from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..common import lattice
from . import constants
from .transforms import MicronToHE


@dataclass
class SpotGrid:
    centers_um: np.ndarray      # (S, 2) (x, y) micron
    centers_he: np.ndarray      # (S, 2) (x, y) H&E pixel
    array_row: np.ndarray       # (S,) int
    array_col: np.ndarray       # (S,) int
    radius_um: float
    radius_he: float


def build_grid(centroids_um: np.ndarray, tf: MicronToHE) -> SpotGrid:
    """Honeycomb (staggered) spot lattice covering the cell bounding box."""
    grid = lattice.honeycomb(centroids_um,
                             spacing=constants.SPOT_SPACING_UM,
                             diameter=constants.SPOT_DIAMETER_UM,
                             margin_rings=lattice.MARGIN_RINGS["sthelar"])
    return SpotGrid(
        centers_um=grid.centers,
        centers_he=tf(grid.centers),
        array_row=grid.array_row,
        array_col=grid.array_col,
        radius_um=grid.radius,
        radius_he=grid.radius * tf.px_per_um,
    )


def assign_cells_to_spots(centroids_um: np.ndarray, grid: SpotGrid) -> np.ndarray:
    """Index of the spot whose circle contains each cell centroid (-1 if none).

    Circles do not overlap (diameter 55 < spacing 100), so the nearest spot
    center is the only possible container.
    """
    return lattice.assign(
        centroids_um,
        lattice.SpotGrid(grid.centers_um, grid.array_row, grid.array_col,
                         constants.SPOT_SPACING_UM, grid.radius_um * 2.0, "honeycomb"))


@dataclass
class SpotAggregation:
    """Result of aggregating cells into spots (annotation-independent).

    ``adata`` carries the raw summed counts in ``X`` and the log-normalized
    counts in ``layers[constants.LOGNORM_LAYER]``. The fields ``member_cells`` /
    ``member_spot`` / ``n_spots`` / ``spot_ids`` let :func:`compute_proportions`
    build a cell-type proportion table for *any* annotation over the exact same
    spot assignment (so the two annotations share ``adata`` and ``n_cells``).
    """
    adata: "ad.AnnData"
    spot_table: pd.DataFrame
    member_cells: np.ndarray      # indices into the non-Other subset, of in-spot cells
    member_spot: np.ndarray       # (k,) spot position in [0, n_spots) for each member cell
    n_spots: int
    n_cells_per_spot: np.ndarray
    spot_ids: np.ndarray


def _log_normalize(counts: np.ndarray) -> np.ndarray:
    """scanpy-style ``normalize_total`` (median target) + ``log1p``.

    ``counts`` is a dense (n_spots, n_genes) raw-count matrix. Each spot is
    scaled to the median library size over non-empty spots, then ``log1p``-ed.
    """
    totals = counts.sum(axis=1)
    nz = totals > 0
    target = float(np.median(totals[nz])) if nz.any() else 1.0
    out = np.zeros_like(counts, dtype=np.float64)
    out[nz] = counts[nz] * (target / totals[nz][:, None])
    np.log1p(out, out=out)
    return out


def aggregate(
    rec_centroids_um: np.ndarray,
    rec_cell_ids: np.ndarray,
    expr_cell_ids: np.ndarray,
    expr_X: csr_matrix,
    genes: np.ndarray,
    grid: SpotGrid,
    tf: MicronToHE,
    sample: str,
) -> SpotAggregation:
    """Aggregate cells into spots (raw-count transcriptome).

    Inputs are restricted to the non-Other cells (identical for both
    annotations). ``expr_X`` must be the **raw counts** matrix. The spot
    transcriptome is the sum of its member cells' raw counts; a log-normalized
    version is stored as a layer. Only spots with >=1 member cell are kept.
    """
    spot_idx = assign_cells_to_spots(rec_centroids_um, grid)
    in_spot = spot_idx >= 0

    # Map (non-Other) cell rows -> rows of the expression matrix.
    expr_pos = pd.Series(np.arange(len(expr_cell_ids)), index=expr_cell_ids)
    cell_expr_row = expr_pos.reindex(rec_cell_ids).to_numpy()
    assert not np.isnan(cell_expr_row).any(), "cell_id missing from expression table"
    cell_expr_row = cell_expr_row.astype(np.int64)

    used_spots = np.unique(spot_idx[in_spot])
    n_spots = len(used_spots)

    # Vectorised aggregation: spot_expr = A @ X_sub, where A is the
    # (n_spots x n_member_cells) 0/1 assignment matrix.
    cells = np.where(in_spot)[0]
    spot_of = np.searchsorted(used_spots, spot_idx[cells])            # (k,)
    rows = cell_expr_row[cells]
    Xcsr = expr_X.tocsr()
    X_sub = Xcsr[rows]                                                # (k, g) raw counts
    A = csr_matrix(
        (np.ones(len(cells), dtype=np.float64), (spot_of, np.arange(len(cells)))),
        shape=(n_spots, len(cells)),
    )
    spot_expr = np.asarray((A @ X_sub).todense())                    # (S, g) raw counts
    log_norm = _log_normalize(spot_expr)

    spot_ids = np.array([f"spot_{i:06d}" for i in range(n_spots)])
    centers_he = grid.centers_he[used_spots]
    centers_um = grid.centers_um[used_spots]
    n_cells_per_spot = np.bincount(spot_of, minlength=n_spots)

    obs = pd.DataFrame(
        {
            "array_row": grid.array_row[used_spots],
            "array_col": grid.array_col[used_spots],
            "pxl_col_in_fullres": centers_he[:, 0],
            "pxl_row_in_fullres": centers_he[:, 1],
            "x_um": centers_um[:, 0],
            "y_um": centers_um[:, 1],
            "n_cells": n_cells_per_spot,
            "in_tissue": 1,
        },
        index=spot_ids,
    )

    adata = ad.AnnData(
        X=csr_matrix(spot_expr.astype(np.float32)),                  # raw counts
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    adata.layers[constants.LOGNORM_LAYER] = csr_matrix(log_norm.astype(np.float32))
    adata.obsm["spatial"] = centers_he.astype(np.float64)
    diam_he = constants.SPOT_DIAMETER_UM * tf.px_per_um
    adata.uns["spatial"] = {
        sample: {
            "scalefactors": {
                "spot_diameter_fullres": float(diam_he),
                "tissue_hires_scalef": 1.0,
            }
        }
    }
    adata.uns["pseudovisium"] = {
        "spot_diameter_um": constants.SPOT_DIAMETER_UM,
        "spot_spacing_um": constants.SPOT_SPACING_UM,
        "mpp": tf.mpp,
        "X": "raw counts: sum of member cells' raw Xenium counts (table_cells layers['count'])",
        "layers": {constants.LOGNORM_LAYER: "normalize_total(median) + log1p of X"},
        "annotations": [constants.ANNOT_ORIGINAL, constants.ANNOT_FINE],
    }

    spot_table = pd.DataFrame(
        {
            "spot_id": spot_ids,
            "x_he": centers_he[:, 0],
            "y_he": centers_he[:, 1],
            "x_um": centers_um[:, 0],
            "y_um": centers_um[:, 1],
            "radius_he": grid.radius_he,
            "n_cells": n_cells_per_spot,
        }
    ).set_index("spot_id")

    return SpotAggregation(
        adata=adata,
        spot_table=spot_table,
        member_cells=cells,
        member_spot=spot_of,
        n_spots=n_spots,
        n_cells_per_spot=n_cells_per_spot,
        spot_ids=spot_ids,
    )


def compute_proportions(agg: SpotAggregation, types_subset: np.ndarray,
                        type_names: list[str]) -> pd.DataFrame:
    """Per-spot cell-type proportions for one annotation.

    ``types_subset`` are the cell-type labels of the *non-Other* cells, in the
    same order as the ``rec_centroids_um`` passed to :func:`aggregate`.
    """
    type_to_int = {t: i for i, t in enumerate(type_names)}
    type_of = np.array([type_to_int[t] for t in types_subset[agg.member_cells]])
    counts = np.zeros((agg.n_spots, len(type_names)), dtype=np.int64)
    np.add.at(counts, (agg.member_spot, type_of), 1)
    props = counts / np.clip(agg.n_cells_per_spot[:, None], 1, None)
    return pd.DataFrame(props, index=agg.spot_ids, columns=type_names)
