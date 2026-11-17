"""Low-level readers for the STHELAR SpatialData `.zarr.zip` archives.

We read individual elements straight out of the zip (zarr ``ZipStore`` for
arrays, ``zipfile`` + parquet for shapes / tables) instead of loading the whole
SpatialData object. The archives are 7-38 GB, so this keeps memory and time
under control and avoids pulling the ~30 GB transcript points we do not need.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import zarr
from scipy.sparse import csr_matrix

from . import constants
from . import paths



def open_zarr(sl: paths.Slide):
    """Return (store, root_group). Caller must close the store."""
    store = zarr.ZipStore(str(sl.zarr_zip), mode="r")
    root = zarr.open_group(store, mode="r", path=sl.zarr_inner_root)
    return store, root


def _read_member_parquet(zf: zipfile.ZipFile, name: str) -> bytes:
    return zf.read(name)


# --- images -----------------------------------------------------------------

def he_affine_and_mpp(root, nucleus_scale: float) -> tuple[np.ndarray, np.ndarray, float]:
    """H&E intrinsic (scale-0 pixel) -> global affine, in (x, y) order.

    Returns ``(L, b, mpp)`` such that for an H&E pixel ``(x, y)``::

        global_xy = L @ [x, y] + b

    NOTE: in the STHELAR NGFF metadata the H&E affine maps input axes
    ``(c, y, x)`` to output axes ``(c, x, y)`` -- i.e. the output order is
    swapped, encoding the ~90 deg rotation between H&E and the Xenium global
    frame. We parse the axis names explicitly instead of assuming an order.
    ``global`` is the morphology-image pixel space (size 1/nucleus_scale um);
    ``mpp`` (H&E micron-per-pixel) = sqrt(|det L|) / nucleus_scale.
    """
    ms = dict(root["images/he"].attrs)["multiscales"][0]
    ct = ms["coordinateTransformations"][0]
    affine = np.asarray(ct["affine"])
    in_axes = [a["name"] for a in ct["input"]["axes"]]
    out_axes = [a["name"] for a in ct["output"]["axes"]]
    ix, iy = in_axes.index("x"), in_axes.index("y")
    ox, oy = out_axes.index("x"), out_axes.index("y")
    L = np.array(
        [[affine[ox, ix], affine[ox, iy]],
         [affine[oy, ix], affine[oy, iy]]],
        dtype=np.float64,
    )
    b = np.array([affine[ox, -1], affine[oy, -1]], dtype=np.float64)
    mpp = float(np.sqrt(abs(np.linalg.det(L))) / nucleus_scale)
    return L, b, mpp


def nucleus_scale(sl: paths.Slide, root) -> float:
    """Scale that maps nucleus-shape (micron) coords to global pixels."""
    with zipfile.ZipFile(str(sl.zarr_zip)) as zf:
        attrs = json.loads(zf.read(f"{sl.zarr_inner_root}/shapes/nucleus_boundaries/.zattrs"))
    scale = attrs["coordinateTransformations"][0]["scale"]
    sx, sy = float(scale[0]), float(scale[1])
    assert abs(sx - sy) < 1e-6, f"non-isotropic shape scale {scale}"
    return sx


def he_level_shapes(root) -> list[tuple[int, int, int]]:
    g = root["images/he"]
    keys = sorted((k for k in g.array_keys()), key=lambda x: int(x))
    return [tuple(g[k].shape) for k in keys]


def read_he_level(root, level: int) -> np.ndarray:
    """Return the H&E pyramid level as a (C, Y, X) uint8 array."""
    return root[f"images/he/{level}"][:]


# --- shapes -----------------------------------------------------------------

def read_nucleus_boundaries(sl: paths.Slide) -> gpd.GeoDataFrame:
    """Nucleus polygons (micron coords), indexed by cell_id."""
    with zipfile.ZipFile(str(sl.zarr_zip)) as zf:
        b = _read_member_parquet(zf, f"{sl.zarr_inner_root}/shapes/nucleus_boundaries/shapes.parquet")
    return gpd.read_parquet(io.BytesIO(b))


def _decode_obs_column(grp, key):
    node = grp[key]
    sub = set(node.array_keys()) | set(node.group_keys()) if hasattr(node, "array_keys") else set()
    if "categories" in sub:
        return pd.Categorical.from_codes(node["codes"][:], node["categories"][:])
    return node[:]


@dataclass
class CellExpression:
    cell_ids: np.ndarray      # (n,) str
    genes: np.ndarray         # (g,) str
    X: csr_matrix             # (n, g) raw counts


def read_cell_expression(root, layer: str = "count") -> CellExpression:
    """Cell x gene counts from ``tables/table_cells``.

    ``layer="count"`` (default) reads the **raw integer counts** from
    ``layers/count`` -- this is what we sum into pseudo-spots. Note that
    ``table_cells.X`` is *not* raw counts: for every non-``Less10`` cell it holds
    the log-normalized expression (``normalize_total`` + ``log1p``); raw counts
    live in ``layers/count`` (and are only valid for cells that are not
    ``Less10`` -- which we always drop before aggregating). Pass ``layer=None``
    to read ``X`` instead.
    """
    g = root["tables/table_cells"]
    node = g["X"] if layer is None else g[f"layers/{layer}"]
    shape = tuple(dict(node.attrs)["shape"])
    X = csr_matrix(
        (node["data"][:], node["indices"][:], node["indptr"][:]),
        shape=shape,
    )
    var_index = _decode_obs_column(g["var"], "_index")
    cell_ids = np.asarray(_decode_obs_column(g["obs"], "cell_id")).astype(str)
    genes = np.asarray(var_index).astype(str)
    return CellExpression(cell_ids=cell_ids, genes=genes, X=X)


def read_cell_labels(sl: paths.Slide, columns) -> pd.DataFrame:
    """Several label columns at once, indexed by (string) cell_id."""
    columns = list(columns)
    df = pd.read_parquet(sl.cell_metadata, columns=["cell_id", *columns])
    df["cell_id"] = df["cell_id"].astype(str)
    return df.set_index("cell_id")[columns].astype(str)
