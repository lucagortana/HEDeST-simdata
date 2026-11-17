"""Coordinate transform: Xenium micron space -> high-res H&E pixel space.

Chain (see sdata_io.he_affine_and_mpp), all in (x, y) order:

    global_xy = L @ he_xy + b                 # H&E pixel -> global (morpho px)
    global_xy = nucleus_scale * micron_xy     # micron     -> global

so for a micron point (x, y)::

    he_xy = L^-1 @ (nucleus_scale * micron_xy - b)
"""
from __future__ import annotations

import numpy as np


class MicronToHE:
    def __init__(self, L: np.ndarray, b: np.ndarray, nucleus_scale: float, mpp: float):
        self.L = np.asarray(L, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.Linv = np.linalg.inv(self.L)
        self.scale = float(nucleus_scale)
        self.mpp = float(mpp)

    def __call__(self, xy_micron: np.ndarray) -> np.ndarray:
        """(N, 2) micron (x, y) -> (N, 2) H&E pixel (x, y)."""
        xy = np.asarray(xy_micron, dtype=np.float64)
        g = self.scale * xy                    # global (x, y)
        return (g - self.b) @ self.Linv.T      # H&E (x, y)

    @property
    def px_per_um(self) -> float:
        return 1.0 / self.mpp
