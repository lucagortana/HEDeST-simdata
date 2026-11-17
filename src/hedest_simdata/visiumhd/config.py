"""One Visium HD sample's paths and the per-stage metadata file.

Built from a :class:`~hedest_simdata.config.Layout` by :func:`hedest_simdata.visiumhd.
paths.sample`, not from argparse as in VisiumHD_analysis -- so the data root is a
command-line argument rather than a constant to edit.

Two differences from the pipeline this came from:

* **no ``<name>_annotated.h5ad``.** The manual cell-type calls are committed as a CSV
  (:mod:`hedest_simdata.visiumhd.cell_labels`) and everything else about those cells is
  re-derived from bin2cell's own output, so the 4.7 GB annotated file is not an input any
  more. ``annotated`` stays on the dataclass as an *optional* cross-check.
* **the bin2cell and segmentation outputs live in ``_work``**, since nothing downstream
  reads them once ``pseudovisium/`` is built.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..config import VISIUMHD_UNASSIGNED_PATTERN as UNASSIGNED_PATTERN
from ..config import SPOT_DIAMETER_UM
from ..config import SPOT_SPACING_UM

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LEVELS = ("lv1_annot", "lv2_annot", "lv3_annot")
DEFAULT_UNASSIGNED = UNASSIGNED_PATTERN
MPP_TOLERANCE = 0.005          # max relative spread between the mpp sources
SEED = 0


@dataclass
class Sample:
    name: str
    data_dir: Path
    annotated: Path
    pyr_tif: Path
    scalefactors: Path
    full_image: Path | None
    hovernet_json: Path
    label_mask: Path
    nucleus_table: Path
    bin_data: Path
    cell_data: Path
    out_dir: Path
    #: ``_work/visiumhd/<name>/``: intermediates nothing reads once ``pseudovisium/``
    #: exists. Named explicitly rather than derived from a sibling path, so a stage that
    #: needs scratch does not have to walk ``cell_data.parent.parent`` and hope.
    work_dir: Path | None = None
    levels: list[str] = field(default_factory=lambda: list(DEFAULT_LEVELS))
    unassigned_pattern: str = DEFAULT_UNASSIGNED
    unassigned_types: list[str] = field(default_factory=list)
    spot_diameter_um: float = SPOT_DIAMETER_UM
    spot_spacing_um: float = SPOT_SPACING_UM

    def required(self) -> dict[str, Path]:
        """Inputs a stage cannot start without.

        ``annotated`` is deliberately absent: the cell types are committed with the
        repository, so a fresh checkout has them without the 4.7 GB file.
        """
        return {k: getattr(self, k) for k in (
            "pyr_tif", "scalefactors", "hovernet_json", "label_mask",
            "bin_data", "cell_data")}

    def missing(self) -> list[str]:
        return [f"{k}: {p}" for k, p in self.required().items() if not Path(p).exists()]


# ---------------------------------------------------------------------------
# metadata.json: one section per stage, merged in place
# ---------------------------------------------------------------------------
def read_metadata(out_dir: Path) -> dict:
    p = Path(out_dir) / "metadata.json"
    return json.loads(p.read_text()) if p.exists() else {}


def update_metadata(out_dir: Path, section: str, content: dict) -> None:
    p = Path(out_dir) / "metadata.json"
    meta = read_metadata(out_dir)
    meta[section] = content
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=_json_default))
    os.replace(tmp, p)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def file_info(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    st = Path(path).stat()
    return {"path": str(path), "bytes": st.st_size, "mtime": st.st_mtime}


def git_commit(repo: Path) -> str | None:
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def log(msg: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
