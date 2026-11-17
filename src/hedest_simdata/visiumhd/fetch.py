"""The Visium HD download stage.

Everything here has a direct, verified URL on 10x's CDN, so this downloads what is
missing -- unpacking ``binned_outputs.tar.gz`` and ``spatial.tar.gz`` in place -- and
checks each finished file against the byte count recorded in ``download/urls.json``.
Anything already present is left alone, which makes this also the cheap pre-flight check
on a manually placed download. See :mod:`hedest_simdata.common.sources`.
"""
from __future__ import annotations

from .. import config
from ..common import sources
from ..config import Layout
from . import paths
from .config import log


def run(layout: Layout, patient: str, dry_run: bool = False) -> list[dict]:
    name = config.visiumhd_name(patient)
    srcs = sources.visiumhd(layout.data_root, name, patient)
    log(f"{patient}: checking raw inputs in {layout.visiumhd_dir(patient)}")
    results = sources.fetch_or_explain(
        srcs, family="visiumhd", label=f"Visium HD {patient}",
        page=sources.page_for("visiumhd"), dry_run=dry_run)

    # the pipeline reads the 2 um bins and the scalefactors; check them specifically,
    # because an unpacked-but-truncated bundle looks present at the folder level
    s = paths.sample(layout, patient)
    needed = {"2 um bins": layout.visiumhd_dir(patient) / "binned_outputs" / "square_002um",
              "scalefactors": s.scalefactors}
    missing = [f"{k}: {p}" for k, p in needed.items() if not p.exists()]
    if missing:
        raise sources.MissingSource(
            f"Visium HD {patient}: the download is there but incomplete:\n  "
            + "\n  ".join(missing))
    log(f"{patient}: raw inputs complete")
    return results
