"""Checks the download/unpack logic, with no network and no large files.

Worth its own test: the archive entries in ``download/urls.json`` were dead code until the
10x URLs were filled in, and the three bundles do not agree on layout --
``binned_outputs.tar.gz`` wraps its contents in ``binned_outputs/``, the breast
``outs.zip`` in ``outs/``, and the lung ``outs.zip`` has its files at the top level. Get
``strip`` wrong for one of them and the raw inputs land one directory off, which the
pipeline only notices several stages later.

    python tests/test_download.py
"""
from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hedest_simdata.common import download as dl    # noqa: E402
from hedest_simdata.common import sources           # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def _make_tar(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for n in names:
            data = n.encode()
            info = tarfile.TarInfo(n)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _make_zip(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for n in names:
            zf.writestr(n, n)


def _extract_into(tmp: Path, kind: str, names: list[str], strip: str | None) -> set[str]:
    item = dl.Item(url="unused", dest=tmp / "dest", unpack=kind, strip=strip)
    item.archive.parent.mkdir(parents=True, exist_ok=True)
    maker = _make_tar if kind == "tar" else _make_zip
    maker(item.archive, names)
    dl._extract(item)
    return {str(p.relative_to(item.dest)) for p in item.dest.rglob("*") if p.is_file()}


def main() -> int:
    print("\nthe three real archive layouts unpack to the same place")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        # Visium HD: tar wrapping everything in binned_outputs/
        got = _extract_into(tmp / "a", "tar",
                            ["binned_outputs/", "binned_outputs/square_002um/x.h5",
                             "binned_outputs/square_008um/y.h5"], "binned_outputs")
        check("tar, wrapped: the wrapper is stripped",
              got == {"square_002um/x.h5", "square_008um/y.h5"}, str(sorted(got)))

        # Xenium breast: zip wrapping everything in outs/
        got = _extract_into(tmp / "b", "zip",
                            ["outs/cells.parquet", "outs/cell_feature_matrix/m.mtx"],
                            "outs")
        check("zip, wrapped: the wrapper is stripped",
              got == {"cells.parquet", "cell_feature_matrix/m.mtx"}, str(sorted(got)))

        # Xenium lung: zip with its files at the top level
        got = _extract_into(tmp / "c", "zip",
                            ["analysis.zarr.zip", "cells.parquet"], None)
        check("zip, flat: nothing is stripped",
              got == {"analysis.zarr.zip", "cells.parquet"}, str(sorted(got)))

        print("\na changed layout is an error, not a silent re-rooting")
        try:
            _extract_into(tmp / "d", "tar", ["something_else/x.h5"], "binned_outputs")
            check("a member outside the declared wrapper raises", False)
        except dl.DownloadError as exc:
            check("a member outside the declared wrapper raises", True,
                  "layout has changed" in str(exc) and "as it should" or "")

        try:
            _extract_into(tmp / "e", "tar", ["outs/../../escape.h5"], "outs")
            check("a member escaping the destination raises", False)
        except dl.DownloadError as exc:
            check("a member escaping the destination raises", "escapes" in str(exc))

        print("\nan unpacked destination counts as complete, an empty one does not")
        item = dl.Item(url="u", dest=tmp / "f" / "dest", unpack="tar", strip=None)
        check("missing destination is not complete", dl.complete(item) is False)
        item.dest.mkdir(parents=True)
        check("empty destination is not complete", dl.complete(item) is False)
        (item.dest / "x").write_text("x")
        check("non-empty destination is complete", dl.complete(item) is True)
        check("the archive is staged beside the destination, hidden and marked .part",
              item.archive.name.startswith(".") and item.archive.name.endswith(".part"),
              item.archive.name)

    print("\nevery declared source resolves, and every archive declares its layout")
    from hedest_simdata import config
    root = Path("/tmp/does_not_exist")
    per_unit: list[tuple[str, list]] = []
    per_unit += [(f"sthelar:{sl}", sources.sthelar(root, sl))
                 for sl in config.STHELAR_SLIDES]
    per_unit += [(f"visiumhd:{p}", sources.visiumhd(root, config.visiumhd_name(p), p))
                 for p in config.VISIUMHD_PATIENTS]
    per_unit += [(f"xenium:{k}", sources.xenium(root, k, config.XENIUM_SLIDES[k].name))
                 for k in config.XENIUM_SLIDES]
    listed = [(unit, s) for unit, srcs in per_unit for s in srcs]

    seen_archive = set()
    for unit, s in listed:
        if s.unpack and (s.unpack, s.strip) not in seen_archive:
            seen_archive.add((s.unpack, s.strip))
            check(f"{unit} {s.role}: unpack={s.unpack} strip={s.strip!r}, with a size",
                  s.size is not None)
    # Every source now has a verified URL. Kept as a check rather than deleted: if one
    # ever has to go back to a manual download, that should be a deliberate edit here and
    # not something a user discovers three stages into a run.
    manual = [f"{unit}:{s.role}" for unit, s in listed if not s.url and not s.optional]
    check("no source needs a manual download", not manual, str(manual))
    # a role listed twice for one unit would mean per_slide_extra appended where it was
    # meant to override, and the first (wrongly stripped) copy would be the one extracted
    for unit, srcs in per_unit:
        roles = [s.role for s in srcs]
        check(f"{unit}: no role listed twice", len(set(roles)) == len(roles),
              f"{len(roles)} source(s)")
    # the override has to take effect, not merely not duplicate
    breast = {s.role: s for s in dict(per_unit)["xenium:breast"]}
    check("xenium:breast's outputs archive is the one that wraps in outs/",
          breast["outputs"].strip == "outs")
    lung = {s.role: s for s in dict(per_unit)["xenium:lung"]}
    check("xenium:lung's outputs archive is the flat one", lung["outputs"].strip is None)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES))
        return 1
    print("all download checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
