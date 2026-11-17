"""Records what a data folder contains, and validates the manifest against it.

    python tests/inventory.py record --reference /path/to/data

Walks the folders and writes ``tests/manifests/<family>.json``: for every deliverable, its
size and (for files small enough) a sha256. It also reports any file the folder has that
:mod:`hedest_simdata.manifest` does not declare, and any declared file the folder does not
have -- which is how the declared file set gets checked against reality rather than against
someone's reading of the code. It exits non-zero on either.

To compare a **rebuild** against a reference, file content by file content, use
``tests/compare.py``: that one knows how each kind of file has to agree.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hedest_simdata import config          # noqa: E402
from hedest_simdata import manifest        # noqa: E402

#: Files below this are hashed in full; bigger ones are identified by size alone at
#: record time (``check`` compares them by policy, which reads them properly).
HASH_LIMIT = 64 * 1024 * 1024

MANIFEST_DIR = REPO / "tests" / "manifests"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(p: Path) -> dict:
    out: dict = {"bytes": p.stat().st_size}
    if out["bytes"] <= HASH_LIMIT:
        out["sha256"] = sha256(p)
    return out


def walk(root: Path) -> set[str]:
    """Every file under ``root``, as posix paths relative to it."""
    if not root.exists():
        return set()
    return {str(q.relative_to(root)) for q in root.rglob("*") if q.is_file()}


def classify(extra: set[str], excluded: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Splits undeclared files into 'excluded on purpose' and 'unexplained'."""
    explained: dict[str, str] = {}
    unexplained: list[str] = []
    for rel in sorted(extra):
        for pattern, reason in excluded.items():
            pat = pattern.rstrip("/")
            if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, f"{pat}/*") \
                    or rel.startswith(f"{pat}/") or fnmatch.fnmatch(Path(rel).name, pat):
                explained[rel] = reason
                break
        else:
            unexplained.append(rel)
    return explained, unexplained


# --------------------------------------------------------------------------------------


def n_levels_of(bench_dir: Path) -> int:
    return int(json.loads((bench_dir / "sim" / "levels.json").read_text())["n_levels"])


def record_sthelar(ref: Path) -> dict:
    root = ref / "STHELAR" / "bench_data"
    out: dict = {"family": "sthelar", "root": "STHELAR/bench_data", "datasets": {}}
    for name in config.BENCH_NAMES:
        d = root / name
        if not d.exists():
            out["datasets"][name] = {"missing": True}
            continue
        nlv = n_levels_of(d)
        specs = manifest.sthelar_files(name, nlv)
        declared = {f.path: f for f in specs}
        present = walk(d)
        excl = {k.format(name=name): v for k, v in manifest.STHELAR_EXCLUDED.items()}
        explained, unexplained = classify(present - set(declared), excl)
        out["datasets"][name] = {
            "n_levels": nlv,
            "files": {rel: {**describe(d / rel), "policy": declared[rel].policy}
                      for rel in sorted(set(declared) & present)},
            "declared_but_absent": sorted(set(declared) - present),
            "excluded_on_purpose": explained,
            "undeclared": unexplained,
        }
    return out


def record_visiumhd(ref: Path) -> dict:
    out: dict = {"family": "visiumhd", "datasets": {}}
    for patient in config.VISIUMHD_PATIENTS:
        slide = config.visiumhd_name(patient)
        d = ref / slide / "pseudovisium"
        if not d.exists():
            out["datasets"][patient] = {"missing": True}
            continue
        declared = {f.path: f for f in manifest.visiumhd_files(patient)}
        present = walk(d)
        explained, unexplained = classify(present - set(declared), manifest.VISIUMHD_EXCLUDED)
        entry: dict = {
            "root": f"{slide}/pseudovisium",
            "files": {rel: {**describe(d / rel), "policy": declared[rel].policy}
                      for rel in sorted(set(declared) & present)},
            "declared_but_absent": sorted(set(declared) - present),
            "excluded_on_purpose": explained,
            "undeclared": unexplained,
        }
        tif = ref / slide / manifest.VISIUMHD_PYR_TIF.path.format(slide=slide)
        entry["pyr_tif"] = ({**describe(tif), "policy": "tiff"} if tif.exists()
                            else {"missing": True})
        out["datasets"][patient] = entry
    return out


def record_xenium(ref: Path) -> dict:
    out: dict = {"family": "xenium", "datasets": {}}
    for key, slide in config.XENIUM_SLIDES.items():
        d = ref / slide.name / slide.sim_subdir
        if not d.exists():
            out["datasets"][key] = {"missing": True}
            continue
        declared = {f.path: f for f in manifest.xenium_files(key)}
        # only the top level of sim/ is ours; sub-folders like classif/ are experiments
        present = {rel for rel in walk(d) if "/" not in rel}
        excl = {k.format(code=slide.code): v for k, v in manifest.XENIUM_EXCLUDED.items()}
        explained, unexplained = classify(present - set(declared), excl)
        out["datasets"][key] = {
            "root": f"{slide.name}/{slide.sim_subdir}",
            "files": {rel: {**describe(d / rel), "policy": declared[rel].policy}
                      for rel in sorted(set(declared) & present)},
            "declared_but_absent": sorted(set(declared) - present),
            "excluded_on_purpose": explained,
            "undeclared": unexplained,
        }
    return out


RECORDERS = {"sthelar": record_sthelar, "visiumhd": record_visiumhd, "xenium": record_xenium}


def cmd_record(args) -> int:
    ref = Path(args.reference).resolve()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    bad = 0
    for family in args.families:
        rec = RECORDERS[family](ref)
        (MANIFEST_DIR / f"{family}.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(f"\n=== {family} ===")
        for name, entry in rec["datasets"].items():
            if entry.get("missing"):
                print(f"  {name}: FOLDER MISSING")
                bad += 1
                continue
            n = len(entry["files"])
            absent = entry["declared_but_absent"]
            undecl = entry["undeclared"]
            size = sum(f["bytes"] for f in entry["files"].values()) / 2**30
            print(f"  {name:<14} {n:>3} files, {size:6.1f} GiB"
                  f"{'  | declared-but-absent: ' + ', '.join(absent) if absent else ''}"
                  f"{'  | UNDECLARED: ' + ', '.join(undecl) if undecl else ''}")
            bad += len(absent) + len(undecl)
    print(f"\nmanifests written to {MANIFEST_DIR}")
    print("spec matches the reference exactly" if not bad else
          f"{bad} discrepancies between the spec and the reference -- fix manifest.py")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="record the reference folders")
    rec.add_argument("--reference", required=True,
                     help="the data root holding the folders to record")
    rec.add_argument("--families", nargs="+", default=list(RECORDERS), choices=list(RECORDERS))
    rec.set_defaults(fn=cmd_record)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
