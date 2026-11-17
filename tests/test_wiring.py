"""Checks the parts of this repo that can be checked without any data.

Cheap to run and worth running: it catches a stage that references a path property that
does not exist, a dependency on a stage that was renamed, a manifest entry whose format
placeholders cannot be filled, and the kind of import cycle that only shows up when the
driver loads every family at once.

    python tests/test_wiring.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hedest_simdata import config      # noqa: E402
from hedest_simdata import manifest    # noqa: E402
from hedest_simdata import pipeline    # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def _raises_bridge_error() -> bool:
    """A Bridge with nothing configured must say so, naming the flag that configures it.

    ``HEDEST_DIR`` is cleared for the duration, and the Bridge built inside: the README
    tells users to export it, and a Bridge that found it there is a *configured* one, so
    without this the check failed for exactly the people who followed the instructions.
    """
    import os

    from hedest_simdata import hedest_bridge
    from hedest_simdata.hedest_bridge import BridgeError

    saved = os.environ.pop("HEDEST_DIR", None)
    try:
        hedest_bridge.Bridge().hedest
    except BridgeError as exc:
        return "--hedest-dir" in str(exc)
    except Exception:
        return False
    finally:
        if saved is not None:
            os.environ["HEDEST_DIR"] = saved
    return False


def main() -> int:
    layout = config.Layout.build("/tmp/hedest_simdata_wiring_check")
    registry = pipeline.load_families()

    print("\nthe stage graph resolves")
    for family in registry.families():
        stages = registry.resolve(family, None)
        names = [s.name for s in stages]
        check(f"{family}: {len(stages)} stages resolve in order", bool(stages),
              " -> ".join(names))
        pos = {n: i for i, n in enumerate(names)}
        for s in stages:
            for dep in s.needs:
                check(f"{family}:{s.name} needs {dep}, which exists and comes first",
                      dep in pos and pos[dep] < pos[s.name])

    print("\nevery stage has units and a resource request")
    for family in registry.families():
        for s in registry.of(family):
            units = pipeline.units(layout, s, None)
            check(f"{family}:{s.name} has units", bool(units), f"{len(units)} unit(s)")
            r = s.resources
            check(f"{family}:{s.name} asks for something sane",
                  r.cpus > 0 and r.mem_gb > 0 and r.hours > 0)
            check(f"{family}:{s.name} has a one-line description", bool(s.doc))

    # `Resources.env` picks the interpreter for the WHOLE driver process, not just for the
    # foreign tool, so leaving the core env costs a stage the core packages in its own
    # body. Only a stage that imports a foreign tool *in-process* has to pay that; one
    # that merely reaches a foreign env through the Bridge does not, because the Bridge
    # spawns that interpreter as a subprocess regardless of the parent env.
    #
    # Hence an allow-list rather than a rule derived from the body: it is one line per
    # exception, and both directions are checked so it cannot go stale. Worth having --
    # sthelar:cellvit declared env="cellvit", was not importing anything from it, and
    # died on `import anndata` in its own spot-proximity filter after nine minutes of GPU
    # work it then had to repeat.
    print("\nonly a stage importing a foreign tool in-process leaves the core env")
    in_process_foreign = {
        "visiumhd:bin2cell": "bin2cell + StarDist (and passes the Bridge down to HoVerNet)",
    }
    for family in registry.families():
        for s in registry.of(family):
            key = f"{family}:{s.name}"
            allowed = key in in_process_foreign
            check(f"{key}: env={s.resources.env!r}",
                  (s.resources.env != "core") == allowed,
                  in_process_foreign.get(key, ""))

    print("\n'already built?' predicates answer on an empty root")
    for family in registry.families():
        for s in registry.of(family):
            if s.done is None:
                continue
            unit = pipeline.units(layout, s, None)[0]
            check(f"{family}:{s.name}.done({unit}) is False on an empty root",
                  s.is_done(layout, unit) is False)

    print("\nthe HEDeST bridge is usable")
    from hedest_simdata import hedest_bridge
    b = hedest_bridge.Bridge()
    # Bridge is a mutable dataclass, so it is unhashable; anything that memoises on self
    # (lru_cache) breaks at the first call, and only a GPU stage would notice.
    try:
        hash(b)
        hashable = True
    except TypeError:
        hashable = False
    check("Bridge is unhashable, as a mutable dataclass is", not hashable)
    check("no bridge method memoises on self", not any(
        hasattr(getattr(type(b), n, None), "cache_info")
        for n in dir(type(b)) if not n.startswith("__")))
    check("an unconfigured bridge raises a readable error, not an AttributeError",
          _raises_bridge_error())

    print("\nthe manifest expands")
    for name in config.BENCH_NAMES:
        files = manifest.sthelar_files(name, 5)
        check(f"sthelar:{name} declares files", len(files) > 20, f"{len(files)} files")
        check(f"sthelar:{name} has no unfilled placeholder",
              not any("{" in f.path for f in files))
    for p in config.VISIUMHD_PATIENTS:
        files = manifest.visiumhd_files(p)
        check(f"visiumhd:{p} declares files", len(files) == 30, f"{len(files)} files")
        check(f"visiumhd:{p} has no unfilled placeholder",
              not any("{" in f.path for f in files))
    for k in config.XENIUM_SLIDES:
        files = manifest.xenium_files(k)
        check(f"xenium:{k} declares files", len(files) == 6, f"{len(files)} files")
        check(f"xenium:{k} has no unfilled placeholder",
              not any("{" in f.path for f in files))

    print("\nthe registries agree with themselves")
    check("every crop names a known slide",
          all(src in config.STHELAR_SLIDES for src, _ in config.STHELAR_CROPS.values()))
    check("the 8 bench datasets are the 6 crops plus the 2 as-is slides",
          len(config.BENCH_NAMES) == 8
          and set(config.BENCH_NAMES) == set(config.STHELAR_CROPS) | set(config.STHELAR_ASIS))
    check("every Visium HD patient has an unassigned-types entry",
          all(p in config.VISIUMHD_UNASSIGNED_TYPES for p in config.VISIUMHD_PATIENTS))
    for k, s in config.XENIUM_SLIDES.items():
        check(f"xenium:{k} spot circles do not overlap", s.diameter < s.spacing,
              f"d={s.diameter:g} < pitch={s.spacing:g}")
        check(f"xenium:{k} has 7 cell types and no overlap with its drops",
              len(s.cell_types) == 7 and not set(s.cell_types) & set(s.drop_types))
        check(f"xenium:{k} uses a known matching route", s.match in {"intersect", "closest"})

    print("\nthe download manifest covers every unit")
    from hedest_simdata.common import sources
    root = layout.data_root
    for slide in config.STHELAR_SLIDES:
        srcs = sources.sthelar(root, slide)
        check(f"sthelar:{slide} has 2 sources, both with a URL",
              len(srcs) == 2 and all(s.url for s in srcs))
    for p in config.VISIUMHD_PATIENTS:
        check(f"visiumhd:{p} lists its sources",
              len(sources.visiumhd(root, config.visiumhd_name(p), p)) >= 3)
    for k, s in config.XENIUM_SLIDES.items():
        check(f"xenium:{k} lists its sources", len(sources.xenium(root, k, s.name)) >= 3)

    print("\n" + ("all wiring checks passed" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {', '.join(FAILURES[:6])}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
