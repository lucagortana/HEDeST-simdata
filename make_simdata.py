#!/usr/bin/env python3
"""Build every simulated dataset HEDeST is validated on.

    python make_simdata.py /path/to/data

That is the whole interface: one folder, and the three families appear inside it. The
raw downloads go there too, so the command is also the download. Everything transient
goes to ``<DATA_ROOT>/_work`` and is removed when a family finishes.

    python make_simdata.py DATA --dry-run                  # the plan, nothing run
    python make_simdata.py DATA --families visiumhd        # one family
    python make_simdata.py DATA --datasets skin_s4 lung_s3 # some datasets
    python make_simdata.py DATA --stages download slides   # some stages (+ their deps)
    python make_simdata.py DATA --slurm                    # submit instead of running

``--stages`` always pulls in what a stage needs: asking for ``plots`` without ``spots``
would draw figures of stale data and report success, so it is not allowed. Normally those
prerequisites are then pruned for being finished, so ``--stages verify`` on a built folder
runs one task -- but ``--force`` turns the pruning off, and the two together re-run the
whole chain including its GPU stages. To redo one stage only, delete what it wrote and omit
``--force``.

Some stages need HEDeST and its environments; see ``hedest_simdata.hedest_bridge`` and
the README. They are resolved when a stage that needs them runs, so a CPU-only or
download-only run needs none of it configured.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from hedest_simdata import config          # noqa: E402
from hedest_simdata import hedest_bridge   # noqa: E402
from hedest_simdata import pipeline        # noqa: E402
from hedest_simdata import preflight       # noqa: E402


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="make_simdata.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root", help="where the simulated data subfolders go")
    ap.add_argument("--families", nargs="+", default=None,
                    choices=["sthelar", "visiumhd", "xenium"])
    ap.add_argument("--datasets", nargs="+", default=None, metavar="NAME",
                    help="dataset / slide / patient names to restrict to; a crop also "
                         "builds the slide it comes from")
    ap.add_argument("--stages", nargs="+", default=None, metavar="STAGE",
                    help="stages to run, plus whatever they depend on")
    ap.add_argument("--work-dir", default=None,
                    help="where intermediates go (default: <DATA_ROOT>/_work)")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the intermediates instead of deleting them on success")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, with resources, and stop")
    ap.add_argument("--force", action="store_true",
                    help="rebuild stages whose output is already complete")
    ap.add_argument("--slurm", action="store_true",
                    help="submit one array per stage instead of running in the foreground")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the check that this environment can import what the stages need")

    g = ap.add_argument_group("HEDeST and its environments")
    g.add_argument("--hedest-dir", default=None, help="the HEDeST checkout ($HEDEST_DIR)")
    g.add_argument("--hovernet-python", default=None,
                   help="interpreter of HEDeST's hovernet env ($HOVERNET_PYTHON)")
    g.add_argument("--hovernet-weights", default=None,
                   help="hovernet_fast_pannuke_type_tf2pytorch.tar ($HOVERNET_WEIGHTS)")
    g.add_argument("--cellvit-python", default=None,
                   help="interpreter of HEDeST's cellvit env ($CELLVIT_PYTHON)")
    g.add_argument("--bin2cell-python", default=None,
                   help="interpreter of the bin2cell env ($BIN2CELL_PYTHON)")
    return ap.parse_args(argv)


def plan(registry: pipeline.Registry, layout: config.Layout, args
         ) -> tuple[list[tuple[pipeline.Stage, list[str]]], int]:
    """The (stage, units) list that would run, in order, family by family.

    Units a stage has already finished are dropped unless ``--force``: a full build runs
    for days and will be interrupted, so re-running the same command has to resume rather
    than start again. The count of skipped units is returned so the plan can say so.
    """
    out: list[tuple[pipeline.Stage, list[str]]] = []
    skipped = 0
    families = args.families or registry.families()
    for family in families:
        stages = registry.resolve(family, args.stages)
        todo = {s.name: pipeline.units(layout, s, args.datasets) for s in stages}

        if not args.force:
            for s in stages:
                done = [u for u in todo[s.name] if s.is_done(layout, u)]
                skipped += len(done)
                todo[s.name] = [u for u in todo[s.name] if u not in set(done)]

            # A prerequisite only has to run if something that still needs it is going to
            # run. Walking backwards: a stage whose every consumer is already finished has
            # nothing to produce, so re-running it would redo hours of work to feed
            # nobody. This is what makes `--stages verify` on a finished build do nothing
            # instead of starting from the download.
            consumers: dict[str, list[str]] = {s.name: [] for s in stages}
            for s in stages:
                for dep in s.needs:
                    if dep in consumers:
                        consumers[dep].append(s.name)
            for s in reversed(stages):
                if not todo[s.name] or not consumers[s.name]:
                    continue
                if all(not todo[c] for c in consumers[s.name]):
                    skipped += len(todo[s.name])
                    todo[s.name] = []

        for s in stages:
            if todo[s.name]:
                out.append((s, todo[s.name]))
    return out, skipped


def show_plan(steps, layout: config.Layout, skipped: int = 0) -> None:
    print(f"data root : {layout.data_root}")
    print(f"work dir  : {layout.work_root}")
    total = sum(len(u) for _, u in steps)
    done = f", {skipped} already done (--force to redo)" if skipped else ""
    print(f"\n{len(steps)} stages, {total} tasks{done}\n")
    print(f"  {'stage':<22} {'tasks':>5}  {'cpus':>4} {'mem':>6} {'gpu':>4} "
          f"{'env':<9} what")
    print("  " + "-" * 100)
    for stage, units in steps:
        r = stage.resources
        print(f"  {stage.family + ':' + stage.name:<22} {len(units):>5}  "
              f"{r.cpus:>4} {str(r.mem_gb) + 'G':>6} {'yes' if r.gpu else '-':>4} "
              f"{r.env:<9} {stage.doc}")
    gpu = sum(len(u) for s, u in steps if s.resources.gpu)
    print(f"\n  {gpu} of {total} tasks need a GPU")


def run(steps, layout: config.Layout) -> int:
    failures: list[str] = []
    for stage, units in steps:
        for unit in units:
            tag = f"{stage.family}:{stage.name}[{unit}]"
            t0 = time.time()
            print(f"\n=== {tag} ===", flush=True)
            try:
                stage.fn(layout, unit)
            except Exception as exc:                      # one unit must not stop the rest
                failures.append(f"{tag}: {type(exc).__name__}: {exc}")
                print(f"!!! {tag} FAILED: {type(exc).__name__}: {exc}", flush=True)
            else:
                print(f"--- {tag} done in {time.time() - t0:.0f}s", flush=True)
    if failures:
        print(f"\n{len(failures)} task(s) failed:")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


def cleanup(layout: config.Layout, keep: bool) -> None:
    if keep or not layout.work_root.exists():
        return
    size = sum(p.stat().st_size for p in layout.work_root.rglob("*") if p.is_file())
    shutil.rmtree(layout.work_root)
    print(f"\nremoved {layout.work_root} ({size / 2**30:.1f} GiB of intermediates); "
          "pass --keep-work to keep it")


def main(argv=None) -> int:
    args = parse(argv)
    layout = config.Layout.build(args.data_root, args.work_dir)
    registry = pipeline.load_families()

    bridge = hedest_bridge.Bridge(
        hedest_dir=args.hedest_dir, hovernet_python=args.hovernet_python,
        hovernet_weights=args.hovernet_weights, cellvit_python=args.cellvit_python)
    pipeline.set_bridge(bridge)

    steps, skipped = plan(registry, layout, args)
    if not steps:
        print(f"nothing to do for that selection"
              + (f" ({skipped} units already built; --force to redo)"
                 if skipped else ""))
        return 0
    show_plan(steps, layout, skipped)

    # Before anything is submitted or downloaded: can this interpreter import what the
    # planned stages will reach for? They import inside the stage body on purpose, so a
    # missing package would otherwise surface one stage at a time, after its inputs had
    # already been built. Checked statically, so it costs milliseconds.
    if not args.no_preflight:
        complaint = preflight.report([s for s, _ in steps])
        if complaint:
            print("\n" + complaint)
            print("\n(--no-preflight runs anyway, if you know better than this check)")
            return 1

    if args.slurm:
        # slurm.submit honours --dry-run itself, printing the sbatch commands and the
        # scripts it would submit; checking dry_run first would hide them.
        from hedest_simdata import slurm
        layout.work_root.mkdir(parents=True, exist_ok=True)
        return slurm.submit(steps, layout, args)
    if args.dry_run:
        return 0

    layout.data_root.mkdir(parents=True, exist_ok=True)
    layout.work_root.mkdir(parents=True, exist_ok=True)
    rc = run(steps, layout)
    if rc == 0:
        cleanup(layout, args.keep_work)
    else:
        print(f"\n{layout.work_root} kept, because something failed")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
