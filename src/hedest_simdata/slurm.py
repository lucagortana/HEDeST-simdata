"""Submits the plan as one Slurm array per stage, chained on the stage dependencies.

An array per stage, one task per unit, and ``--dependency=aftercorr`` between stages so
each dataset flows through independently instead of waiting for the slowest one at every
step. That matters here: ``lymph_node_s0`` is four times the size of ``skin_s4``, and
lock-step would idle the cluster for most of the run.

Nothing about this site is hard-coded. The partitions, the conda root and any node
exclusions come from the environment, because they are the one thing that cannot be
guessed for someone else's cluster:

=========================  ===========================================================
``SIMDATA_CPU_PARTITION``  partition for the CPU stages (required for ``--slurm``)
``SIMDATA_GPU_PARTITION``  partition for the GPU stages (defaults to the CPU one)
``SIMDATA_CONDA_SH``       ``.../etc/profile.d/conda.sh`` to source
``SIMDATA_SBATCH_EXTRA``   anything else, verbatim, e.g. ``--exclude=node005,node009``
=========================  ===========================================================

Each stage's ``Resources.env`` names the interpreter it needs; the mapping to a real
environment comes from the ``--*-python`` flags, so the generated script activates nothing
it does not have to.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from .config import Layout
from .pipeline import Stage

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=simdata_{family}_{stage}
#SBATCH --output={logs}/{family}_{stage}_%A_%a.log
#SBATCH --error={logs}/{family}_{stage}_%A_%a.err
#SBATCH -p {partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G
#SBATCH --time={hours}:00:00
#SBATCH --array=0-{last}{concurrency}
{gres}{extra}
set -uo pipefail

UNITS=({units})
UNIT="${{UNITS[$SLURM_ARRAY_TASK_ID]}}"
echo "== {family}:{stage} $UNIT (array $SLURM_ARRAY_TASK_ID on $(hostname)) =="

{conda}
# The interpreter is called by absolute path rather than activated, so its own lib/ has
# to be put on the loader path by hand. `conda activate` does NOT do this -- it leaves
# LD_LIBRARY_PATH alone on purpose -- and without it the system libstdc++ can be loaded
# first by an early import, after which a conda library needing a newer GLIBCXX fails to
# load. Measured here: the `plots` stage died on `import openslide` ->
# libopenslide -> libicuuc.so.73 requiring GLIBCXX_3.4.30, which the system
# libstdc++.so.6 does not carry. Same reasoning as Bridge._run's `lib` argument.
{ld_library_path}
cd {repo}
# no `set -e`: the exit code has to be reported and then propagated deliberately, or a
# failing task looks like a successful one in the log and sacct disagrees with it.
{python} -u make_simdata.py {data_root} \\
    --families {family} --stages {stage} --datasets "$UNIT" \\
    --work-dir {work_dir} --keep-work {passthrough}
rc=$?
echo "PYTHON_RC=$rc"
exit $rc
"""


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def script_for(stage: Stage, units: list[str], layout: Layout, args,
               logs: Path, repo: Path) -> str:
    """The sbatch script for one stage. Returned as text so ``--dry-run`` can show it."""
    cpu_part = _env("SIMDATA_CPU_PARTITION")
    if not cpu_part:
        raise SystemExit(
            "--slurm needs SIMDATA_CPU_PARTITION (and SIMDATA_GPU_PARTITION for the GPU "
            "stages). They name one site's queues, so they are not shipped as defaults.")
    partition = (_env("SIMDATA_GPU_PARTITION", cpu_part) if stage.resources.gpu
                 else cpu_part)
    conda_sh = _env("SIMDATA_CONDA_SH")
    conda = f"source {conda_sh}" if conda_sh else "# no SIMDATA_CONDA_SH set"
    extra = _env("SIMDATA_SBATCH_EXTRA", "")

    # the interpreter the stage body needs; the core one is whatever submitted
    python = {"core": sys.executable,
              "hovernet": args.hovernet_python or sys.executable,
              "cellvit": args.cellvit_python or sys.executable,
              "bin2cell": args.bin2cell_python or sys.executable}[stage.resources.env]

    # <env>/bin/python -> <env>/lib, when that is really how the interpreter is laid out
    lib = Path(python).resolve().parent.parent / "lib"
    ld = (f"export LD_LIBRARY_PATH={shlex.quote(str(lib))}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
          if lib.is_dir() else f"# {lib} is not a directory; LD_LIBRARY_PATH left alone")

    # the bridge flags have to reach the job, which is a fresh process
    passthrough = " ".join(
        f"{flag} {shlex.quote(str(val))}"
        for flag, val in (("--hedest-dir", args.hedest_dir),
                          ("--hovernet-python", args.hovernet_python),
                          ("--hovernet-weights", args.hovernet_weights),
                          ("--cellvit-python", args.cellvit_python))
        if val)

    return TEMPLATE.format(
        family=stage.family, stage=stage.name, logs=logs, partition=partition,
        cpus=stage.resources.cpus, mem=stage.resources.mem_gb,
        hours=stage.resources.hours, last=len(units) - 1,
        concurrency=f"%{min(len(units), stage.resources.max_concurrent or 8)}",
        gres="#SBATCH --gres=gpu:1\n" if stage.resources.gpu else "",
        extra=f"#SBATCH {extra}\n" if extra else "",
        units=" ".join(shlex.quote(u) for u in units),
        conda=conda, repo=repo, python=shlex.quote(python), ld_library_path=ld,
        data_root=shlex.quote(str(layout.data_root)),
        work_dir=shlex.quote(str(layout.work_root)),
        passthrough=passthrough)


def submit(steps, layout: Layout, args) -> int:
    """Writes and submits one array per stage, chained with ``aftercorr``.

    ``aftercorr`` pairs array *task* n of a stage with task n of the stage before it, so
    it is only correct while consecutive stages iterate the same units in the same order.
    Where the scopes differ -- ``slides`` is per slide, ``crops`` per dataset -- that does
    not hold, so those two are chained with a whole-job ``afterok`` instead.
    """
    repo = Path(__file__).resolve().parents[2]
    logs = layout.work_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    scripts = layout.work_root / "sbatch"
    scripts.mkdir(parents=True, exist_ok=True)

    job_of: dict[str, str] = {}
    submitted: list[tuple[str, str, int]] = []
    for stage, units in steps:
        text = script_for(stage, units, layout, args, logs, repo)
        path = scripts / f"{stage.family}_{stage.name}.sbatch"
        path.write_text(text)
        path.chmod(0o755)

        cmd = ["sbatch"]
        deps = []
        for dep in stage.needs:
            key = f"{stage.family}:{dep}"
            if key not in job_of:
                continue
            dep_stage = next((s for s, _ in steps if s.family == stage.family
                              and s.name == dep), None)
            kind = ("aftercorr" if dep_stage is not None
                    and dep_stage.scope == stage.scope else "afterok")
            deps.append(f"{kind}:{job_of[key]}")
        if deps:
            cmd.append(f"--dependency={','.join(deps)}")
        cmd.append(str(path))

        if args.dry_run:
            print(" ".join(cmd))
            job_of[f"{stage.family}:{stage.name}"] = "PENDING"
            continue
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"sbatch failed for {stage.family}:{stage.name}: {out.stderr.strip()}")
            return 1
        job_id = out.stdout.strip().split()[-1]
        job_of[f"{stage.family}:{stage.name}"] = job_id
        submitted.append((f"{stage.family}:{stage.name}", job_id, len(units)))
        print(f"  submitted {stage.family}:{stage.name:<10} job {job_id} "
              f"({len(units)} tasks)")

    if submitted:
        ids = ",".join(j for _, j, _ in submitted)
        print(f"\n{len(submitted)} arrays, {sum(n for *_, n in submitted)} tasks")
        print(f"watch with:  squeue -j {ids}")
        print(f"logs in:     {logs}")
        print(f"\n--keep-work is set in every job, so {layout.work_root} survives; "
              "remove it yourself once the verify stages pass.")
    return 0
