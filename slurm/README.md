# Slurm

There are no job scripts to edit here. `make_simdata.py --slurm` generates one array per
stage into `<DATA_ROOT>/_work/sbatch/` and submits them, chained on the stage dependencies,
with the cpu/memory/time/GPU request each stage declares in its own definition.

```bash
export SIMDATA_CPU_PARTITION=<your cpu partition>
export SIMDATA_GPU_PARTITION=<your gpu partition>
export SIMDATA_CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh
export SIMDATA_SBATCH_EXTRA="--exclude=node005,node009"      # optional, verbatim

python make_simdata.py /path/to/data --slurm --dry-run       # print the scripts
python make_simdata.py /path/to/data --slurm                 # submit
```

## Why one array per stage, and why `aftercorr`

One array task per dataset, and consecutive stages chained with
`--dependency=aftercorr:<previous job>`, which pairs array *task n* with task *n* of the
stage before it. So each dataset flows `stage → hovernet → post → …` independently instead
of every dataset waiting for the slowest one at each step. That is not a micro-optimisation
here: `lymph_node_s0` is four times the size of `skin_s4`, and lock-step would leave the
cluster idle for most of the run.

`aftercorr` is only correct while consecutive stages iterate the same units in the same
order. Where the scopes differ — `sthelar:slides` is per *slide*, `sthelar:crops` per
*dataset* — that does not hold, so those are chained with a whole-job `afterok` instead.
`slurm.py` works that out from each stage's declared scope rather than being told.

## Resources

Each stage declares its own, because the pipeline is very uneven: the H&E exports want a big
CPU box for a couple of minutes, HoVerNet wants a GPU for an hour, the figures want four
cores for forty seconds. `--dry-run` prints the table. If your cluster needs different
numbers, they are the `Resources(...)` in each family's `__init__.py`, in one place per
stage.

## Concurrency

Each array is submitted with `%8`, so at most eight tasks of a stage run at once. Raise it
in `slurm.py` (`concurrency=`) if your queue allows more; the GPU stages are the ones worth
widening, since they dominate the wall clock.

## Logs, and what survives a failure

Logs go to `<DATA_ROOT>/_work/logs/<family>_<stage>_<jobid>_<task>.{log,err}`. Every
generated script ends with

```bash
rc=$?
echo "PYTHON_RC=$rc"
exit $rc
```

rather than `set -e`, so a failing task reports its code *and* fails in `sacct` — with
`set -e` the script would exit before the echo and the log would look like a success next to
a failed job.

Jobs always pass `--keep-work`, so `_work/` survives for inspection; remove it yourself once
the `verify` stages pass, or re-run the same command without `--slurm` to have it cleaned up.
