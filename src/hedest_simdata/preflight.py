"""Does the current environment have what the planned stages will import?

Stages import their heavy dependencies **inside** the stage function, on purpose: a
download-only or CPU-only run should not have to have spatialdata, torch or StarDist
installed. The cost of that choice is that a missing package surfaces only when the stage
runs -- and by then the driver has submitted every task and a download may have spent an
hour. On 2026-11-08 a run was submitted into an environment without ``geopandas`` or
``spatialdata_io``: five tasks failed in one second each, but only after the 137 GB of
downloads they depended on had completed.

So the imports are checked statically, before anything is submitted. Nothing is imported --
:func:`importlib.util.find_spec` only locates a module -- so the check costs milliseconds
and cannot itself fail on a broken dependency.

Two things are deliberately **not** checked:

* stages whose :attr:`~hedest_simdata.pipeline.Resources.env` is not ``"core"``. Their body
  runs under a different interpreter, so this process cannot see what they have.
* ``hedest`` itself, which the :class:`~hedest_simdata.hedest_bridge.Bridge` puts on
  ``sys.path`` at run time from ``--hedest-dir``. The Bridge validates that separately and
  with a better error message than "no module named hedest".
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

#: Resolved by the Bridge at run time, not installed, so its absence means nothing here.
NOT_INSTALLED = frozenset({"hedest"})

PACKAGE_ROOT = Path(__file__).resolve().parent


def _stdlib_names() -> frozenset[str]:
    """Top-level standard-library module names.

    ``sys.stdlib_module_names`` is 3.10+, and this repo's environment is 3.10 -- but the
    check is also useful from whatever interpreter someone happens to have, and guessing
    wrong here would report half the standard library as missing. So fall back to reading
    the stdlib directory, which is accurate enough for the only thing this is used for:
    deciding what is *not* worth looking up.
    """
    names = getattr(sys, "stdlib_module_names", None)
    if names is not None:
        return frozenset(names)
    import sysconfig
    out = set(sys.builtin_module_names)
    stdlib = sysconfig.get_paths().get("stdlib")
    if stdlib and Path(stdlib).is_dir():
        for p in Path(stdlib).iterdir():
            if p.suffix == ".py" or (p.is_dir() and (p / "__init__.py").exists()):
                out.add(p.stem)
    return frozenset(out)


STDLIB = _stdlib_names()


def _module_files() -> dict[str, Path]:
    """Every module of this package, dotted name -> file, found without importing any."""
    out: dict[str, Path] = {}
    for p in PACKAGE_ROOT.rglob("*.py"):
        rel = p.relative_to(PACKAGE_ROOT.parent)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        out[".".join(parts)] = p
    return out


def _imports(path: Path, module: str,
             func: str | None = None) -> tuple[set[str], set[str]]:
    """``(third_party_top_level, first_party_modules)`` imported by ``path``.

    Function-body imports count: that is where the stages put theirs. Relative imports are
    resolved against ``module``, so ``from . import build`` inside
    ``hedest_simdata.xenium`` becomes ``hedest_simdata.xenium.build``.

    ``func`` narrows a family module to one stage: its module-level imports (which always
    run) plus that one function's. Without it every stage in a family would inherit every
    other stage's dependencies, and ``sthelar:download`` would be reported as needing
    geopandas -- which would be both wrong and the kind of noise that gets a check ignored.
    """
    third: set[str] = set()
    first: set[str] = set()
    pkg = module.split(".")

    def record(name: str) -> None:
        top = name.split(".")[0]
        if name.startswith("hedest_simdata"):
            first.add(name)
        elif top not in STDLIB and top not in NOT_INSTALLED:
            third.add(top)

    tree = ast.parse(path.read_text())
    if func is None:
        scope: list[ast.AST] = [tree]
    else:
        # module level, plus the one stage function; a stage body may also be nested in a
        # helper defined beside it, which the first-party hop below picks up anyway
        top = [n for n in tree.body if not isinstance(n, (ast.FunctionDef,
                                                          ast.AsyncFunctionDef))]
        wanted = [n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == func]
        scope = top + wanted
    for node in [n for root in scope for n in ast.walk(root)]:
        if isinstance(node, ast.Import):
            for a in node.names:
                record(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # `from . import x` inside a package's __init__ is that package's child;
                # inside a submodule it is a sibling, so one more level comes off.
                base = pkg if path.name == "__init__.py" else pkg[:-1]
                base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
                prefix = ".".join(base + ([node.module] if node.module else []))
                for a in node.names:
                    record(f"{prefix}.{a.name}" if not node.module else prefix)
            elif node.module:
                record(node.module)
    return third, first


def requirements(module: str, func: str | None = None) -> set[str]:
    """Third-party packages reachable from ``module`` (optionally just its ``func``).

    Transitive within this repo on purpose: ``xenium:spots`` imports no third-party module
    itself, it does ``from . import build`` -- and ``build`` is what needs ``spatialdata_io``.
    Only the first hop is narrowed to ``func``; once inside a helper module there is no way
    to tell which of its functions will be reached, so all of it counts.
    """
    files = _module_files()
    seen: set[str] = set()
    need: set[str] = set()
    queue = [(module, func)]
    while queue:
        name, only = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        path = files.get(name)
        if path is None:            # a first-party name that is a symbol, not a module
            continue
        third, first = _imports(path, name, only)
        need |= third
        queue += [(f, None) for f in first if f not in seen]
    return need


def missing(stages) -> dict[str, list[str]]:
    """``{stage name: [packages this environment cannot find]}`` for core-env stages."""
    out: dict[str, list[str]] = {}
    for stage in stages:
        if getattr(stage.resources, "env", "core") != "core":
            continue
        mod = getattr(stage.fn, "__module__", None)
        if not mod:
            continue
        gone = sorted(p for p in requirements(mod, getattr(stage.fn, "__name__", None))
                      if importlib.util.find_spec(p) is None)
        if gone:
            out[f"{stage.family}:{stage.name}"] = gone
    return out


def report(stages) -> str | None:
    """A message naming what to install, or ``None`` when the environment is complete."""
    gone = missing(stages)
    if not gone:
        return None
    every = sorted({p for ps in gone.values() for p in ps})
    lines = [f"this environment ({sys.executable}) cannot import "
             f"{len(every)} package(s) that planned stages need: " + ", ".join(every),
             "",
             "    stage                     missing",
             "    " + "-" * 60]
    lines += [f"    {name:<24}  {', '.join(ps)}" for name, ps in sorted(gone.items())]
    lines += ["",
              "Create the environment this repo declares and run from it:",
              "",
              "    CONDA_SOLVER=libmamba conda env create -f environment.yml",
              "    conda activate hedest-simdata",
              "",
              "Checked before submitting because these imports happen inside the stages, so "
              "without this they would fail only after everything they depend on had run."]
    return "\n".join(lines)
