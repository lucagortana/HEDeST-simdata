"""Does ``environment.yml`` declare everything the stages import?

The environment had never been built when the first full run was submitted, and five tasks
died on `ModuleNotFoundError` -- after the 137 GB of downloads they depended on. The
environment is now built and validated, but "it worked once on this machine" is not the
property worth having: what matters is that the *declared* spec covers every import, so the
next person gets a working environment from the file rather than from luck.

This needs no environment at all. It reads the spec, walks the stages' imports with
:mod:`hedest_simdata.preflight`, and compares the two -- so it also catches a package that
is only present transitively. That is how ``h5py`` was found: imported directly by the
STHELAR and Visium HD stages, declared nowhere, and working only because ``anndata``
happens to depend on it.

    python tests/test_env_spec.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hedest_simdata import pipeline    # noqa: E402
from hedest_simdata import preflight   # noqa: E402

FAILURES: list[str] = []

#: Import name -> the conda/pip package that provides it, where the two differ. Only
#: packages this repo actually imports are listed; a new entry means a new direct import.
PROVIDES = {
    "PIL": "pillow",
    "cv2": "opencv",
    "skimage": "scikit-image",
    "matplotlib": "matplotlib-base",
    "torch": "pytorch",
    "openslide": "openslide-python",
    "spatialdata_io": "spatialdata-io",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
}


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def declared(path: Path) -> set[str]:
    """Package names in a conda env file, from both the conda list and the pip list.

    Parsed with a regex rather than a YAML loader so the test needs nothing installed --
    the point is to be runnable in an environment that is, precisely, wrong.
    """
    names = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("- ") or line.startswith("- pip:"):
            continue
        spec = line[2:].strip()
        m = re.match(r"^([A-Za-z0-9._-]+)", spec)
        if m:
            names.add(m.group(1).lower())
    return names


def main() -> int:
    registry = pipeline.load_families()
    stages = [s for f in pipeline.FAMILY_MODULES for s in registry.of(f)]
    core = Path(REPO / "environment.yml")
    spec = declared(core)

    print(f"\nenvironment.yml declares {len(spec)} packages")
    check("it is not empty, so this comparison means something", len(spec) > 10)
    check("it declares python, so it is the file it claims to be", "python" in spec)

    print("\nevery import of a core-env stage is declared")
    per_stage: dict[str, set[str]] = {}
    for s in stages:
        if getattr(s.resources, "env", "core") != "core":
            continue
        per_stage[f"{s.family}:{s.name}"] = preflight.requirements(
            s.fn.__module__, getattr(s.fn, "__name__", None))

    needed = sorted({m for r in per_stage.values() for m in r})
    check("the stages import something, so this is not vacuous", len(needed) > 5,
          f"{len(needed)} modules")
    for mod in needed:
        pkg = PROVIDES.get(mod, mod).lower()
        users = sorted(k for k, r in per_stage.items() if mod in r)
        check(f"{mod} (as {pkg})", pkg in spec,
              f"imported by {users[0]}" + (f" +{len(users) - 1} more" if len(users) > 1
                                           else ""))

    print("\na download-only run really does need only a download's dependencies")
    # The README promises that downloading needs none of the heavy stack, and the preflight
    # in make_simdata.py would turn that promise into a refusal if a download stage pulled
    # in geopandas or torch through a shared helper. This is what the per-stage narrowing in
    # preflight.requirements() buys, so it is worth asserting rather than assuming.
    heavy = {"torch", "spatialdata", "spatialdata_io", "geopandas", "skimage", "openslide"}
    for name, req in sorted(per_stage.items()):
        if not name.endswith(":download"):
            continue
        check(f"{name} needs none of the heavy stack", not (req & heavy),
              ", ".join(sorted(req)))

    print("\nthe foreign-environment stages are declared where they belong")
    foreign = {m for s in stages
               if getattr(s.resources, "env", "core") != "core"
               for m in preflight.requirements(s.fn.__module__,
                                               getattr(s.fn, "__name__", None))}
    extra = sorted(foreign - set(needed))
    check("the foreign stages need something the core ones do not", bool(extra),
          ", ".join(extra))
    b2c = declared(REPO / "envs" / "bin2cell.yml")
    for mod in extra:
        pkg = PROVIDES.get(mod, mod).lower()
        check(f"{mod} is declared in envs/bin2cell.yml", pkg in b2c)

    # Only the packages that are the *reason* for a second environment must stay out of the
    # core spec: bin2cell pins StarDist and TensorFlow, which is what cannot coexist with
    # the core pins. Overlap is otherwise expected and fine -- scanpy is declared in both
    # because both need it, and an earlier version of this check wrongly called that a
    # failure. "Needed by a foreign stage" does not mean "must be absent from core".
    for pkg in ("bin2cell", "stardist", "tensorflow"):
        check(f"{pkg} is absent from the core environment", pkg not in spec)

    print("\nkeyword arguments to third-party readers exist in the pinned versions")
    # `xenium(..., cell_boundaries=True)` -- singular -- is not a parameter of any
    # spatialdata-io this repo can install; the plural `cells_boundaries` is, and the
    # singular form is the element name in the written archive. Copied faithfully from the
    # original converter, it meant `xenium:convert` raised TypeError on its first line, and
    # nothing noticed for as long as the converted archive was carried over by hand.
    # Skipped rather than failed when the library is absent, and the skip is printed: a
    # check that quietly passes when it cannot run is worse than no check.
    checked = skipped = 0
    for mod_name, func_name, call in (
            ("spatialdata_io", "xenium", "xenium"),
            ("spatialdata_io.readers.xenium", "xenium_aligned_image",
             "xenium_aligned_image")):
        try:
            import importlib
            import inspect
            fn = getattr(importlib.import_module(mod_name), func_name)
        except Exception as exc:
            skipped += 1
            print(f"  skip  {mod_name}.{func_name} is not importable here ({type(exc).__name__})")
            continue
        sig = inspect.signature(fn)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            # **kwargs means any keyword is legal and usually forwarded to something else,
            # so the signature says nothing about correctness. bin2cell's `stardist` is
            # like this and an earlier version of this check called its `prob_thresh`
            # (a genuine StarDist argument) a bug.
            skipped += 1
            print(f"  skip  {mod_name}.{func_name} takes **kwargs, so any keyword is legal")
            continue
        allowed = set(sig.parameters)
        for p in sorted((REPO / "src").rglob("*.py")):
            import ast
            for node in ast.walk(ast.parse(p.read_text())):
                if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == call):
                    continue
                bad = sorted(k.arg for k in node.keywords if k.arg and k.arg not in allowed)
                checked += 1
                check(f"{p.relative_to(REPO)}:{node.lineno} calls {call}() with known "
                      f"keywords", not bad, ", ".join(bad) or f"{len(node.keywords)} kwargs")
    if skipped and not checked:
        print("  (no call could be verified -- run this from the hedest-simdata env)")

    print("\nevery call this package makes on its own modules names something real")
    # `visiumhd:embed` called `hovernet.load_nuc`, which the port invented; `hovernet`
    # offers `load`. It is the stage's last step, so finding out cost a full GPU pass --
    # 16 min for P5, with the 178,601 embeddings already written -- and it is the second
    # bug of exactly this shape, after a spawned `plot_hovernet_types.py` that did not
    # exist. No import error catches either, because the module imports fine; the
    # *attribute* is missing.
    #
    # So: parse every module, find `name.attr(...)` where `name` is a sibling module this
    # file imported from the package, and require `attr` to be defined there. Parsing
    # only, no imports, so it costs nothing and runs anywhere.
    import ast as _ast
    pkg = REPO / "src" / "hedest_simdata"
    defined: dict[str, set[str]] = {}
    for mod in pkg.rglob("*.py"):
        tree = _ast.parse(mod.read_text())
        names = {n.name for n in tree.body
                 if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef))}
        for n in tree.body:
            if isinstance(n, _ast.Assign):
                names |= {tgt.id for tgt in n.targets if isinstance(tgt, _ast.Name)}
            elif isinstance(n, _ast.AnnAssign) and isinstance(n.target, _ast.Name):
                names.add(n.target.id)
            elif isinstance(n, (_ast.Import, _ast.ImportFrom)):
                names |= {(a.asname or a.name).split(".")[0] for a in n.names}
        key = mod.stem if mod.name != "__init__.py" else mod.parent.name
        defined.setdefault(key, set()).update(names)

    seen = 0
    for mod in sorted(pkg.rglob("*.py")):
        tree = _ast.parse(mod.read_text())
        # which sibling modules this file pulled in, and under what local name
        local: dict[str, str] = {}
        for n in _ast.walk(tree):
            if isinstance(n, _ast.ImportFrom) and n.module in (None, "", "."):
                for a in n.names:
                    if a.name in defined:
                        local[a.asname or a.name] = a.name
        for n in _ast.walk(tree):
            if not (isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
                    and isinstance(n.func.value, _ast.Name)):
                continue
            target = local.get(n.func.value.id)
            if target is None:
                continue
            seen += 1
            check(f"{mod.relative_to(REPO)}:{n.lineno} calls "
                  f"{n.func.value.id}.{n.func.attr}(), which exists",
                  n.func.attr in defined[target], f"in {target}.py")
    print(f"  --    {seen} such call(s) checked")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES))
        return 1
    print("all environment-spec checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
