"""Checks the foreign command lines this repo builds, without running anything.

Two bugs that a full run only revealed after hours of GPU time motivated this file, and
both were visible in the argv alone:

* ``hovernet_infer`` passed neither ``--image_dict_path`` nor ``--skip_image_dict`` when
  no image dict was wanted. ``run_infer.py`` accepts that, segments the whole slide, writes
  the json *and* the geojson, and only then raises. The cost of forgetting was a complete
  GPU pass (32 min for Xenium lung, 54 for breast) before the error appeared, and only the
  Xenium stages passed ``image_dict=None`` -- the Visium HD path has its own copy of the
  command, which did pass the flag, so nothing compared the two.
* the Visium HD HoVerNet cache defaulted to a hardcoded ``/cluster/CBIO/...`` path shared
  by every sample, so two concurrent samples mmapped one ``pred_map.npy`` and both died
  with SIGBUS. Upstream supplied a per-sample path from a wrapper script, which the port
  did not carry over.

So: build the argv and assert the invariants, and refuse absolute site paths repo-wide --
this repo is going to GitHub, where ``/cluster/CBIO/data1/lgortana`` means nothing.

    python tests/test_bridge_cmd.py
"""
from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hedest_simdata import hedest_bridge as hb    # noqa: E402

FAILURES: list[str] = []

#: Any string literal naming one of these is a path that only exists on the machine this
#: was developed on. Docstrings and comments may of course discuss them.
SITE_ROOTS = ("/cluster/", "/home/lgortana", "~/anaconda3")


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def _code_strings(path: Path) -> list[ast.Constant]:
    """Every string literal in ``path`` that is not a docstring.

    Docstrings name real paths and real filenames on purpose -- measurements, provenance,
    prose about the tools -- so both checks below have to look at code only. Comments never
    reach the AST, so they are excluded for free.
    """
    tree = ast.parse(path.read_text())
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                docs.add(id(first.value))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and isinstance(n.value, str) and id(n) not in docs]


def _fake_hedest(root: Path) -> Path:
    """The minimum layout :class:`Bridge` insists on before it will resolve anything."""
    (root / "hedest").mkdir(parents=True, exist_ok=True)
    hv = root / "external" / "hovernet"
    hv.mkdir(parents=True, exist_ok=True)
    (hv / "run_infer.py").write_text("")
    (hv / "type_info.json").write_text("{}")
    return root


def _capture(bridge: hb.Bridge) -> list[list[str]]:
    """Replace the subprocess call with a recorder, so no foreign tool is needed."""
    seen: list[list[str]] = []
    bridge._run = lambda cmd, lib=None: seen.append([str(c) for c in cmd])  # type: ignore
    return seen


def main() -> int:
    print("\nhovernet_infer always decides the image-dict question")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        hedest = _fake_hedest(tmp / "HEDeST")
        py = tmp / "env" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
        weights = tmp / "weights.tar"
        weights.write_text("")
        bridge = hb.Bridge(hedest_dir=hedest, hovernet_python=py, hovernet_weights=weights)

        for label, image_dict, expect, forbid in (
            ("without an image dict", None, "--skip_image_dict", "--image_dict_path"),
            ("with an image dict", tmp / "crops.pt", "--image_dict_path", "--skip_image_dict"),
        ):
            seen = _capture(bridge)
            bridge.hovernet_infer(wsi_dir=tmp / "wsi", mask_dir=tmp / "mask",
                                  out_dir=tmp / "out", mpp=0.25,
                                  cache_dir=tmp / "unit" / "cache", image_dict=image_dict)
            argv = seen[0]
            joined = " ".join(argv)
            check(f"{label}: passes {expect}", any(a.startswith(expect) for a in argv),
                  joined[-90:])
            check(f"{label}: does not also pass {forbid}",
                  not any(a.startswith(forbid) for a in argv))
            # the whole point: exactly one of the two, never neither
            got = sum(a.startswith(f) for a in argv
                      for f in ("--image_dict_path", "--skip_image_dict"))
            check(f"{label}: exactly one of the two flags", got == 1, f"{got}")

    print("\nwhere HoVerNet writes its json agrees with the flags we pass it")
    # infer/wsi.py: `output_dir/json/<name>.json` if --save_mask or --save_thumb, else
    # `output_dir/<name>.json`. The two have to be decided together, and were not: the
    # Xenium paths expected the json/ subdirectory while the bridge asked for neither flag,
    # so the file was never found and the stage could not be marked done. An empty json/
    # directory is created regardless, which is what made the wrong answer look right.
    from hedest_simdata import config                 # noqa: E402
    from hedest_simdata.xenium import paths as xpaths  # noqa: E402
    with tempfile.TemporaryDirectory() as d:
        layout = config.Layout.build(str(Path(d) / "data"), None)
        argv_has_subdir_flags = any(
            f in a for a in argv for f in ("--save_mask", "--save_thumb"))
        check("the bridge asks for neither --save_mask nor --save_thumb",
              not argv_has_subdir_flags)
        for key in config.XENIUM_SLIDES:
            sl = xpaths.slide(layout, key)
            check(f"xenium:{key} expects the flat json, matching those flags",
                  "json" not in sl.seg_json.relative_to(sl.seg_dir).parts[:-1],
                  str(sl.seg_json.relative_to(sl.seg_dir)))

    print("\nevery other place that builds a run_infer.py command decides it too")
    # the Visium HD path has its own copy of this command line; a second copy is exactly how
    # the bug above stayed invisible, so hold every copy to the same rule
    # a builder names run_infer.py in *code*; hovernet_stage.py and paths.py only discuss it
    builders = sorted(p for p in (REPO / "src").rglob("*.py")
                      if any("run_infer.py" in n.value for n in _code_strings(p)))
    check("at least two builders exist, so this check has something to compare",
          len(builders) >= 2, f"{[p.name for p in builders]}")
    for p in builders:
        check(f"{p.relative_to(REPO)} decides the image-dict question",
              any(f in n.value for n in _code_strings(p)
                  for f in ("skip_image_dict", "image_dict_path")))

    print("\nevery script this repo spawns is actually in this repo")
    # `visiumhd:bin2cell` spawned plot_hovernet_types.py, which the port never brought
    # across: the stage ran a full hour of StarDist and HoVerNet and then died with
    # "can't open file ... plot_hovernet_types.py" on its last step. Checking that a
    # referenced script exists costs nothing and does not need the script to run.
    # `external/...` paths are excluded: they live in the HEDeST checkout, which the Bridge
    # resolves at run time and validates there.
    # Matched by shape, not by looking for ".py" anywhere: the first version of this check
    # flagged error-message prose and the HEDeST-side scripts as well, 11 false positives,
    # and a check that cries wolf gets switched off. What is actually meant is a path built
    # next to the module -- `Path(__file__).parent / "script.py"` -- which is this repo
    # saying "a file of mine", and is exactly how the missing script was referenced.
    spawned = 0
    for p in sorted((REPO / "src").rglob("*.py")):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                    and isinstance(node.right, ast.Constant)
                    and isinstance(node.right.value, str)
                    and node.right.value.endswith(".py")):
                continue
            if not any(isinstance(n, ast.Name) and n.id == "__file__"
                       for n in ast.walk(node.left)):
                continue
            spawned += 1
            name = node.right.value
            check(f"{p.relative_to(REPO)}:{node.lineno} builds a path to {name}, "
                  f"which exists", (p.parent / name).exists())
    # Stated rather than asserted: there are none left in the tree today, so this is a
    # dormant regression guard, and pretending it passed a check would be the same kind of
    # lie as the figure comparison that only looked at image dimensions.
    print(f"  --    {spawned} sibling-script reference(s) in the tree"
          + ("; nothing to check here until one is added" if not spawned else ""))

    print("\nno source hardcodes a path from the machine this was written on")
    # This file is skipped: it is the one that spells out the patterns being looked for.
    scanned, offenders = 0, 0
    for p in sorted((REPO / "src").rglob("*.py")) + sorted(REPO.glob("tests/*.py")):
        if p.name == Path(__file__).name:
            continue
        scanned += 1
        bad = [f"line {n.lineno}: {n.value!r}" for n in _code_strings(p)
               if any(r in n.value for r in SITE_ROOTS)]
        if bad:
            offenders += 1
            check(f"{p.relative_to(REPO)} has no site path in code", False, "; ".join(bad))
    check(f"all {scanned} source files are free of site paths", offenders == 0)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES))
        return 1
    print("all bridge command checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
