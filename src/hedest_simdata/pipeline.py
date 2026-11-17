"""The stage graph: what can run, what it needs, and how much of a machine it wants.

Every family declares its stages here rather than shipping its own launcher, so one
command builds everything and ``--stages`` means the same thing across the three. A stage
is a pure function of ``(Layout, unit)`` -- ``unit`` being a dataset, a slide or a patient,
depending on ``scope`` -- and declares what must have run before it. The driver resolves
the order; nothing calls anything else directly.

Resources are declared because the three families are wildly uneven: the H&E TIFF exports
want a big CPU box for an hour, HoVerNet wants a GPU for the same, and the plots want four
cores for two minutes. One set of Slurm defaults for all of them would either waste a GPU
or get a job killed.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Iterable

from .config import Layout

#: What a stage iterates over.
SCOPES = ("slide", "dataset", "patient", "xenium")


@dataclass(frozen=True)
class Resources:
    """What one task of a stage needs. Used only by the Slurm backend."""

    cpus: int = 4
    mem_gb: int = 32
    gpu: bool = False
    hours: int = 4
    #: Interpreter to run the stage with, when it is not the core environment. Named
    #: rather than pathed: ``make_simdata.py`` maps the name to a ``--*-python`` flag.
    #:
    #: It is the interpreter for the stage's **own body**, not merely for the foreign
    #: tool, so set it only where that tool is imported *in-process* --
    #: ``visiumhd:bin2cell`` is the only such stage. A stage that reaches a foreign env
    #: through :class:`~hedest_simdata.hedest_bridge.Bridge` stays ``"core"``: the Bridge
    #: spawns the other interpreter as a subprocess, while the stage's own code still
    #: needs the core packages around it. ``tests/test_wiring.py`` enforces that.
    env: str = "core"

    #: Cap on how many of this stage's tasks Slurm may run at once, when fewer than the
    #: default. Only ``visiumhd:bin2cell`` sets it, and the reason is worth keeping: its
    #: HoVerNet call uses 8 inference and 16 post-processing workers, which are HoVerNet's
    #: own defaults and what the delivered Visium HD data was built with. Lowering them --
    #: as :meth:`~hedest_simdata.hedest_bridge.Bridge.hovernet_infer` does for every other
    #: caller, to stay inside ``/dev/shm`` and the semaphore limit -- would change how
    #: multi-worker post-processing assigns cell ids, and so would stop ``hovernet.json``
    #: reproducing byte for byte. Serialising the stage keeps the worker counts the data
    #: was made with and still avoids two of them sharing one node's shared memory.
    max_concurrent: int | None = None


@dataclass(frozen=True)
class Stage:
    name: str
    family: str
    scope: str
    fn: Callable[[Layout, str], None]
    needs: tuple[str, ...] = ()
    resources: Resources = field(default_factory=Resources)
    doc: str = ""
    #: Has this unit already been built? Lets a re-run skip finished work, which matters
    #: because a full build is days long and will be interrupted. A stage without one is
    #: always re-run, which is the safe default -- so this is added per stage, and only
    #: where "finished" can be decided without trusting a flag file.
    done: Callable[[Layout, str], bool] | None = None

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise ValueError(f"{self.name}: bad scope {self.scope!r}")

    def is_done(self, layout: Layout, unit: str) -> bool:
        if self.done is None:
            return False
        try:
            return bool(self.done(layout, unit))
        except Exception:
            # a `done` check that cannot run means "not done", never "done"
            return False


class Registry:
    """All stages of all families, in declaration order."""

    def __init__(self) -> None:
        self._stages: dict[str, Stage] = {}

    def add(self, stage: Stage) -> Stage:
        key = f"{stage.family}:{stage.name}"
        if key in self._stages:
            raise ValueError(f"duplicate stage {key}")
        self._stages[key] = stage
        return stage

    def stage(self, family: str, name: str, scope: str, needs: Iterable[str] = (),
              resources: Resources | None = None, doc: str = "",
              done: Callable[[Layout, str], bool] | None = None):
        """Decorator form: ``@registry.stage("sthelar", "crops", "dataset")``."""
        def deco(fn: Callable[[Layout, str], None]) -> Callable[[Layout, str], None]:
            self.add(Stage(name=name, family=family, scope=scope, fn=fn,
                           needs=tuple(needs), resources=resources or Resources(),
                           doc=doc or (fn.__doc__ or "").strip().split("\n")[0],
                           done=done))
            return fn
        return deco

    def of(self, family: str) -> list[Stage]:
        return [s for s in self._stages.values() if s.family == family]

    def get(self, family: str, name: str) -> Stage:
        try:
            return self._stages[f"{family}:{name}"]
        except KeyError:
            known = ", ".join(s.name for s in self.of(family)) or "(none registered)"
            raise KeyError(f"no stage {name!r} for {family}; known: {known}") from None

    def families(self) -> list[str]:
        out: list[str] = []
        for s in self._stages.values():
            if s.family not in out:
                out.append(s.family)
        return out

    def resolve(self, family: str, names: Iterable[str] | None) -> list[Stage]:
        """The requested stages plus their prerequisites, in a runnable order.

        Prerequisites are included because a half-built dataset is worse than an
        unbuilt one: asking for ``plots`` without ``spots`` would draw figures of
        stale data and report success.
        """
        declared = self.of(family)
        order = {s.name: i for i, s in enumerate(declared)}
        wanted = [s.name for s in declared] if names is None else list(names)
        seen: set[str] = set()
        out: list[str] = []

        def visit(name: str, chain: tuple[str, ...] = ()) -> None:
            if name in seen:
                return
            if name in chain:
                raise ValueError(f"cycle in {family} stages: {' -> '.join(chain + (name,))}")
            stage = self.get(family, name)
            for dep in stage.needs:
                visit(dep, chain + (name,))
            seen.add(name)
            out.append(name)

        for name in wanted:
            visit(name)
        return [self.get(family, n) for n in sorted(out, key=lambda n: order[n])]


REGISTRY = Registry()


#: The family modules, in the order the driver runs them.
FAMILY_MODULES = ("sthelar", "visiumhd", "xenium")


def load_families() -> Registry:
    """Imports the family modules so their stages register themselves."""
    import importlib
    for name in FAMILY_MODULES:
        importlib.import_module(f".{name}", __package__)
    return REGISTRY


def set_bridge(bridge) -> None:
    """Hands every family the resolved HEDeST locations.

    The bridge carries command-line overrides, so it cannot be a module constant; it is
    injected once here rather than threaded through every stage signature, since a stage
    that does not touch HEDeST should not have to take it.
    """
    import importlib
    for name in FAMILY_MODULES:
        importlib.import_module(f".{name}", __package__).BRIDGE = bridge


def units(layout: Layout, stage: Stage, only: Iterable[str] | None = None) -> list[str]:
    """The units one stage runs over, filtered by ``--datasets``."""
    from . import config
    table = {
        "slide": config.STHELAR_SLIDES,
        "dataset": config.BENCH_NAMES,
        "patient": config.VISIUMHD_PATIENTS,
        "xenium": list(config.XENIUM_SLIDES),
    }
    all_units = table[stage.scope]
    if only is None:
        return list(all_units)
    keep = set(only)
    # a crop's slide is implied: asking for lung_s3 must also build the slide it crops
    if stage.scope == "slide":
        implied = {config.STHELAR_CROPS[d][0] for d in keep if d in config.STHELAR_CROPS}
        keep |= implied
    return [u for u in all_units if u in keep]
