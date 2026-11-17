"""The only module that reaches into HEDeST. Everything site-specific is here.

HEDeST ships the segmentation and feature-extraction code this repo needs, so none of it
is duplicated: ``external/hovernet`` (Otsu tissue mask + WSI inference),
``external/CellViT`` (the second segmentation) and ``hedest.features.hoptimus``
(H-Optimus-0 tile inference with per-cell pooling).

The dependency runs one way only. HEDeST must stay self-contained, so nothing here is ever
imported from HEDeST -- if a helper is needed on both sides it gets copied into HEDeST, not
shared from here.

**Three environments, not one.** HoVerNet, CellViT and the rest have mutually incompatible
dependency sets, so the first two are driven as subprocesses through their own
interpreters, exactly as the original pipelines did. Only H-Optimus-0 is imported into our
own process. Set them with environment variables or the ``make_simdata.py`` flags:

====================  ============================  ==========================================
variable              flag                          what
====================  ============================  ==========================================
``HEDEST_DIR``        ``--hedest-dir``              the HEDeST checkout
``HOVERNET_PYTHON``   ``--hovernet-python``         interpreter of HEDeST's hovernet env
``HOVERNET_WEIGHTS``  ``--hovernet-weights``        ``hovernet_fast_pannuke_type_tf2pytorch.tar``
``CELLVIT_PYTHON``    ``--cellvit-python``          interpreter of HEDeST's cellvit env
====================  ============================  ==========================================

Each is resolved lazily and only when a stage actually needs it, so a CPU-only run never
has to have a GPU environment configured.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

#: HoVerNet *fast* mode predicts a 164 px output from a 256 px input, so it cannot segment
#: within (256-164)/2 = 46 px of a real image edge. Callers reflect-pad by this much,
#: then shift the coordinates back, so no cell is lost at a crop boundary.
HOVERNET_PAD_PX = 64

#: Pyramid level the Otsu tissue mask is computed at.
MASK_LEVEL = 3

#: Cell-crop geometry used throughout the project: 64 px covering 20 um.
CELL_CROP_PX = 64
CELL_CROP_UM = 20


class BridgeError(RuntimeError):
    """A HEDeST path or environment is missing or not where it was said to be."""


def _resolve(flag: str, env: str, value: str | Path | None, *, must_exist: bool) -> Path:
    if value is None:
        value = os.environ.get(env)
    if value is None:
        raise BridgeError(
            f"{flag} is not set. Pass {flag} to make_simdata.py or export {env}=...")
    p = Path(value).expanduser()
    if must_exist and not p.exists():
        raise BridgeError(f"{flag} points at {p}, which does not exist")
    return p


@dataclass
class Bridge:
    """Resolved locations of everything this repo borrows."""

    hedest_dir: Path | None = None
    hovernet_python: Path | None = None
    hovernet_weights: Path | None = None
    cellvit_python: Path | None = None
    #: Only needed by the Visium HD segmentation step, which activates the environment in
    #: a shell rather than calling its interpreter: bin2cell's HoVerNet path was built and
    #: tested that way, and ``conda activate`` also applies any env-var hooks the
    #: environment defines. Defaults to ``$SIMDATA_CONDA_SH``.
    conda_sh: Path | None = None

    #: HEDeST's embedding function, imported once and cached here. Deliberately an
    #: instance field rather than ``functools.lru_cache`` on the method -- see
    #: :meth:`_load_hoptimus`.
    _hoptimus: object = field(default=None, repr=False, compare=False)

    # --- derived locations -------------------------------------------------------------

    @property
    def hovernet_repo(self) -> Path:
        """HEDeST's vendored HoVerNet, which both segmentation paths run from."""
        p = self.hedest / "external" / "hovernet"
        if not (p / "run_infer.py").exists():
            raise BridgeError(f"{p} does not hold run_infer.py")
        return p

    @property
    def hovernet_env(self) -> Path:
        """Root of the hovernet environment, from its interpreter path."""
        return self._hovernet_py().parent.parent

    def _conda_sh(self) -> Path:
        return _resolve("--conda-sh", "SIMDATA_CONDA_SH", self.conda_sh, must_exist=True)

    # --- resolution -------------------------------------------------------------------

    @property
    def hedest(self) -> Path:
        p = _resolve("--hedest-dir", "HEDEST_DIR", self.hedest_dir, must_exist=True)
        if not (p / "hedest").is_dir():
            raise BridgeError(f"{p} does not look like a HEDeST checkout (no hedest/ package)")
        return p

    def _hovernet_py(self) -> Path:
        return _resolve("--hovernet-python", "HOVERNET_PYTHON", self.hovernet_python,
                        must_exist=True)

    def _cellvit_py(self) -> Path:
        return _resolve("--cellvit-python", "CELLVIT_PYTHON", self.cellvit_python,
                        must_exist=True)

    def hovernet_weights_path(self) -> Path:
        """The pretrained PanNuke checkpoint both segmentation paths load."""
        return _resolve("--hovernet-weights", "HOVERNET_WEIGHTS", self.hovernet_weights,
                        must_exist=True)

    def provenance(self) -> dict:
        """What to record in a dataset's metadata so it says which HEDeST built it."""
        return {"hedest_dir": str(self.hedest), "hedest_commit": git_commit(self.hedest)}

    # --- H-Optimus-0 (imported, same process) -----------------------------------------

    def hoptimus_embeddings(self, slide: Path, segmentation: Path, out: Path,
                            mpp: float | None = None, batch_size: int = 32,
                            num_workers: int = 6) -> dict:
        """One tile-context embedding per segmented cell, saved to ``out``.

        Tiles sit on a half-tile-stride grid at H-Optimus-0's native 0.5 um/px
        (224 px = 112 um); every cell is pooled from the patch tokens its nucleus
        covers in the tile whose centre is nearest its centroid. See
        ``hedest/features/hoptimus.py`` for the full description -- this is a thin
        call, deliberately, so there is one implementation of it in the project.
        """
        extract = self._load_hoptimus()
        out.parent.mkdir(parents=True, exist_ok=True)
        return extract(slide_path=str(slide), json_path=str(segmentation),
                       out_path=str(out), mpp=mpp, batch_size=batch_size,
                       num_workers=num_workers)

    def _load_hoptimus(self):
        """Imports HEDeST's extractor once, caching it on the instance.

        Not ``functools.lru_cache``: this is a mutable dataclass, so Python gives it an
        ``__eq__`` and sets ``__hash__ = None``, and ``lru_cache`` hashes ``self`` on the
        very first call -- ``TypeError: unhashable type: 'Bridge'``. Nothing caught it
        until a GPU stage actually ran, because this is the only method that used it.
        """
        if self._hoptimus is not None:
            return self._hoptimus
        hedest = self.hedest
        if str(hedest) not in sys.path:
            sys.path.insert(0, str(hedest))
        try:
            from hedest.features.hoptimus import extract_hoptimus_embeddings
        except ImportError as exc:  # pragma: no cover - depends on the env
            raise BridgeError(
                f"cannot import hedest.features.hoptimus from {hedest}: {exc}. "
                "The core environment needs torch, timm, openslide-python, loguru and "
                "typer (that module doubles as a CLI, so it imports typer at module "
                "level), and the H-optimus-0 weights must be in the Hugging Face cache."
            ) from exc
        self._hoptimus = extract_hoptimus_embeddings
        return self._hoptimus

    # --- HoVerNet (subprocess, its own env) -------------------------------------------

    def hovernet_mask(self, slide: Path, mask_png: Path) -> None:
        """Otsu tissue mask at :data:`MASK_LEVEL`, written as a PNG."""
        mask_png.parent.mkdir(parents=True, exist_ok=True)
        self._run([str(self._hovernet_py()), "-u", "external/hovernet/run_mask.py",
                   str(slide), str(mask_png), "--mask-level", str(MASK_LEVEL)],
                  lib=self._hovernet_py().parent.parent / "lib")

    def hovernet_infer(self, wsi_dir: Path, mask_dir: Path, out_dir: Path, mpp: float,
                       cache_dir: Path, image_dict: Path | None = None,
                       gpu: str = "0", batch_size: int = 16,
                       inference_workers: int = 4, post_proc_workers: int = 6,
                       save_geojson: bool = True) -> None:
        """WSI segmentation of the single slide in ``wsi_dir``.

        ``run_infer.py`` names its outputs after the input file and requires the input
        directory to hold exactly one, which is why callers stage a single-file folder.
        ``mpp`` is the dataset's own resolution, so the 20 um cell windows are physically
        correct rather than assumed.

        Three arguments exist for reasons that are not obvious and were learned the hard
        way upstream, so they are not optional in practice:

        * ``cache_dir`` **must be per-dataset**. HoVerNet writes fixed-name scratch files
          (``cache_chunk.npy``, ``pred_map.npy``, ``src_wsi.npy``) into it and
          ``rm_n_mkdir``s the whole directory, so two concurrent runs sharing a cache
          corrupt each other.
        * the worker counts are deliberately below HoVerNet's defaults of 8 and 16, which
          exhaust ``/dev/shm`` and the semaphore limit and crash workers with
          ``SemLock FileNotFoundError``, especially when several datasets run at once.

        It also refuses to re-segment when the output directory already holds a ``.json``,
        silently reusing it. Callers clear stale outputs first --
        :func:`sthelar.hovernet_stage.run_infer` does, because it knows which files are
        stale.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(self._hovernet_py()), "-u", "external/hovernet/run_infer.py",
               f"--gpu={gpu}", "--nr_types=6",
               "--type_info_path=external/hovernet/type_info.json",
               f"--nr_inference_workers={inference_workers}",
               f"--nr_post_proc_workers={post_proc_workers}",
               f"--batch_size={batch_size}", "--model_mode=fast",
               f"--model_path={self.hovernet_weights_path()}",
               f"--size_px={CELL_CROP_PX}", f"--size_um={CELL_CROP_UM}",
               f"--mpp={mpp:.6f}",
               "wsi", f"--input_dir={wsi_dir}", f"--output_dir={out_dir}",
               f"--input_mask_dir={mask_dir}", f"--cache_path={cache_dir}"]
        # run_infer.py demands one of the two: without either it segments the whole slide,
        # writes the json and the geojson, and only *then* raises
        # "--image_dict_path is required unless --skip_image_dict is given" -- so the cost
        # of forgetting is a full GPU pass (measured: 32 min for Xenium lung, 54 for
        # breast, 2026-11-08) before the error surfaces. Never leave both off.
        cmd.append(f"--image_dict_path={image_dict}" if image_dict is not None
                   else "--skip_image_dict")
        if save_geojson:
            cmd.append("--save_geojson")
        self._run(cmd, lib=self._hovernet_py().parent.parent / "lib")

    # --- CellViT (subprocess, its own env) --------------------------------------------

    def cellvit_detect(self, slide: Path, out_dir: Path, mpp: float,
                       magnification: int = 40, gpu: int = 0, batch_size: int = 8,
                       image_dict: Path | None = None) -> None:
        """CellViT-SAM detection, exported in the HoVer-Net schema.

        CellViT-SAM predicts the full 1024 px patch (patches overlap by 64 px, merged by
        ``OverlapCellCleaner``), so unlike HoVerNet there is no valid-region shrink and
        nothing to reflect-pad.

        **No ``--adata_path``, deliberately.** That flag is how CellViT applies the same
        300 um spot-proximity filter the HoVerNet route applies, and HEDeST-bench's README
        says the stage uses it -- but its code does not, and the code is right: the flag
        makes ``detect_cells.py`` import ``anndata``, which is not in ``cellvit-env``, so
        the run crashes *after* all the GPU work is done. The filtering happens afterwards,
        on the CPU, in :func:`sthelar.cellvit_stage.finalize`.

        ``mpp`` is passed explicitly rather than read from the file because it drives both
        that filter and the 20 um crop windows, and must not be guessed.
        """
        detect = self.hedest / "external" / "CellViT" / "cellvit" / "detect_cells.py"
        if not detect.exists():
            raise BridgeError(f"CellViT entry point not found: {detect}")
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [str(self._cellvit_py()), "-u", str(detect),
               "--model", "SAM", "--nuclei_taxonomy", "pannuke",
               "--gpu", str(gpu), "--batch_size", str(batch_size),
               "--outdir", str(out_dir),
               "--export_cells",
               "--size_px", str(CELL_CROP_PX), "--size_um", str(float(CELL_CROP_UM))]
        if image_dict is not None:
            cmd += ["--image_dict_path", str(image_dict)]
        cmd += ["process_wsi", "--wsi_path", str(slide),
                "--wsi_mpp", f"{mpp:.6f}", "--wsi_magnification", str(magnification)]
        self._run(cmd, lib=self._cellvit_py().parent.parent / "lib")

    # --- shared ------------------------------------------------------------------------

    def run_in_hovernet_env(self, args: list[str], cwd: Path | None = None) -> None:
        """Runs a command inside the hovernet environment, through ``conda activate``.

        Used by the Visium HD segmentation path only; see :attr:`conda_sh`. The HEDeST root
        goes on ``PYTHONPATH`` because ``run_infer.py`` imports both ``external.hovernet``
        and ``hedest``.
        """
        env_root, hedest = self.hovernet_env, self.hedest
        shell = (f"source {self._conda_sh()} && conda activate {env_root} && "
                 f"export LD_LIBRARY_PATH={env_root}/lib:${{LD_LIBRARY_PATH:-}} && "
                 f"export PYTHONPATH={hedest}:${{PYTHONPATH:-}} && "
                 + " ".join(shlex.quote(str(a)) for a in args))
        subprocess.run(["bash", "-lc", shell], cwd=str(cwd or self.hovernet_repo),
                       check=True)

    def _run(self, cmd: list[str], lib: Path | None = None) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.hedest}:" + env.get("PYTHONPATH", "")
        if lib is not None and lib.is_dir():
            # these envs ship their own CUDA / OpenSlide shared objects
            env["LD_LIBRARY_PATH"] = f"{lib}:" + env.get("LD_LIBRARY_PATH", "")
        subprocess.run(cmd, cwd=self.hedest, check=True, env=env)


def git_commit(repo: Path) -> str | None:
    """Short commit of ``repo``, or None if it is not a clean git checkout."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
