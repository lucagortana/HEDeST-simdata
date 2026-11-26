# HEDeST-simdata

Builds the simulated datasets used to validate [HEDeST](https://github.com/sysbio-curie/HEDeST)
from public single-cell spatial data. Each dataset is a real H&E slide with Visium-like spots
whose cell-type proportions are known, along with the true cell type of the cells.

## What it builds

| family | source data | datasets | output folder |
|---|---|---|---|
| STHELAR | 7 Xenium 5K slides ([EBI S-BIAD2146](https://www.ebi.ac.uk/biostudies/studies/S-BIAD2146)) | `breast_s6`, `cervix_s0_0`, `cervix_s0_1`, `lung_s3`, `lymph_node_s0`, `ovary_s1`, `prostate_s0`, `skin_s4` | `STHELAR/bench_data/<dataset>/` |
| Visium HD | 3 colorectal cancer samples ([10x](https://www.10xgenomics.com/platforms/visium/product-family/dataset-human-crc)) | `P1`, `P2`, `P5` | `Visium_HD_Human_Colon_Cancer_<P>/pseudovisium/` |
| Xenium | Lung ([10x](https://www.10xgenomics.com/datasets/preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard)) and Breast ([10x](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast)) slides | `lung`, `breast` | `Xenium_V1_humanLung_Cancer_FFPE/sim/LuCA/`, `Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/` |

Every dataset contains:
- the HoVer-Net nuclei;
- the spots (an AnnData, plus the nuclei in each spot) and the cell-type proportions of each spot;
- the true cell types;
- an H-Optimus-0 embedding for each nucleus.

The true cell types are given per nucleus for Visium HD and Xenium. For STHELAR they are
given per Xenium cell (`sim/level*/dapi_seg.json`), and HEDeST's
`simulations/semi_simulations/STHELAR/match_cells.py` matches them to the nuclei.

What differs by family:
- **STHELAR:** proportions and cell types at 4 to 5 annotation levels, plus the H&E, a CellViT segmentation and nucleus crops.
- **Visium HD:** proportions and cell types at 3 levels; the H&E is in the sample's `pyr_tif/` folder.
- **STHELAR and Visium HD:** QC figures in `plots/`.

The full list of files is in `src/hedest_simdata/manifest.py`.

Some cell types cannot be computed from the downloads: the Visium HD annotations and the Xenium Lung label transfer. They ship with the repository in `annotations/`. The STHELAR and Xenium Breast cell types come with their downloads.

## Requirements

- Linux, conda and a CUDA GPU (for segmentation and embeddings). Some stages request up to
  160 GB of RAM.
- About 450 GB of free disk space: 260 GB of downloads (kept), 76 GB of datasets, and up to
  120 GB of temporary files while STHELAR is built.
- A few hours of downloads, then about a day of computing on a Slurm cluster (`--slurm`).
  It takes much longer when everything runs one task at a time on a single machine.

## Tutorial

### 1. Install

```bash
git clone https://github.com/lucagortana/HEDeST-simdata.git
cd HEDeST-simdata
conda env create -f environment.yml        # env "hedest-simdata"; with conda < 23.10, prefix with CONDA_SOLVER=libmamba
conda activate hedest-simdata
pip install 'torch==2.4.0+cu118' 'torchvision==0.19.0+cu118' --index-url https://download.pytorch.org/whl/cu118
conda env create -f envs/bin2cell.yml      # env "hedest-simdata-bin2cell", for Visium HD only
```

The segmentation uses HEDeST's code and environments:

```bash
git clone https://github.com/sysbio-curie/HEDeST.git
cd HEDeST && ./setup_env.sh                # creates hedest-env and hovernet-env
```

STHELAR also needs CellViT, which runs in its own environment. Create it by following
`HEDeST/external/CellViT/README.md`.

### 2. Get the model weights

- **HoVer-Net:** download `hovernet_fast_pannuke_type_tf2pytorch.tar` from the
  [HoVer-Net repository](https://github.com/vqdang/hover_net#model-weights).
- **H-Optimus-0:** this is a gated model, so first request access on
  [its page](https://huggingface.co/bioptimus/H-optimus-0). Then download it once into the
  local Hugging Face cache, which is where the pipeline reads it from:
  ```bash
  huggingface-cli login
  huggingface-cli download bioptimus/H-optimus-0
  ```
- **CellViT:** it downloads its own weights the first time it runs (3 GB, into
  `~/.cache/cellvit`, or into `$CELLVIT_CACHE` if that is set).

### 3. Tell the pipeline where they are

```bash
export HEDEST_DIR=/path/to/HEDeST
export HOVERNET_PYTHON=/path/to/envs/hovernet-env/bin/python
export HOVERNET_WEIGHTS=/path/to/hovernet_fast_pannuke_type_tf2pytorch.tar
export CELLVIT_PYTHON=/path/to/envs/<cellvit env>/bin/python
export BIN2CELL_PYTHON=/path/to/envs/hedest-simdata-bin2cell/bin/python
export SIMDATA_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
```

The first five can also be passed as options: `--hedest-dir`, `--hovernet-python`,
`--hovernet-weights`, `--cellvit-python` and `--bin2cell-python`. Each one is needed only by
the stages that use it; downloading needs none of them.

### 4. Build

Everything goes into one folder: the downloads, the datasets and the temporary files. The
temporary files are in `<data>/_work` and are deleted once a family is done.

```bash
python make_simdata.py /path/to/data --dry-run    # show the plan: stages, tasks, resources
python make_simdata.py /path/to/data              # download and build everything
```

| option | what it does |
|---|---|
| `--families sthelar visiumhd xenium` | build only some families |
| `--datasets skin_s4 P2 lung` | build only some datasets (names as in the table above) |
| `--stages download` | run only some stages, plus the stages they depend on (`--dry-run` lists them) |
| `--slurm` | submit to Slurm instead of running locally (see below) |
| `--force` | rebuild what is already built |
| `--keep-work` | keep the temporary files |
| `--work-dir DIR` | put the temporary files in another folder |

Running the command again resumes the build: finished stages are skipped.

**On a Slurm cluster**, each stage is a job array with one task per dataset. A dataset moves
on to its next stage as soon as it has finished the previous one.

```bash
export SIMDATA_CPU_PARTITION=<cpu partition>
export SIMDATA_GPU_PARTITION=<gpu partition>             # defaults to the CPU one
export SIMDATA_SBATCH_EXTRA="<extra sbatch options>"     # optional
python make_simdata.py /path/to/data --slurm --dry-run   # print the job scripts
python make_simdata.py /path/to/data --slurm
```

### 5. Check

- The last stage of every dataset, `verify`, checks every file the dataset should contain.
- `python tests/test_wiring.py` checks the installation without any data.
- `python tests/compare.py --built /path/to/new --reference /path/to/other` compares two
  builds file by file.

A rebuild matches the published datasets closely, but not exactly:
- HoVer-Net does not segment exactly the same nuclei twice. This changes up to 0.7% of the
  cells (Visium HD `P2`; 0.3% on `lung_s3`, less elsewhere).
- Xenium Breast differs more, with about 2.5% fewer nuclei, because the published version was
  built from a cropped copy of its H&E.

All known differences are listed in `tests/expected_residuals.json`.

Nucleus ids are valid only within one build, so anything keyed by nucleus id must be
recomputed on your own build. For example, run HEDeST's `match_cells.py` again on a rebuilt
STHELAR.

## Repository

```
make_simdata.py        the entry point
environment.yml        main environment (envs/bin2cell.yml: Visium HD)
download/urls.json     URLs and sizes of every download
annotations/           cell types shipped with the repository, and the scripts that made them
src/hedest_simdata/    the pipeline: sthelar/, visiumhd/, xenium/, and common/ for shared code
tests/                 checks with and without data
slurm/                 notes on the Slurm mode
```
