# HEDeST-simdata

Regenerates every simulated dataset [HEDeST](https://github.com/) is validated on, from the
raw public downloads, with one command.

HEDeST annotates single cells on an H&E slide from the image plus spot-level
(deconvolution-style) proportions. To test that, you need slides where the per-cell answer
is already known — so three kinds of single-cell spatial data are turned into **pseudo
Visium v2** datasets: spots with known cell-type proportions over a real H&E. This
repository builds all three. It is a companion to HEDeST and imports from it; HEDeST never
imports from here.

```bash
python make_simdata.py /path/to/data
```

That one folder is the whole interface. The raw downloads go into it, the simulated data
comes out of it, and everything transient lives in `<DATA_ROOT>/_work` and is deleted when
a family finishes.

---

## 1. What you get

| family | source | output | size |
|---|---|---|--:|
| **STHELAR** | 7 Xenium 5K slides | `STHELAR/bench_data/<name>/` — 8 benchmark datasets, 4–5 annotation levels each | 64.5 GiB |
| **Visium HD** | 3 human-colon-cancer samples | `Visium_HD_Human_Colon_Cancer_<P>/pseudovisium/` + `pyr_tif/` | 10.3 GiB |
| **Xenium** | Lung and Breast slides | `Xenium_<slide>/sim[/LuCA]/` — the files HEDeST's gridsearch reads | 1.5 GiB |

**357 files, ~76 GiB.** The exact list is `src/hedest_simdata/manifest.py`, and
`tests/manifests/*.json` records what the reference folders held when this was written, so
"it matches" is a check rather than a claim:

```bash
python tests/inventory.py record --reference /path/to/reference   # what is there
```

The Xenium output paths are deliberately the historical ones (breast in `sim/`, lung in
`sim/LuCA/`): `HEDeST/gridsearch/semi_sim.yaml` hard-codes them, so a regenerated folder
works with that config unedited.

## 2. What you have to download

```bash
python make_simdata.py DATA --stages download        # checks, fetches what it can
```

**STHELAR** is on [EBI BioStudies S-BIAD2146](https://www.ebi.ac.uk/biostudies/studies/S-BIAD2146),
an open FTP, so this just works: 7 `.zarr.zip` archives (7.4–38.8 GB) and 7 cell-metadata
parquets, **157 GB**. Sizes are checked after the download, because resume turns a
truncated source into a silently short file.

**The two 10x datasets** are fetched from 10x's CDN, **105 GB** in total (79 GB Visium HD,
19 GB Xenium), with `binned_outputs.tar.gz`, `spatial.tar.gz` and the Xenium `outs.zip`
unpacked in place. The three bundles do not agree on internal layout — two wrap their
contents in a directory and one does not — so each declares its own in
`download/urls.json` rather than having it sniffed.

Every one of these URLs was **verified, not inferred from a pattern**: a HEAD request has
to answer 200, and the `Content-Length` is recorded and re-checked after the transfer.
Where a local copy existed the two byte counts agreed exactly, which is what makes these
the published files rather than URLs that merely resolve.

- Visium HD: <https://www.10xgenomics.com/platforms/visium/product-family/dataset-human-crc> — per patient, the H&E `.btf` (11–14 GB) and the binned outputs (**79 GB** for the three)
- Xenium Lung: <https://www.10xgenomics.com/datasets/preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard>
- Xenium Breast: <https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast>

**Nothing has to be fetched by hand.** Xenium Breast's
`Cell_Barcode_Type_Matrices.xlsx` carries that slide's published cell types and is served
from 10x's media CDN rather than the sample's directory, so it has its own absolute URL in
`download/urls.json`. If a URL ever does go missing, the stage stops with the dataset page
and the exact path the file goes to, rather than guessing: a guessed URL does not fail
cleanly — it saves an HTML error page under the data file's name, and the first thing to
notice is a confusing crash several stages later.

**Measured throughput**, from one cluster:

| source | rate | what that means here |
|---|--:|---|
| EBI BioStudies FTP | **14.4 MiB/s aggregate** over three concurrent slides (sustained, 52.5 GB) | STHELAR's 157 GB takes **~2.9 h** |
| `cf.10xgenomics.com` | **55 MB/s** including unpacking (sustained, 80 GB) | the 105 GB of 10x data takes **~32 min** |

Both rows are whole-stage measurements from the 2026-11-08 run, not benchmarks: the clock is
the Slurm task's, and the 10x figure *includes* extracting the archives.

EBI parallelises, but **not cleanly** — and the earlier version of this table said it did,
on the strength of a 4.6 GiB burst. Per stream, sustained: 4.2 MiB/s for the 7.9 GB slide
that ran against two others throughout, 6.1 for the 20.2 GB one, 6.7 for the 24.4 GB one
that had the link increasingly to itself. So the per-stream rate falls as streams are added
and the aggregate lands around 14.4 MiB/s, roughly a quarter below what the burst
extrapolated to. The `download` stage still gets what parallelism there is for free, being
one array task per slide, with the files inside a task sequential.

Measure your own link rather than trusting either figure, and if you plan from these, plan
from the sustained numbers above: every short measurement taken here was optimistic.

Single-cell **references** are not needed and are not downloaded. Where a dataset's cell
types came from one, the result is committed instead (§4).

## 3. Install

Three environments, because the dependency sets genuinely conflict. Only the first is
needed for a download-only or CPU-only run.

```bash
CONDA_SOLVER=libmamba conda env create -f environment.yml      # hedest-simdata
CONDA_SOLVER=libmamba conda env create -f envs/bin2cell.yml    # Visium HD bin2cell only
```

`CONDA_SOLVER=libmamba` matters below conda 23.10, where the classic solver does not
converge on this spec in any reasonable time. Check the plugin is actually there —
`conda list -n base conda-libmamba-solver` — because an unrecognised `CONDA_SOLVER` is not
an error, it falls back to classic silently.

**The pip section of `environment.yml` is a lock, and it matters.** The Xenium reader chain
is pinned to the 18 exact versions that built the delivered datasets (§7), because the
`xenium()` reader's output shape moves with them. Two things forced it into pip rather than
the conda list:

- asking conda for the old `spatialdata` alongside these pins makes libsolv 0.7.24 — what
  conda 23.7 ships — run for **69 minutes and then abort** with an internal assertion,
  `solver_addrule: Assertion '!p2 && d > 0' failed`. The chain reaches `xarray-datatree`,
  since folded into xarray, so the pin drags the solver through a large space of retired
  combinations. pip resolves the same packages in seconds.
- conda-forge's `spatialdata-io` starts at 0.2.0, and the version used was 0.1.5.

Resolving that chain *loosely* does not work, and the file says which four versions are
load-bearing and what each one breaks: `xarray` (2025.x kills the import inside
`xarray-datatree`), `xarray-dataclasses` (3.0.0 requires numpy≥2 and silently overrides the
`numpy=1.26` conda pin), the `dask[dataframe]` extra, and `setuptools<81` for
`pkg_resources`. All of it is pure Python, so nothing compiled is shadowed — verified by
importing all 23 packages the stages use.

One consequence to know when editing a stage: **import `spatialdata` before anything that
pulls in `dask.dataframe`.** spatialdata disables dask's query planning itself, but only if
`dask.dataframe` has not already been initialised; otherwise it raises at import. The stages
import it first inside the stage body, which is why this works.

**Then install a CUDA build of pytorch**, or every embedding stage runs on the CPU and takes
hours instead of minutes:

```bash
conda activate hedest-simdata
pip install 'torch==2.4.0+cu118' 'torchvision==0.19.0+cu118' \
    --index-url https://download.pytorch.org/whl/cu118
```

Three details in that one line, each of which cost a failure to find:

- **`torchvision` has to go with it.** Replacing torch alone leaves the conda-built
  torchvision linked against the old binary, and the first thing to notice is
  `RuntimeError: operator torchvision::nms does not exist` — raised by `timm`, on the way to
  loading H-Optimus-0. 0.19.0 is the torchvision that pairs with torch 2.4.0.
- **`+cu118`, so no `--force-reinstall` is needed.** The local version differs from the
  conda-installed `2.4.1`, so pip replaces torch and leaves already-satisfied dependencies
  alone. `--force-reinstall` instead reinstalls every dependency at the newest version on
  that index, which pulls `fsspec 2026.x` into an environment where `spatialdata 0.2.3`
  requires `fsspec<=2023.6`. If you hit that, `pip install 'fsspec==2023.6.0'` repairs it.
- **2.4.0, not newer**: it is the version the reference embeddings were produced with.

`environment.yml` deliberately does not pin this: a CUDA build needs conda's `__cuda`
virtual package, which is absent on a login node without a GPU — exactly where environments
get created — so pinning it would make the file fail to solve in the normal case. Verify on
a GPU node, not where you installed it, since `torch.cuda.is_available()` is `False` on a
login node no matter how torch was built:

```bash
srun --partition=<gpu> --gres=gpu:1 python -c "import torch; print(torch.cuda.is_available())"
```

HoVerNet and CellViT run in **HEDeST's own** environments — this repo does not duplicate
them. They are *spawned*, so the stage around them still runs in `hedest-simdata`:
`visiumhd:bin2cell` is the only stage whose own code runs in a foreign environment, because
it imports StarDist in-process. Then tell the driver where everything is, by flag or by
environment variable:

| variable | flag | what |
|---|---|---|
| `HEDEST_DIR` | `--hedest-dir` | the HEDeST checkout |
| `HOVERNET_PYTHON` | `--hovernet-python` | interpreter of HEDeST's hovernet env |
| `HOVERNET_WEIGHTS` | `--hovernet-weights` | `hovernet_fast_pannuke_type_tf2pytorch.tar` |
| `CELLVIT_PYTHON` | `--cellvit-python` | interpreter of HEDeST's cellvit env |
| `BIN2CELL_PYTHON` | `--bin2cell-python` | interpreter of `envs/bin2cell.yml` |
| `SIMDATA_CONDA_SH` | `--conda-sh` | `.../etc/profile.d/conda.sh`, for the one stage that activates a shell |

Each is resolved **only when a stage that needs it runs**, so none of it has to be
configured to download or to build the CPU stages.

The H-Optimus-0 weights must be in the local Hugging Face cache; the embedding stages run
offline (`HF_HUB_OFFLINE=1`).

## 4. The things no code can reproduce

Everything else is derived by this repository. These are committed, because reproducing
them would mean shipping a reference atlas and making every user re-run a label transfer
for an input that is the same every time:

| file | what | size |
|---|---|--:|
| `annotations/visiumhd/*_annotation.csv.gz` | the manual cell-type calls for P1/P2/P5 — 232,934 / 283,047 / 178,601 cells | 16.4 MiB |
| `annotations/xenium/Xenium_V1_humanLung_*.csv.gz` | the Lung label transfer, 162,254 cells, re-keyed to the published ids with centroids | 3.1 MiB |

Xenium Breast's cell types need nothing committed: they are the spreadsheet on the dataset
page. `annotations/extract_*.py` are the one-off scripts that produced the four committed
tables, shipped so the provenance is auditable rather than asserted — and both of them had
to be rewritten once the ids they keyed on turned out not to survive a rebuild.

**There used to be a fourth row here: the Breast H&E affine.** It is gone, and how it
left is worth the paragraph. The dataset page's `*_he_imagealignment.csv` disagrees with
the copy the delivered data was built with — same scale to five digits, translation off by
(3570, 4551) — so this README said the published one was a *different alignment* that
would put every nucleus in unrelated tissue, and the `download` stage installed the local
copy instead. That was the wrong conclusion from a real observation. Pushing the
translation difference through the linear part turns it into a **crop offset of
(2070.3, 2667.7) px**, and the delivered H&E pyramid is 20511 × 27587 where the one this
repo builds from the byte-identical OME-TIFF is 24241 × 30786. The committed affine was
re-fitted to a *cropped* image that no code here produces.

Measured on the same segmentation, by how many of Breast's 167,780 Xenium nuclei find a
HoVerNet nucleus within the 40 px matching window:

| affine | matched | median distance |
|---|--:|--:|
| committed (fitted to the crop) | 65,634 — 39.1% | 62.38 px |
| published | 124,724 — **74.3%** | **6.85 px** |

The delivered build reached 75.8%, so the published affine is simply right for the image
this repo builds, and a committed input disappeared along with the belief. The crop itself
is *not* recoverable — no integer offset reproduces the delivered pixels (best mean
|difference| 9.2 of 255 at (2664, 2066), correlation 0.985), so that pyramid was
re-encoded as well as cropped — which is why this is fixed by dropping the crop rather
than by committing a crop box that would only look exact. Lung was never affected: its
local alignment is byte-identical to the published one.

The cost is that Breast no longer rebuilds to the byte: segmenting the uncropped pyramid
finds 170,888 nuclei against the delivered 175,204, so the ids do not correspond. Section
7 records what that leaves.

**The Visium HD rows carry more than labels, and that is what saved them.** They used to
be joined to freshly segmented nuclei by bin2cell `object_id`, on the grounds that a re-run
of StarDist/HoVerNet/bin2cell is deterministic and would number the cells the same way. It
does not. The first clean-tree build joined 281,034 of P2's 283,047 annotated cells by id
and **0.09%** of them sat where the annotation said, the largest gap 26,942 px; removing the
median translation left a median of 5,302 px, so the frame had not moved, the numbering had.
The centroid each row carries is what caught it, and is now also what fixes it: the rows are
re-keyed onto this build's cells by mutual nearest centroid.

| sample | re-keyed within 0.01 px | median distance | 5 px-shifted control |
|---|--:|--:|--:|
| P2 | 281,185 of 283,047 — 99.3% | 0.0004 px | **0** |
| P5 | 178,601 of 178,601 — 100% | 0.0004 px | **0** |

0.0004 px is the CSV's own 3-decimal rounding and nothing more, so these are the same cells
down to the stored precision: the segmentation reproduces, only its labels are renumbered.
P2's 1,862 unmatched cells are HoVerNet's own run-to-run variation, not a join failure. The
**control is what makes this a proof** rather than a hopeful nearest-neighbour: the whole
match is repeated with every annotated centroid displaced 5 px and must recover nothing,
and it recovers exactly zero on both samples.

**Xenium Lung needed the same guard, and that took a failed build to establish.** This
README used to say it did not, on the grounds that its key is the vendor's own barcode. The
barcode is stable across re-*runs* of the vendor pipeline but not across **re-releases of
the dataset**: 10x has re-published this slide with a different id space, so the annotation
as first committed joined **8 of 162,254** cells against a fresh download — and nothing
noticed for as long as the converted archive was carried over by hand instead of rebuilt.

The cells themselves never changed: all 162,254 reported centroids are bit-identical
between the two releases. The delivered archive happens to hold *both* id spaces
(`cell_boundaries` by the old ids, `cell_circles` by the published ones), so
`annotations/extract_xenium.py` recovers the map by matching geometry — mutual nearest
neighbour settles 162,248 pairs, with a boundary's polygon centroid sitting a median
**0.13 µm** from its own cell's reported centroid and **52× further** from any other's, and
the 6 left over contested only among themselves are settled by minimum total distance. The
committed CSV now carries each cell's centroid, and the pipeline checks the join against it,
so an id that joins to the *wrong* cell is an error rather than a silent success. A
5 µm-shifted control is rejected for all 162,254 cells.

## 5. Running it

```bash
python make_simdata.py DATA --dry-run                   # the plan, with resources
python make_simdata.py DATA --families visiumhd         # one family
python make_simdata.py DATA --datasets skin_s4 lung_s3  # some datasets
python make_simdata.py DATA --stages download slides    # some stages (+ their deps)
python make_simdata.py DATA --slurm                     # submit instead of running
```

**Re-running resumes.** Every stage knows the file it writes *last*, so finished work is
skipped and a stage that died partway is never mistaken for complete. A prerequisite is
also skipped when everything that needs it is already done, so `--stages verify` on a
finished build does nothing rather than starting from the download. `--force` overrides.

`--stages` always pulls in what a stage depends on: asking for `plots` without `spots`
would draw figures of stale data and report success.

### On Slurm

`--slurm` submits one array per stage, one task per dataset, chained with `aftercorr` so
each dataset flows through independently instead of waiting for the slowest one at every
step — which matters when `lymph_node_s0` is four times the size of `skin_s4`. Nothing about
any particular cluster is hard-coded; set:

| variable | what |
|---|---|
| `SIMDATA_CPU_PARTITION` | partition for the CPU stages (required) |
| `SIMDATA_GPU_PARTITION` | partition for the GPU stages (defaults to the CPU one) |
| `SIMDATA_SBATCH_EXTRA` | anything else, verbatim, e.g. `--exclude=node005,node009` |

The generated scripts are written to `_work/sbatch/` and `--slurm --dry-run` prints them
without submitting.

## 6. What it costs

Measured, not estimated. Every range below is `sacct` elapsed time from the clean-tree run
of 2026-11-08 — `cbio-cpu` with 8 cores, GPU stages on a P100 — over `lung_s3` / `ovary_s1` /
`skin_s4`, Visium HD P2 / P5 and both Xenium slides. **GPU stages are marked in bold.**

| stage | per dataset | notes |
|---|--:|---|
| `sthelar:download` | 30 min – 58 min | three concurrent EBI streams |
| `sthelar:slides` | 1 min 40 – 4 min 16 | the pyramidal H&E export dominates |
| `sthelar:crops` | 1 min 56 – 3 min 09 | |
| `sthelar:stage` | 1 min 31 – 1 min 55 | reflect-pad + Otsu mask |
| `sthelar:hovernet` | **55 min – 1 h 05** | the single biggest cost of the whole build |
| `sthelar:post` | 1 min 32 – 4 min 06 | spot filter, reindex, copies, sim/ |
| `sthelar:cellvit` | **18 min 40 – 30 min 43** | |
| `sthelar:hoptimus` | **8 min 39 – 25 min 08** | the datasets differ by ~3x in cell count |
| `sthelar:plots` | 49 s – 1 min 57 | |
| `sthelar:verify` | 47 s – 2 min 38 | |
| `visiumhd:download` | 16 min – 18 min | 10x CDN, ~55 MB/s including unpacking |
| `visiumhd:bin2cell` | **52 min – 1 h 00** | destripe + StarDist + a full HoVerNet WSI pass |
| `visiumhd:seg` | 51 s – 1 min 17 | with every provenance and matching check |
| `visiumhd:spots` | 29 s – 34 s | 6.9–7.4 M bins |
| `visiumhd:embed` | **~16 min** | 178,601 nuclei, 311 tiles, 10.5 tiles/s |
| `visiumhd:plots` | 40 s – 50 s | 20 figures |
| `xenium:download` | 5 min 21 – 6 min 09 | |
| `xenium:pyr_tif` | 38 s – 42 s | |
| `xenium:convert` | 6 min 59 – 7 min 55 | the Xenium bundle into a zarr |
| `xenium:mask` | 3 min 08 – 4 min 25 | |
| `xenium:hovernet` | **32 min – 54 min** | 522 and 746 Mpx |
| `xenium:spots` | 1 min 39 – 1 min 55 | match, lattice, proportions, GT |
| `xenium:hoptimus` | **10 min 38 – 26 min 28** | |
| `xenium:verify` | 22 s – 28 s | |

So the GPU work is the build: roughly 1 h per STHELAR slide for HoVerNet plus 20–30 min each
for CellViT and H-Optimus, 1 h per Visium HD sample, and 45 min – 1 h 20 per Xenium slide.
With one array per stage the whole thing is on the order of a day once the data is down,
against ~7 h+ of downloading. Note that `visiumhd:bin2cell` declares `max_concurrent=1`, so
its samples are serialised on purpose.

**Disk.** The deliverables are ~76 GiB. STHELAR transiently needs **~117 GB more** in
`_work`: the 7 per-slide pseudo-Visium datasets and the 6 crops are inputs to the benchmark,
not outputs — `bench_data/<name>/` takes its own copies of `he.tiff` and `pseudovisium.h5ad`
— so they are deleted on success. `--keep-work` keeps them.

## 7. Reproducibility

**How to read this section, and the comparison it comes from.** `tests/compare.py` sorts
every deliverable into *equal*, a *known residual*, or *differs*. "Known residual" is not a
euphemism: each one is an entry in `tests/expected_residuals.json` naming the file and the
measured cause, and the tool prints a warning for any entry that stopped matching, so an
explanation cannot be inherited after it stops being true. The exit code counts only new
differences, which is what makes the comparison usable as a regression gate on a GPU
pipeline; `--strict` counts the known ones too.

Given the same downloads, the same HEDeST commit and the same HoVerNet weights, you get the
same datasets — with residuals that are measured below rather than claimed away. Everything
here comes from a **clean-tree** build: an empty data folder, 137 GB downloaded, every stage
run. That qualifier is load-bearing. Earlier rounds of this section were measured in a work
directory that still held the delivered segmentation outputs, so three of the stages had
never executed at all, and each of them had a bug waiting (§7 below, and `annotations/`).

Across all 168 deliverables of the seven datasets, **124 are equal, 44 are known residuals
and nothing is unexplained**:

| family | files | equal | known residuals | unexplained |
|---|--:|--:|--:|--:|
| STHELAR — `lung_s3`, `ovary_s1`, `skin_s4` | 96 | 86 | 10 | **0** |
| Visium HD — `P2`, `P5` | 60 | 36 | 24 | **0** |
| Xenium — lung, breast | 12 | 2 | 10 | **0** |

Twenty-eight of the 44 residuals belong to two datasets and one cause each: Xenium breast's
unrecoverable crop (§4) and Visium HD P2's 0.66% cell loss, below. The rest are a handful of
HoVerNet detections per ten thousand.

- **STHELAR** (`lung_s3`, `ovary_s1`, `skin_s4`, every stage including the GPU ones):
  **86 of 96 files equal, 10 known residuals, nothing unexplained.** Equal means every
  `proportions.csv` at all five levels, every `dapi_seg.json` (the same cell set at 0 px,
  the same types, 100% of ids preserved), `pseudovisium.h5ad`, `he.tiff`, all three
  `cellvit.json`, two of the three `hovernet.json`, and all 42 figures. The residuals are
  `image_dict.pt`, `hoptimus_tile_embed_*.pt` and `spot_dict.json` on each slide, plus
  `hovernet.json` on `lung_s3` — and they all differ for one reason, the HoVerNet cell set
  being a few cells per ten thousand apart. On `ovary_s1` and `skin_s4` the rebuild is a
  strict **superset**: 100% of the delivered keys are present, with 7 and 8 extra cells.
  Per slide, `ovary_s1` has 17 of 5,431 spots holding a different number of cells,
  `skin_s4` 9 of 1,646, `lung_s3` 54 of 3,560 plus 4 spots it does not reach at all. See
  the HoVerNet table below.
- **Visium HD P5**: **26 of 30 files equal**, and the four that are not are figures. All
  three `proportions_lv*.csv` are **identical**, as are `pseudovisium.h5ad` (3,500 ×
  18,058), `spot_dict.json`, `sim/levels.json` and all 25 of `metadata.json`'s data
  fields; `hovernet.json` holds the same 178,601 cells with 99.98% at 0 px and
  `hoptimus_tile_embed.pt` pairs **178,601 of 178,601** nuclei by position.
  `cell_types.csv` agrees on **100.000000%** of all three level labels for every one of its
  178,567 cells — once the rows are aligned by centroid, because its index is this build's
  own numbering.
- **Visium HD P2** is the same picture minus 0.66% of its cells, and that is the only thing
  wrong with it. Its segmentation reproduces to 0.004% (290,840 nuclei found against
  290,829), but 1,862 of its 283,047 annotated cells land on a nucleus the fresh pass drew
  differently, so the centroid re-key cannot carry their labels. They are not
  tolerance-boundary cases — their nearest fresh nucleus is a median **1.49 px** away (p90
  5.9, max 26.4), where a cell that merely jittered sits at 0.0004 px — and no tolerance
  recovers them: 2 px would add 1,201 while the 5 px-shifted control starts accepting 37
  matches it exists to reject. Every P2 residual is that 0.66% propagating: two spots of
  3,564 lost, 94–96% of proportion rows identical, 7 of 280,574 cells in a neighbouring
  spot, and the cell-scatter figures differing wherever a missing cell would have been
  drawn. P5 loses none, which is what says this is P2's slide rather than the method.

**One thing is not bit-identical, and it is worth knowing why.** 21,567 of 86,532 nucleus
centroids in the DAPI segmentation JSONs differ by **exactly one ULP** — at most 7.3e-12 px,
2.0e-12 µm — because float64 association order in the micron-to-pixel matrix multiply
depends on how BLAS blocks the work, which depends on the thread count. `bbox`, `contour`,
`type` and `type_prob` were identical for every entry, and nothing downstream moved: the
centroid only feeds spot assignment, and the spot assignment came out the same. That is why
`manifest.py` compares JSON with a tolerance rather than byte-for-byte — comparing bytes
there would report machine noise as a failure and bury a real regression next to it.

**The Xenium datasets took the most measurement, and are where the residuals are.** Their
construction lived in two notebooks that explored several variants side by side, were last
saved in a configuration that did not ship, and whose stored cell outputs come from
inconsistent executions — one cell prints Xenium barcodes as `nucleus_id`, the next prints
HoVerNet integers. So how HoVerNet nuclei were tied to Xenium cells was recovered by
rebuilding each candidate and comparing the resulting nucleus-id *sets* against the
delivered segmentation:

| candidate | covers, of 118,005 reference nuclei |
|---|--:|
| nearest Xenium nucleus, < 40 px (**used**) | 117,998 — **99.99%** |
| nearest Xenium nucleus, < 20 px | 113,224 — 95.95% |
| intersect the Xenium cell, overlap ≥ 0.10 | 102,574 — 86.92% |
| intersect the Xenium cell, overlap ≥ 0.55 | 97,623 — 82.73% |

The last piece was that the **two slides are from different Xenium releases**: breast's
nuclei *are* its cells (167,780 of each, sharing one id), while lung's are separate objects
in their own id space — only 6 of its first 5,000 nucleus ids appear among its cell ids at
all — so lung needs an extra nucleus→cell hop before a matched nucleus means anything.

With that settled, here is what a **clean-tree** rebuild gives. The distinction matters: an
earlier round of this section reported both slides as near-exact, and it had been measured
in a work directory that still held the delivered segmentation — so `xenium:convert`,
`xenium:hovernet` and `visiumhd:bin2cell` had never actually run. These numbers are from a
build that started with an empty folder.

**Lung agrees on everything HEDeST reads.**

| file | result |
|---|---|
| `pannuke_fast_mask_lvl3_annotated.json` | 141,257 cells both ways, 100% coincident within 1 px (99.98% at 0 px), and of the 107,576 cells callable in either build **exactly 1** disagrees about its cell type |
| `hoptimus_tile_embed_HL.pt` | **equal** — 107,286 cells paired by position, all (1536,) |
| `sim_…_gt.csv` | 107,433 rows against 107,429; all 141,257 nuclei pair within 1 px and **9 of 107,286** paired rows differ, each by a single type flip |
| `spot_dict_real.json` | all **1,957 of 1,957** delivered spots present, 1,928 of them holding the same cells, plus 1 extra spot holding 1 nucleus |
| `prop_real.csv` / `pseudo_adata_real.h5ad` | the same 1,957 plus 1; 98.93% of shared proportion rows identical |

Two residuals are left and both are understood. The first is that **HoVerNet is not
bit-stable here**: 31 of the 141,257 nuclei move by up to 0.5704 px, which is what adds the
4 ground-truth rows and lifts one extra spot over the threshold. The second is cosmetic —
2,551 nuclei carry a different *non-callable* code (code 7 "matched a dropped type" vs code
8 "matched nothing"). Neither code is a cell type, neither enters the ground truth, the
proportions or the spots, and HEDeST reads them only to build a fallback palette for an
overlay. The cause is in the dataset rather than the code: the delivered archive's
`cell_boundaries` is keyed by the **retired** id space while `cell_circles` and
`nucleus_boundaries` carry the published one, so the reference's nucleus→cell→type hop ran
in a space a fresh download no longer has.

**Breast is not exact, and the H&E pyramid is why** — see §4. The delivered dataset was
built on a 20511 × 27587 crop that this repo does not reproduce, so HoVerNet sees more
tissue and finds 170,888 nuclei against 175,204. What comes out:

- **2,656 spots against 2,651** (+0.19%), 43,362 nuclei in them.
- 123,358 of 167,780 Xenium nuclei matched, **73.5%** against the delivered 75.8%.
- 116,752 ground-truth rows against 120,336, and per-type nucleus counts 1–4% below the
  delivered ones — a shortfall that tracks the 2.5% fewer nuclei, not a change of rule.

`src/hedest_simdata/xenium/build.py` carries the full table.

**HoVerNet is where the residuals live, and the earlier version of this paragraph was
wrong about it.** It said the detections reproduce exactly and only their *order* varies,
citing `skin_s4`: the identical 98,678 cells, every one at **0 px** in both directions. That
measurement was taken in a work directory that still held the delivered segmentation, so it
compared a file with itself. On a clean tree:

| dataset | rebuilt | delivered | difference |
|---|--:|--:|--:|
| STHELAR `ovary_s1` | 317,771 | 317,764 | **+7** (0.002%) |
| STHELAR `skin_s4` | 98,686 | 98,678 | **+8** (0.008%) |
| STHELAR `lung_s3` | 170,794 | 171,239 | **−445** (0.26%) |
| Visium HD P5 | 178,601 | 178,601 | 0 — but 34 cells each way, ≤0.089 px |
| Xenium Lung | 141,257 | 141,257 | 0 — but 31 cells each way, ≤0.5704 px |

So a few cells per ten thousand, and `lung_s3` an order of magnitude worse. That one was
chased down rather than averaged away: its `he.tiff` is pixel-equal to the delivered one, and
**90%** of the cells it finds that the reference lacks sit within 1% of an image edge, against
**1.1%** for cells in general — an 80× enrichment at the border. The difference is the tissue
mask HoVerNet was given, not the image and not the model. The original mask is an
intermediate that was never delivered, so the two cannot be compared directly.

What *is* exact, measured across three slides and five levels each, is the **DAPI ground
truth** — the same cell set at 0 px, the same types, and 100% of ids preserved — and every
`proportions.csv` and `pseudovisium.h5ad` with it. The data HEDeST trains on reproduces; the
HoVerNet segmentation beside it reproduces to about four decimal places.

**The cell *ids* are a separate problem and a sharper one.** The **order** HoVerNet emits
cells in is not reproducible — the WSI post-processing runs across several workers, so which
tile's cells arrive first depends on scheduling, and the 0..n−1 reindexing that follows
permutes. Only **68.7%** of `skin_s4`'s cells landed at the same id.

Within one build everything stays consistent — `hovernet.json`, `image_dict.pt`,
`spot_dict.json` and `hoptimus_tile_embed_*.pt` share one id space, and the `verify` stage
checks exactly that. Across builds the ids are **not comparable**, with one concrete
consequence worth stating plainly:

> HEDeST's `simulations/semi_simulations/STHELAR/matches/gt_<name>_<level>.json` maps
> HoVerNet ids to ground-truth cell types, and `gridsearch/semi_sim.yaml` reads it. Those
> files were built against the ids of the *original* `bench_data`. **If you regenerate
> STHELAR, they no longer line up and must be rebuilt** (`match_cells.py` in that folder).

**This section used to end by saying Visium HD and Xenium were unaffected** — Visium HD
because bin2cell's `object_id` is stable, Xenium because its ids come from the vendor. Both
halves were wrong, and finding out is most of what the first clean-tree run was worth:

- **bin2cell's `object_id` is not stable.** It is derived from the nucleus label image, and
  a re-run paints the same cells with a different numbering: **1.0%** of P5's ids and
  **0.09%** of P2's came back the same. The committed cell types joined by that id and
  landed on other cells, with every output file still well-formed. They are re-keyed by
  centroid now, and the centroids agree to the CSV's own rounding (§4). The sparseness of
  that id space also makes a key-set comparison actively misleading: P5's embeddings cover
  exactly the nuclei its `hovernet.json` holds, and the two builds' nuclei coincide 100%
  within 1 px, yet only **95.4%** of the delivered keys appear in the rebuild — 178,601
  ids drawn independently from 1..187,378 overlap by that much by chance. The embeddings
  are compared through the two segmentations' centroids for that reason, not by key.
- **A vendor id is stable across re-runs, not across re-releases.** 10x re-published Xenium
  Lung with a different `cell_id` space, so the committed label transfer joined **8 of
  162,254** cells against a fresh download (§4).

What *is* stable, in all three families, is where a cell is. Every one of these joins is
geometric now, and every one is proved against a shifted control rather than assumed — which
is also why the comparison in `tests/compare.py` aligns files by position before comparing
them, and says so in each line it prints.

**A third kind of id mistake is worth separating from those two,** because it is not about
stability at all: two tools using different key *spaces* for the same cells. HEDeST's
embedding extractor names its output by **position** in the segmentation file it is handed
(`cell_ids = [str(i) for i in range(len(nuclei))]`). For STHELAR and Xenium that is the
right key, because their segmentations are renumbered 0..n−1 before anything is keyed to
them. Visium HD's `hovernet.json` keeps bin2cell's own sparse cell ids — 1..187,378 for
178,601 cells — and `spot_dict.json`, `cell_types.csv` and the delivered embedding file are
all keyed by those. Taking the extractor's output as-is therefore produced a file keyed
0..178,600, in which **not one** of the spot dictionary's cells can be looked up: the right
size, the right shapes, well-formed, and useless. `visiumhd/embed.py` now maps position back
to cell id and checks the spot dictionary against the result; the pipeline's own `verify`
(check V2) would also have caught it, and did not only because the stage crashed twice
before reaching it.

**CellViT was expected to be the inexact one, and it turned out to be the better-behaved
of the two.** `cellvit.json` has a reason to drift: the published checkpoint is an AMP one
(`CellViT-SAM-H-x40-AMP.pth`), so in fp16 a borderline cell falls on either side of
threshold, and the `OverlapCellCleaner` that merges cells across overlapping 1024 px patches
resolves duplicates in whatever order the ray workers finish. Measured on the clean run it
nonetheless lands inside its own `segment_amp` tolerance on all three slides:

| dataset | rebuilt | delivered | within 1 px, each way | at 0 px | types agree |
|---|--:|--:|--:|--:|--:|
| `lung_s3` | 156,239 | 156,246 | 99.97% / 99.97% | 99.70% | 99.98% |
| `ovary_s1` | 300,318 | 300,319 | 99.97% / 99.97% | 99.68% | 99.99% |
| `skin_s4` | 86,125 | 86,119 | 99.96% / 99.96% | 99.65% | 99.97% |

So CellViT's drift is ~0.3% of cells moving sub-pixel and ~0.005% of cells appearing or
disappearing — tighter, and far more *consistent* between slides, than HoVerNet's. The
`segment_amp` thresholds were set from the `skin_s4` row and the other two then fell inside
them, which is the only kind of confirmation a tolerance can get.

**The figures differ too, by a sliver, and that was worth checking rather than assuming.**
All 14 of `skin_s4`'s differ — in 0.27% to 0.96% of their pixels, mean absolute channel
difference under 1.4 of 255 — and in every one the differing pixels start at the legend
column, with the data panel to its left pixel-identical. Shifted text extents also move a
`bbox_inches="tight"` figure's width, which is the 1–2 px seen on 8 of the 14. It is the
freetype/font version, not the content. The first version of the `figure` policy compared
only the pixel dimensions, and so reported 5 of these files as equal while they differed in
half a percent of their pixels; it now measures the difference and states it.

**Two bugs that only a from-scratch run could find, and what now catches them.** Until
2026-11-08 every validation reused work directories that already held the segmentation
outputs, so two stages had never actually built their own inputs. A clean-tree run found:

- `hovernet_infer` passed neither `--image_dict_path` nor `--skip_image_dict` when no cell
  crops were wanted. `run_infer.py` accepts that, segments the whole slide, writes the json
  *and* the geojson, and only then raises — so the cost of the omission was a full GPU pass
  (32 min for Xenium lung, 54 for breast) before the error appeared. Only the Xenium stages
  ask for no crops; the Visium HD path has its own copy of that command line and did pass
  the flag, so there was nothing to compare it against.
- the Visium HD HoVerNet cache defaulted to one hardcoded absolute path shared by every
  sample. HoVerNet writes fixed-name scratch there and clears the directory first, so P2 and
  P5 on one node mapped the same `pred_map.npy` and both died with `SIGBUS`. Upstream's
  wrapper script had always supplied a per-sample path through an environment variable; the
  port kept the fallback and left the wrapper behind, which made a branch that had never run
  the operative one.

- `xenium/paths.py` looked for the segmentation at `seg_json/json/<name>.json`. HoVerNet
  writes that subdirectory **only** when `--save_mask` or `--save_thumb` is given and the
  flat `seg_json/<name>.json` otherwise, and the bridge gives neither — so the file was
  never found, the stage could never be marked done, and `xenium:spots` would have failed on
  a missing file after the hour of GPU work that produced it. It looked right because
  `process_wsi_list` creates an empty `json/` directory regardless. The STHELAR stage, which
  clears the flat path before re-segmenting, had it correct all along — a third case of one
  copy of this knowledge being right and another wrong.
- `environment.yml` left `spatialdata` and `spatialdata-io` unpinned, and the solve takes
  0.4+. The delivered Xenium datasets were built with **spatialdata 0.2.3 and
  spatialdata-io 0.1.5**, whose `xenium()` reader shapes the SpatialData archive
  differently — and `xenium:spots` reads that archive, so every Xenium deliverable would
  have moved. Both are now pinned to the versions that made the data. 0.1.5 has to stay in
  the pip list: conda-forge's `spatialdata-io` starts at 0.2.0.

Both command-line bugs were visible in the argv without running anything, so
`tests/test_bridge_cmd.py` now
builds the commands and asserts the invariants: every builder of a `run_infer.py` command
line passes exactly one of the two flags, and no string literal in code names a path under
`/cluster` or a home directory (docstrings may, since they record measurements). The second
check is why this repo is safe to publish: it found that shared cache path was the only such
literal in 62 files. `tests/test_env_spec.py` then compares `environment.yml` against every
import the stages make, which is how `h5py` was found — imported directly by four stages,
declared nowhere, working only because `anndata` happens to depend on it.

The root cause of three of these was the same: **a stage that had never built its own
inputs.** `make_simdata.py` now also refuses to submit when the current interpreter cannot
import what the planned stages will reach for (`--no-preflight` overrides). It is a static
check over the stage functions' imports, so it costs milliseconds and would have turned that
night's five failures into one message before the first download started.

**Not reproducible, and excluded:** the MoCo-v3 embeddings the STHELAR benchmark used to
ship. They were trained without a fixed seed, so re-running gives statistically equivalent
but numerically different vectors, and they are superseded by H-Optimus-0. `image_dict.pt`
is unaffected — it is written by the `post` stage, not by the MoCo stage.

**Checking a rebuild.** `tests/compare.py` does it, honouring each file's policy:

```bash
python tests/compare.py --built /path/to/new --reference /path/to/old
```

## 8. Layout

```
make_simdata.py              THE entry point
environment.yml              the core environment
envs/bin2cell.yml            Visium HD bin2cell only
download/urls.json           where the raw data comes from
annotations/                 the committed cell types, and the scripts that made them
src/hedest_simdata/
├── config.py                the registries: samples, crops, spot geometry, per-sample quirks
├── manifest.py              what each family must produce, and how to compare it
├── pipeline.py              the stage graph
├── hedest_bridge.py         the only module that reaches into HEDeST
├── slurm.py                 one array per stage
├── common/                  lattice · download · sources · tiff
├── sthelar/                 7 slides -> 8 benchmark datasets
├── visiumhd/                3 samples -> pseudo-Visium
└── xenium/                  2 slides -> pseudo-Visium
tests/
├── inventory.py             record a folder's contents against the manifest
├── compare.py               compare a rebuild to a reference, policy by policy
├── test_lattice.py          the three spot lattices, against verbatim copies of the originals
├── test_wiring.py           the stage graph, the manifests and the registries, with no data
├── test_download.py         the three archive layouts, unpacked from synthetic tars and zips
├── test_bridge_cmd.py       the foreign command lines, and the absence of site paths in code
├── test_env_spec.py         environment.yml against the stages' imports; and that every call
│                            on one of this package's own modules names something real
├── test_cell_labels_rekey.py  the centroid re-key, on synthetic cells where the answer is
│                            known: renumbered must work, translated and jittered must not
├── expected_residuals.json  the differences from the delivered data that were measured and
│                            explained, so compare.py can report only the new ones
└── manifests/               what the reference folders held
```

### Where this came from

Three codebases, merged and pruned:

| was | is |
|---|---|
| `HEDeST-bench` (`src/hedest_bench`, 6,939 lines) | `sthelar/`, 3,072 lines — `hier/` dropped, MoCo dropped, slide/crop figures dropped, an `hoptimus` stage **added** |
| `VisiumHD_analysis` (`pseudovisium/`, `analysis/preprocessing/`) | `visiumhd/` — the annotated h5ad replaced by a committed CSV, one H-Optimus implementation |
| `HEDeST/simulations/semi_simulations/Xenium/dataset_construction` (two notebooks, 157 and 117 cells) | `xenium/` — see `xenium/build.py` for what had to be recovered from the delivered files |

Three things that look shared and are not, each documented where it matters: the **three
spot lattices** (`common/lattice.py` — the two "honeycombs" differ in stagger parity, and
Xenium's is not a honeycomb at all), the **two Xenium matching routes** (`xenium/build.py`),
and the **catch-all label** that every family has under a different name — `Other`,
`unassigned`, `Unlabeled`, `low_counts`.
