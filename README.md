# Aligning incompatible pathology foundation models for federated learning

Code, the frozen pre-registered analysis plan, and aggregate result tables
for *A Prototype-Anchored Projector Protocol and Heterogeneity-Tolerance
Testbed for Federated Learning across Incompatible Pathology Foundation
Models* (single author: Yen-Jung Chiu, Chang Gung University).

Each site holds a *different, frozen* pathology foundation model with
incompatible embedding dimensionality; weight-averaging federated learning
is undefined. Sites learn per-model projectors into a shared latent space
anchored to server-held class prototypes. Only projectors, a small shared
head, and class prototypes are exchanged; encoders and slides never move.

## Repository layout

```
hetfm/          protocol, baselines, runners, figure/table generators
tools/          table and figure generators from result JSONs
data/           TCGA tissue-source-site partitioner (reused infrastructure)
PREREGISTRATION.md   frozen analysis plan (timestamp inside)
requirements.txt / LICENSE / .gitignore
```

Key modules: `assign.py` (model-independent partition + 4 assignment
schemes), `projector.py`, `proto_anchor.py` (the protocol), `baselines_het.py`
(zero-pad / local-PCA / Procrustes), `fedprotokd.py` (FedProtoKD, faithful
and tuned variants), `diagnostics.py` (silhouette + cross-model prototype
agreement). Runners: `verify_week1.py`, `run_hetfm.py`, `run_grid.py`,
`run_week4/5/6.py`, `run_linear_rebaseline.py`, `run_perclass.py`,
`run_prereg.py` (executes the frozen plan), `run_seedext.py` (every
scheme-level comparison at the thirty headline seeds), `run_mincount.py` /
`run_mincount_clean.py` (minimum-count gate sweep), `make_figures.py`,
`make_supp.py`.

## Extended runners and baselines

- `hetfm/r2_trainer.py`   trainer with head modes (shared / local / none),
  per-class recall, nearest-prototype accuracy, alignment diagnostics, a
  model-identity probe, timing and peak memory; reproduces `proto_anchor.py`
  bit-for-bit on the same backend
- `hetfm/fedgh.py`        FedGH (Yi et al., ACM MM 2023) in faithful and
  tied form, with a cross-site routing probe
- `hetfm/run_r2_batch.py` single-GPU batch runner over the thirty headline
  seeds (matched homogeneous controls, loss-term and head ablations, FedGH,
  450-round checks) and its summariser
- `hetfm/r2_backend.py`   GPU versus CPU (24 / 2 threads) backend study
- `hetfm/r2_cost.py`      timing and memory pass for the cost table
- `tools/mi_fm_y.py`      mutual information between model identity and
  class, and the identity-only classifier, per assignment scheme
- `tools/make_tables_v2.py`, `tools/make_figures_v2.py`,
  `tools/fig1_overview_v2.py`   tables and figures from the batch summary;
  no hand-typed numbers

These files carry `<repo-root>`, `<analysis-dir>` and `<out-dir>`
placeholders where the original environment had absolute paths; set them
before running (`python -m hetfm.run_r2_batch --help`).

## Data

Whole-slide images are from **The Cancer Genome Atlas (TCGA)**, public via
the GDC Data Portal. Slide-level features were pre-extracted with the
frozen **UNI v2 (12288-d)**, **CONCH v1.5 (6144-d)**, and **Virchow2
(20480-d)** models (mean-pooled tile features). **Derived features and raw
slides are NOT redistributed here**; regenerate them from TCGA with the
respective public models, then point the code at them (below).

## Configuration (edit before running)

Paths are hard-coded for the original environment; set them for yours:

- `hetfm/assign.py` — `FEAT_ROOT`: directory of cached per-model `.npy`
  features (subdirs `UNI_v2/`, `Conch_v15/`, `Virchow2/`).
- `data/partition.py` — `WSI_IMAGE_ROOT`: TCGA WSI directory used to build
  the case to project map.
- `hetfm/make_figures.py`, `hetfm/make_supp.py`, `hetfm/diagnostics.py` —
  output paths currently point at the manuscript directory; change to a
  local `figures/` / output directory.

## Environment

Python 3.11.9; see `requirements.txt` (torch 2.10, numpy 1.26.4,
scipy 1.13.1, matplotlib 3.10.8, scikit-learn 1.6.1). Run module-style
from the repository root, e.g.:

```
python -m hetfm.verify_week1          # testbed gates
python -m hetfm.run_prereg            # the pre-registered n=30 confirmation
python -m hetfm.run_week6             # assignment-skew (RQ4)
python -m hetfm.run_perclass          # pre-registered per-cancer analysis
python -m hetfm.diagnostics           # alignment-quality diagnostics
```

## Pre-registration

`PREREGISTRATION.md` is the analysis plan, fixed (timestamp inside) before
the confirmatory runs and not edited afterwards; it is reproduced in the
paper's Supplementary Information (Section S7). It is an author-committed
pre-specified plan, not a third-party registry entry.

## Results

Per-seed aggregate metric files (macro accuracy / F1 and diagnostics; no
patient-level or slide data) are available from the author and will be
added here on publication.

## License & citation

MIT (see `LICENSE`; confirm or change to your preferred license before
publishing). If you use this code, please cite the paper (Discover
Computing; full reference to be added on acceptance).
