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
data/           TCGA tissue-source-site partitioner (reused infrastructure)
results/        aggregate metric JSONs (macro accuracy etc.; no patient data)
PREREGISTRATION.md   frozen analysis plan (timestamp inside; = paper SI S7)
requirements.txt / LICENSE / .gitignore
```

Key modules: `assign.py` (model-independent partition + 4 assignment
schemes), `projector.py`, `proto_anchor.py` (the protocol), `baselines_het.py`
(zero-pad / local-PCA / Procrustes), `fedprotokd.py` (FedProtoKD, faithful
and tuned variants), `diagnostics.py` (silhouette + cross-model prototype
agreement). Runners: `verify_week1.py`, `run_hetfm.py`, `run_grid.py`,
`run_week4/5/6.py`, `run_linear_rebaseline.py`, `run_perclass.py`,
`run_prereg.py` (executes the frozen plan), `make_figures.py`,
`make_supp.py`.

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

`results/*.json` are aggregate metrics only (macro accuracy / F1, per-seed
and summarised; no patient-level or slide data) and reproduce the numbers
in the manuscript and supplementary tables.

## License & citation

MIT (see `LICENSE`; confirm or change to your preferred license before
publishing). If you use this code, please cite the paper (Discover
Computing; full reference to be added on acceptance).
