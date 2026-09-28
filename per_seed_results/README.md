# Per-seed result files

These files hold every per-seed result behind the tables and figures of the
article *A Heterogeneity-Tolerance Testbed and Controlled Dissection of
Projector-Based Federated Learning Across Incompatible Pathology Foundation
Models*, so that the reported means, confidence intervals, paired
comparisons, Holm adjustments, and backend differences can be checked
independently. The values are copied unchanged from the result files written
by the training runs.

## Contents

| File | What it holds |
|---|---|
| `per_seed_results.csv` | One row per configuration (arm), seed, and compute backend: 2,002 rows, 71 arms. |
| `label_information_per_seed.csv` | Table 3: per seed and assignment scheme, the mutual information between foundation model and class, the class entropy, the identity-only accuracy, and the classes seen per model. |
| `selection_grid_per_seed.csv` | Supplementary Table S1: the five-seed selection grid. |
| `recompute_tables.py` | Recomputes Tables 3 and 5 to 11, the gate sweep, the 450-round checks, and Supplementary Tables S1 and S3 to S6 from the three CSV files, with a paired t p-value next to each Wilcoxon p-value (needs `numpy` and `scipy`). |
| `raw_json/` | The raw result files themselves (one JSON per run), in the folder layout used by the code repository: `r2_batch/` (all 150- and 450-round arms), `r2_batch/backend/` (CPU backend study), `mincount/` (gate sweep, mixed CPU/GPU batch), `mincount_clean/` (gate sweep, single GPU batch), plus `mi_fm_y.json`, `ablation_grid.json`, `r2_cost.json` and `cost_analytic.json`. |

## Columns of `per_seed_results.csv`

| Column | Meaning |
|---|---|
| `arm` | configuration name used in the code and the raw files (e.g. `method_linear_balanced`); the three dimension-matching and classical-alignment baselines are split as `base_<scheme>:zeropad`, `:localpca`, `:procrustes` |
| `configuration` | the configuration as described in the article |
| `scheme` | model-to-site assignment scheme (`stratified`, `balanced`, `cancer_correlated`, `adversarial`), or `homogeneous` for single-model controls (scheme-independent) |
| `seed` | seed; it fixes the case-level split and the model-to-site assignment (62–91; the selection grid's seeds 42–46 are in `selection_grid_per_seed.csv`) |
| `rounds` | communication rounds (150, or 450 for the longer-training checks) |
| `backend` | `gpu` (one NVIDIA GeForce RTX 5070 Ti), `cpu_24_threads`, `cpu_2_threads`, or `cpu_mixed_batch` (seeds 62–73 of the mixed CPU/GPU gate-sweep batch; seeds 74–91 of that batch ran on the GPU) |
| `gate_m` | per-class minimum-count gate on prototype contributions |
| `primary_eval` | `head` (shared-head inference) or `prototype` (nearest global prototype); the primary score is `macro_acc` |
| `macro_acc`, `macro_f1` | nine-class macro accuracy (mean per-class recall) and macro-F1 of the primary classifier |
| `acc_head`, `f1_head`, `acc_proto`, `f1_proto` | the same metrics for shared-head and nearest-global-prototype inference on the same trained model, where both exist (Table 9 and the common-rule rows of Table 7) |
| `acc_local_routed` | local-head variant: score when each test slide is routed through its own site (depends on knowing the site) |
| `fpkd_head_selfeval_per_site`, `fpkd_proto_per_site_routed` | faithful FedProtoKD: per-site-routed scores (Supplementary Table S5(a)) |
| `fedgh_random_partner_acc`, `fedgh_random_partner_same_class_frac`, `fedgh_other_class_partner_acc` | faithful FedGH: cross-site routing probes (Supplementary Table S5(b)) |
| `silhouette_class`, `xfm_prototype_cosine`, `fm_probe_balanced_acc` | alignment diagnostics (Table 10, Fig. 5, Supplementary Table S8) |
| `recall_<cancer>` | per-class recall for BRCA, COAD, STAD, LGG, LUAD, HNSC, SKCM, CESC, PAAD (Supplementary Tables S3, S4, S6) |
| `used_in` | where the arm is reported |
| `source_file` | the raw file the row was read from, relative to `raw_json/` |

Empty cells mean the quantity is not defined for that configuration (for
example, no prototype score for FedGH, which keeps no prototypes).

## Statistical procedure

`recompute_tables.py` follows the pipeline that produced the article:
per-seed values are rounded to six decimals before any statistic (the
backend differences of Table 11 and the per-class recalls use the unrounded
values); comparisons are paired by seed; the 95% confidence interval of a
mean difference is the 2.5th and 97.5th percentile of 10,000 bootstrap means
over seeds, drawn with `numpy.random.default_rng(0)` created afresh for each
comparison (in the per-cancer Supplementary Tables S3, S4 and S6 one
`default_rng(0)` stream is used in turn for the nine cancers, in table
order); p-values
are two-sided Wilcoxon signed-rank tests (`scipy.stats.wilcoxon`, SciPy
1.17.1), unadjusted, with Holm-adjusted values over the comparisons of each
of Tables 6 and 7; standard deviations over seeds are population standard
deviations, and the standard deviation of backend differences is the sample
standard deviation. With another SciPy version the exact Wilcoxon p-values
can differ in the last digit. The selection-grid values are stored with four
decimals, so recomputed means and standard deviations of Supplementary Table
S1 (computed from unrounded values) can differ in the fourth decimal.

Run `python recompute_tables.py` in this folder; it prints each table in
order.
