"""Recompute the statistics of the article from the per-seed result files.

Usage:  python recompute_tables.py            (needs numpy and scipy; SciPy 1.17.1 was used)

Reads per_seed_results.csv, label_information_per_seed.csv and
selection_grid_per_seed.csv from this folder and prints, table by table, the
means, standard deviations, paired differences with 95% bootstrap confidence
intervals, seeds on which a difference is positive, two-sided Wilcoxon
signed-rank p-values (unadjusted and, for Tables 6 and 7, Holm-adjusted over
the comparisons of the table), and the backend differences of Table 11.

Procedure (identical to the pipeline that produced the article):
  * per-seed values are rounded to six decimals before any statistic
    (backend differences in Table 11 use the unrounded values);
  * comparisons are paired by seed; the 95% CI of the mean difference is the
    2.5th/97.5th percentile of 10,000 bootstrap means over seeds, drawn with
    numpy.random.default_rng(0) created afresh for each comparison;
  * standard deviations over seeds are population SDs (as in Table 5);
    the SD of backend differences is the sample SD.
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROWS = list(csv.DictReader(open(HERE / "per_seed_results.csv")))
SCHEMES = ["stratified", "balanced", "cancer_correlated", "adversarial"]
CANCERS = ["BRCA", "COAD", "STAD", "LGG", "LUAD", "HNSC", "SKCM", "CESC", "PAAD"]
BY = defaultdict(dict)
for r in ROWS:
    BY[r["arm"]][int(r["seed"])] = r


def vals(arm, key="macro_acc", rnd=True):
    out = {}
    for s, r in BY[arm].items():
        if r[key] not in ("", "nan"):
            out[s] = round(float(r[key]), 6) if rnd else float(r[key])
    return out


def pm(arm, key="macro_acc"):
    v = list(vals(arm, key).values())
    return f"{st.mean(v):.3f}±{st.pstdev(v):.3f}" if v else "--"


def paired(a, b):
    """a, b: {seed: value}; returns mean difference a-b, bootstrap CI, seeds a>b, Wilcoxon p."""
    common = sorted(set(a) & set(b))
    x = [a[s] for s in common]
    y = [b[s] for s in common]
    d = np.asarray([p - q for p, q in zip(x, y)])
    rng = np.random.default_rng(0)
    means = rng.choice(d, size=(10000, d.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, 2.5), np.percentile(means, 97.5)
    p = stats.wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
    tp = stats.ttest_rel(x, y).pvalue if np.any(d != 0) else 1.0
    return {"n": len(d), "d": float(st.mean(d.tolist())), "lo": float(lo), "hi": float(hi),
            "pos": int((d > 0).sum()), "p": float(p), "tp": float(tp)}


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def fp(p):
    return "<1e-4" if p < 1e-4 else (f"{p:#.2g}" if p < 0.1 else f"{p:.2f}")


def fd(c, sign=1):
    lo, hi = (c["lo"], c["hi"]) if sign > 0 else (-c["hi"], -c["lo"])
    return f"{sign * c['d']:+.3f} [{lo:+.3f},{hi:+.3f}]"


def seeds(c, sign=1):
    return f"{c['pos'] if sign > 0 else c['n'] - c['pos']}/{c['n']}"


def header(t):
    print("\n" + t + "\n" + "-" * len(t))


# ---------------------------------------------------------------- Table 3
header("Table 3  label information carried by model identity (mean over 30 seeds)")
li = defaultdict(list)
for r in csv.DictReader(open(HERE / "label_information_per_seed.csv")):
    li[r["scheme"]].append(r)
for sc in SCHEMES:
    I = [float(r["I_FM_Y_bits"]) for r in li[sc]]
    cpm = [np.mean([int(r["classes_seen_UNI_v2"]), int(r["classes_seen_CONCH_v1.5"]), int(r["classes_seen_Virchow2"])]) for r in li[sc]]
    print(f"{sc:18s} I(FM;Y) {st.mean(I):.2f}±{st.pstdev(I):.2f}  I/H(Y) {st.mean(float(r['I_over_H_Y']) for r in li[sc]):.2f}  "
          f"identity-only acc {st.mean(float(r['identity_only_macro_acc']) for r in li[sc]):.2f}  classes/model {st.mean(cpm):.1f}")

# ---------------------------------------------------------------- Table 5
header("Table 5  consolidated results (macro accuracy, mean±SD over seeds)")
ce = {sc: ("abl_ce_only" if sc == "balanced" else f"abl_ce_only_{sc}") for sc in SCHEMES}
lines = [("Homogeneous CONCH, one projector", lambda sc: pm("A_Conch_v15_linear")),
         ("Zero-padding + FedAvg", lambda sc: pm(f"base_{sc}:zeropad")),
         ("Per-model PCA + FedAvg", lambda sc: pm(f"base_{sc}:localpca")),
         ("PCA + Procrustes + FedAvg", lambda sc: pm(f"base_{sc}:procrustes")),
         ("FedProtoKD (tuned)", lambda sc: pm(f"fpkd_B_{sc}")),
         ("FedGH (tied)", lambda sc: pm(f"fedgh_tied_{sc}")),
         ("Protocol, cross-entropy only", lambda sc: pm(ce[sc])),
         ("Protocol (reference)", lambda sc: pm(f"method_linear_{sc}")),
         ("Protocol (reference), macro-F1", lambda sc: pm(f"method_linear_{sc}", "macro_f1")),
         ("Reference - CONCH control [CI]", lambda sc: fd(paired(vals(f"method_linear_{sc}"), vals("A_Conch_v15_linear"))))]
print(f"{'':34s}" + "".join(f"{sc:>26s}" for sc in SCHEMES))
for lab, fn in lines:
    print(f"{lab:34s}" + "".join(f"{fn(sc):>26s}" for sc in SCHEMES))
print("plain FedAvg (CONCH):", pm("plain_fedavg_Conch_v15"), "  local-only floor:",
      {sc: pm(f"localonly_{sc}") for sc in SCHEMES})

# ---------------------------------------------------------------- Table 6
header("Table 6  homogeneous controls, balanced scheme (exploratory; p unadjusted and Holm over the 10 rows)")
m = "method_linear_balanced"
t6 = [("CONCH, one linear projector (pre-specified control)", m, "A_Conch_v15_linear"),
      ("CONCH, three group-tied linear projectors", m, "A_Conch_v15_3group_linear"),
      ("UNI v2, one linear projector", m, "A_UNI_v2_linear"),
      ("UNI v2, three group-tied linear projectors", m, "A_UNI_v2_3group_linear"),
      ("Virchow2, one linear projector", m, "A_Virchow2_linear"),
      ("Virchow2, three group-tied linear projectors", m, "A_Virchow2_3group_linear"),
      ("Plain FedAvg, CONCH, two-layer head", m, "plain_fedavg_Conch_v15"),
      ("CONCH, one projector, CE only (vs CE-only protocol)", "abl_ce_only", "A_Conch_v15_ce_only"),
      ("CONCH, three groups, CE only (vs CE-only protocol)", "abl_ce_only", "A_Conch_v15_3group_ce_only"),
      ("CONCH, one 1-hidden projector (vs 1-hidden protocol)", "method_1hidden_balanced", "A_Conch_v15_1hidden")]
res = [paired(vals(a), vals(b)) for _, a, b in t6]
for (lab, a, b), c, h in zip(t6, res, holm([c["p"] for c in res])):
    print(f"{lab:56s} {pm(b):>12s}  {fd(c):>24s}  {seeds(c):>6s}  p={fp(c['p']):>6s}  Holm={fp(h):>6s}  t-test p={fp(c['tp'])}")

# ---------------------------------------------------------------- Table 7
header("Table 7  ablation, balanced scheme (exploratory; p unadjusted and Holm over the 12 comparisons)")
t7 = [("Cross-entropy only", m, "abl_ce_only", "macro_acc", -1),
      ("Cross-entropy + prototype matching", m, "abl_ce_proto", "macro_acc", -1),
      ("Cross-entropy + contrastive", m, "abl_ce_con", "macro_acc", -1),
      ("No head: nearest global prototype (vs head-scored full)", m, "head_none_proto", "macro_acc", -1),
      ("Local heads; nearest prototype (vs head-scored full)", m, "head_local", "macro_acc", -1),
      ("Server-trained head (tied FedGH)", m, "fedgh_tied_balanced", "macro_acc", -1),
      ("One hidden layer", m, "method_1hidden_balanced", "macro_acc", -1),
      ("Two hidden layers", m, "method_2hidden_balanced", "macro_acc", -1),
      ("1 hidden, CE only (vs 1-hidden full)", "method_1hidden_balanced", "abl_ce_only_1hidden_balanced", "macro_acc", -1),
      ("1 hidden, CE + contrastive (vs 1-hidden full)", "method_1hidden_balanced", "abl_ce_con_1hidden_balanced", "macro_acc", -1),
      ("Common rule: local heads - shared head (both nearest prototype)", "head_local", m, "acc_proto", +1),
      ("Common rule: no head - shared head (both nearest prototype)", "head_none_proto", m, "acc_proto", +1)]
res = [paired(vals(a, k), vals(b, k)) for _, a, b, k, _ in t7]
print(f"full protocol: head {pm(m)} (F1 {pm(m, 'macro_f1')}); nearest prototype {pm(m, 'acc_proto')} (F1 {pm(m, 'f1_proto')})")
for (lab, a, b, k, sg), c, h in zip(t7, res, holm([c["p"] for c in res])):
    arm = b if sg < 0 else a
    f1 = pm(arm, "f1_proto" if k == "acc_proto" and arm == m else "macro_f1")
    print(f"{lab:66s} {pm(arm, k):>12s}  F1 {f1:>12s}  {fd(c, sg):>24s}  {seeds(c, sg):>6s}  p={fp(c['p']):>7s}  Holm={fp(h):>7s}  t-test p={fp(c['tp'])}")
c = paired(vals("head_none_proto", "acc_proto"), vals("head_local", "acc_proto"))
print(f"{'Common rule: no head - local heads (text)':66s} {'':12s}  {fd(c):>24s}  {seeds(c):>6s}  p={fp(c['p']):>6s}  t-test p={fp(c['tp'])}")

# ---------------------------------------------------------------- Table 8
header("Table 8  model-heterogeneous methods per scheme (reference minus method)")
for sc in SCHEMES:
    mm = f"method_linear_{sc}"
    c1, c2 = paired(vals(mm), vals(f'fpkd_B_{sc}')), paired(vals(mm), vals(f'fedgh_tied_{sc}'))
    print(f"{sc:18s} protocol {pm(mm)}  FedProtoKD {pm(f'fpkd_B_{sc}')}  ref-FPKD {fd(c1)} (p={fp(c1['p'])}, t-test p={fp(c1['tp'])})"
          f"  FedGH {pm(f'fedgh_tied_{sc}')}  ref-FedGH {fd(c2)} (p={fp(c2['p'])}, t-test p={fp(c2['tp'])})")

# ---------------------------------------------------------------- Table 9
header("Table 9  shared head versus nearest global prototype (mean±SD)")
for lab, arm in [("Reference protocol, balanced", "method_linear_balanced"), ("Reference protocol, adversarial", "method_linear_adversarial"),
                 ("Cross-entropy only, balanced", "abl_ce_only"), ("Cross-entropy only, adversarial", "abl_ce_only_adversarial"),
                 ("Tied FedGH, balanced (head only)", "fedgh_tied_balanced"), ("Tied FedGH, adversarial (head only)", "fedgh_tied_adversarial"),
                 ("No head (prototype only)", "head_none_proto"), ("CONCH, one projector", "A_Conch_v15_linear"),
                 ("CONCH, one projector, CE only", "A_Conch_v15_ce_only"), ("CONCH, three groups", "A_Conch_v15_3group_linear"),
                 ("CONCH, three groups, CE only", "A_Conch_v15_3group_ce_only"), ("UNI v2, one projector", "A_UNI_v2_linear"),
                 ("UNI v2, three groups", "A_UNI_v2_3group_linear"), ("Virchow2, one projector", "A_Virchow2_linear"),
                 ("Virchow2, three groups", "A_Virchow2_3group_linear"), ("CONCH three groups, tied FedGH (head only)", "fedgh_tied_homog3_Conch_v15")]:
    head = pm(arm, "acc_head") if vals(arm, "acc_head") else (pm(arm) if "fedgh" in arm else "--")
    print(f"{lab:44s} head {head:>12s}   prototype {pm(arm, 'acc_proto'):>12s}")

# ---------------------------------------------------------------- Table 10
header("Table 10  alignment diagnostics of the reference protocol (mean and 95% bootstrap CI over seeds)")
for sc in SCHEMES:
    out = []
    for k in ("silhouette_class", "xfm_prototype_cosine", "fm_probe_balanced_acc"):
        x = np.array(list(vals(f"method_linear_{sc}", k).values()))
        if x.size == 0:
            out.append("--")
            continue
        b = np.random.default_rng(0).choice(x, size=(10000, x.size), replace=True).mean(1)
        out.append(f"{x.mean():+.3f} [{np.percentile(b, 2.5):+.3f},{np.percentile(b, 97.5):+.3f}]")
    print(f"{sc:18s} silhouette {out[0]}  cross-model cosine {out[1]}  model-identity probe {out[2]}")

# ---------------------------------------------------------------- Table 11
header("Table 11  seed versus backend variability (balanced, reference configuration)")


def backend_row(lab, a, b):
    va, vb = vals(a, rnd=False), vals(b, rnd=False)
    common = sorted(set(va) & set(vb))
    d = np.array([va[s] - vb[s] for s in common])
    print(f"{lab:52s} n={len(d):2d}  mean {d.mean():+.3f}  SD {d.std(ddof=1):.3f}  max|d| {np.abs(d).max():.3f}")


for g in (1, 16):
    cpu = {s: r for s, r in BY[f"gate_sweep_mixed_m{g}"].items() if r["backend"].startswith("cpu")}
    BY[f"_cpu_m{g}"] = cpu
    backend_row(f"CPU vs GPU, same seed, m={g} (mixed CPU/GPU batch)", f"_cpu_m{g}", f"gate_sweep_m{g}")
n_ident = sum(1 for g in (1, 16) for s, r in BY[f"gate_sweep_mixed_m{g}"].items()
              if r["backend"] == "gpu" and float(r["macro_acc"]) == float(BY[f"gate_sweep_m{g}"][s]["macro_acc"]))
print(f"{'GPU vs GPU, same seed, m=1 / m=16':52s} bit-identical pairs: {n_ident}")
backend_row("CPU (24 threads) vs GPU, same seed, m=8", "backend_cpu_t24", "method_linear_balanced")
backend_row("CPU (2 threads) vs GPU, same seed, m=8", "backend_cpu_t2", "method_linear_balanced")
print(f"{'Seed to seed, one backend, m=8 (population SD)':52s} {st.pstdev(vals('method_linear_balanced').values()):.3f}")
print(f"{'Seed to seed, one backend, m=1 / m=16 gate sweeps':52s} "
      f"{st.pstdev(vals('gate_sweep_m1').values()):.3f} / {st.pstdev(vals('gate_sweep_m16').values()):.3f}")

# ---------------------------------------------------------------- text: gate sweep, 450 rounds, grouped controls
header("Section III-D  minimum-count gate sweep (single GPU batch)")
for g in (1, 16):
    c = paired(vals(f"gate_sweep_m{g}"), vals("gate_sweep_m8"))
    print(f"m={g:2d} {pm(f'gate_sweep_m{g}')} vs m=8 {pm('gate_sweep_m8')}: {c['d']:+.3f}, Wilcoxon p={c['p']:.2f}, t-test p={c['tp']:.2f}, higher on {c['pos']}/30")
header("Sections V-C to V-E  450-round checks (common seeds)")
for lab, a, b in [("protocol 450 - 150", "method_linear_balanced_r450", "method_linear_balanced"),
                  ("CONCH 1-proj 450 - 150", "A_Conch_v15_linear_r450", "A_Conch_v15_linear"),
                  ("CONCH 3-group 450 - 150", "A_Conch_v15_3group_linear_r450", "A_Conch_v15_3group_linear"),
                  ("protocol - CONCH 3-group, 450 rounds", "method_linear_balanced_r450", "A_Conch_v15_3group_linear_r450"),
                  ("protocol - CONCH 1-proj, 450 rounds", "method_linear_balanced_r450", "A_Conch_v15_linear_r450"),
                  ("FedGH - CE-only, 450 rounds", "fedgh_tied_balanced_r450", "abl_ce_only_r450"),
                  ("CE-only - protocol, 450 rounds", "abl_ce_only_r450", "method_linear_balanced_r450")]:
    if lab.endswith("450 - 150"):   # three-way comparison uses the 16 seeds common to the three 450-round arms
        common = set(vals("method_linear_balanced_r450")) & set(vals("A_Conch_v15_linear_r450")) & set(vals("A_Conch_v15_3group_linear_r450"))
        c = paired({s: v for s, v in vals(a).items() if s in common}, {s: v for s, v in vals(b).items() if s in common})
    elif "CONCH" in lab:
        common = set(vals("method_linear_balanced_r450")) & set(vals("A_Conch_v15_linear_r450")) & set(vals("A_Conch_v15_3group_linear_r450"))
        c = paired({s: v for s, v in vals(a).items() if s in common}, {s: v for s, v in vals(b).items() if s in common})
    else:
        c = paired(vals(a), vals(b))
    print(f"{lab:40s} n={c['n']:2d} {fd(c)}  p={fp(c['p'])}  t-test p={fp(c['tp'])}")
header("Section V-C  grouped controls (differences not in Table 6)")
for lab, a, b in [("UNI v2 3-group - 1-proj", "A_UNI_v2_3group_linear", "A_UNI_v2_linear"),
                  ("Virchow2 3-group - 1-proj", "A_Virchow2_3group_linear", "A_Virchow2_linear"),
                  ("CONCH 3-group - 1-proj", "A_Conch_v15_3group_linear", "A_Conch_v15_linear"),
                  ("Virchow2 3-group - CONCH 3-group", "A_Virchow2_3group_linear", "A_Conch_v15_3group_linear"),
                  ("UNI v2 3-group - CONCH 3-group", "A_UNI_v2_3group_linear", "A_Conch_v15_3group_linear"),
                  ("tied FedGH (hetero) - tied FedGH (CONCH 3-group)", "fedgh_tied_balanced", "fedgh_tied_homog3_Conch_v15"),
                  ("tied FedGH (CONCH 3-group) - CE-only CONCH 3-group", "fedgh_tied_homog3_Conch_v15", "A_Conch_v15_3group_ce_only"),
                  ("tied FedGH - Virchow2 1-proj", "fedgh_tied_balanced", "A_Virchow2_linear"),
                  ("tied FedGH - Virchow2 3-group", "fedgh_tied_balanced", "A_Virchow2_3group_linear")]:
    c = paired(vals(a), vals(b))
    print(f"{lab:52s} {fd(c)}  {seeds(c)}  p={fp(c['p'])}  t-test p={fp(c['tp'])}")

# ---------------------------------------------------------------- supplementary per-cancer tables
header("Supplementary Tables S3, S4, S6  per-cancer recall, protocol minus control (Holm over nine cancers)")
for lab, ctrl in [("S3 vs single-projector CONCH", "A_Conch_v15_linear"), ("S4 vs three-group CONCH", "A_Conch_v15_3group_linear"),
                  ("S6(a) vs single-projector Virchow2", "A_Virchow2_linear"), ("S6(b) vs three-group Virchow2", "A_Virchow2_3group_linear")]:
    rng = np.random.default_rng(0)            # one stream per table, used in turn for the nine cancers
    rows, ps = [], []
    for k in CANCERS:
        a, b = vals(m, f"recall_{k}", rnd=False), vals(ctrl, f"recall_{k}", rnd=False)
        common = sorted(set(a) & set(b))
        d = np.array([a[s] - b[s] for s in common])
        try:
            p = stats.wilcoxon(d).pvalue
        except ValueError:
            p = 1.0
        bm = rng.choice(d, size=(10000, d.size), replace=True).mean(1)
        rows.append((k, np.mean([a[s] for s in common]), np.mean([b[s] for s in common]), d.mean(),
                     np.percentile(bm, 2.5), np.percentile(bm, 97.5), int((d > 0).sum()), len(d)))
        ps.append(p)
    print(lab)
    for (k, ma, mb, dd, lo, hi, npos, n), hh in zip(rows, holm(ps)):
        print(f"  {k:5s} protocol {ma:.3f}  control {mb:.3f}  diff {dd:+.3f} [{lo:+.3f},{hi:+.3f}]  {npos}/{n}  Holm p {fp(hh)}")

# ---------------------------------------------------------------- supplementary S1 and S5
header("Supplementary Table S1  selection grid (five seeds; per-seed values stored with four decimals, so means and SDs can differ from the table in the fourth decimal)")
grid = defaultdict(list)
for r in csv.DictReader(open(HERE / "selection_grid_per_seed.csv")):
    grid[r["grid_row"]].append(float(r["macro_acc"]))
for k, v in grid.items():
    print(f"{k:14s} {st.mean(v):.4f} (SD {st.pstdev(v):.4f})")
header("Supplementary Table S5  per-site extractors")
for sc in SCHEMES:
    print(f"{sc:18s} faithful FedProtoKD head {pm(f'fpkd_A_{sc}', 'fpkd_head_selfeval_per_site')}  routed prototype "
          f"{pm(f'fpkd_A_{sc}', 'fpkd_proto_per_site_routed')}  | faithful FedGH routed {pm(f'fedgh_faithful_{sc}')}  "
          f"random partner {st.mean(vals(f'fedgh_faithful_{sc}', 'fedgh_random_partner_acc').values()):.3f}  "
          f"same-cancer fraction {st.mean(vals(f'fedgh_faithful_{sc}', 'fedgh_random_partner_same_class_frac').values()):.2f}  "
          f"other-cancer partner {st.mean(vals(f'fedgh_faithful_{sc}', 'fedgh_other_class_partner_acc').values()):.3f}")
