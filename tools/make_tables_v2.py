"""LaTeX table bodies from results/r2_summary.json and the analysis files
(mi_fm_y.json, perclass_ci.json, cost_analytic.json).  Nothing hand-typed.
Writes <out-dir>/tables/*.tex (main text, \\input) and supp_*.tex.
Missing arms render as '--' so the document builds while the batch runs."""
import json, math, shutil
from pathlib import Path
import numpy as np

R = Path("<repo-root>/hetfm/results")
REV = Path("<analysis-dir>")
V2 = Path(__file__).resolve().parents[1]
OUT = V2 / "tables"; OUT.mkdir(exist_ok=True)
S = json.loads((R / "r2_summary.json").read_text())
A, CON = S["arms"], S["contrasts"]
LB = json.loads((R / "prereg_confirm.json").read_text())["LB"]
SEEDS = S["seeds"]
ORDER = ["stratified", "balanced", "cancer_correlated", "adversarial"]
DISP = {"stratified": "Stratified", "balanced": "Balanced",
        "cancer_correlated": "Cancer-corr.", "adversarial": "Adversarial"}
NA = "--"


def has(n): return n in A and "macro_acc" in A[n]
def pm(name, key="macro_acc", d=3):
    if not has(name) or key not in A[name]: return NA
    m = A[name][key]; return f"${m['mean']:.{d}f}\\pm{m['std']:.{d}f}$"
def base_pm(sc, b, d=3):
    n = f"base_{sc}"
    if n not in A or b not in A[n]: return NA
    m = A[n][b]["macro_acc"]; return f"${m['mean']:.{d}f}\\pm{m['std']:.{d}f}$"
def ci(c): lo, hi = c["boot_ci95"]; return f"$[{lo:+.3f},{hi:+.3f}]$"
def dci(c, sign=1):
    if not c: return NA
    lo, hi = c["boot_ci95"]
    if sign < 0: lo, hi = -hi, -lo
    return f"${sign*c['mean_diff']:+.3f}$ $[{lo:+.3f},{hi:+.3f}]$"
def seeds_of(c, sign=1):
    if not c: return NA
    a, n = c["n_a_gt_b"].split("/")
    return f"{a}/{n}" if sign > 0 else f"{int(n)-int(a)}/{n}"
def pval(c):
    if not c or c.get("wilcoxon_p") is None: return NA
    p = c["wilcoxon_p"]
    if p < 1e-4: return "$<10^{-4}$"
    return f"${p:.2g}$" if p < 0.01 else f"${p:.2f}$"
def rows_write(name, rows):
    (OUT / name).write_text("\n".join(rows) + "\n"); print("wrote", OUT / name, len(rows), "rows")
def jload(name, seed):
    p = R / "r2_batch" / f"{name}_s{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---- main table --------------------------------------------------------------------
rows = []
for sc in ORDER:
    m = f"method_linear_{sc}"; ce = "abl_ce_only" if sc == "balanced" else f"abl_ce_only_{sc}"
    d = CON.get(f"{m}_minus_A_Conch_v15_linear")
    rows.append(" & ".join([DISP[sc], f"${LB:.3f}$", pm("plain_fedavg_Conch_v15"), pm("A_Conch_v15_linear"),
                            base_pm(sc, "zeropad"), base_pm(sc, "localpca"), base_pm(sc, "procrustes"),
                            pm(f"fpkd_B_{sc}"), pm(f"fedgh_tied_{sc}"), pm(m), pm(m, "macro_f1"), dci(d)]) + " \\\\")
rows_write("tab_main.tex", rows)

# ---- controls table (balanced) -----------------------------------------------------------
m = "method_linear_balanced"; rows = []
def crow(label, arm, key):
    c = CON.get(key); return f"{label} & {pm(arm)} & {dci(c)} & {seeds_of(c)} & {pval(c)} \\\\"
rows.append(crow("CONCH, one linear projector shared by all sites (submitted control)", "A_Conch_v15_linear", f"{m}_minus_A_Conch_v15_linear"))
rows.append(crow("CONCH, three group-tied linear projectors (capacity-matched)", "A_Conch_v15_3group_linear", f"{m}_minus_A_Conch_v15_3group_linear"))
rows.append(crow("UNI v2, one linear projector", "A_UNI_v2_linear", f"{m}_minus_A_UNI_v2_linear"))
rows.append(crow("Virchow2, one linear projector", "A_Virchow2_linear", f"{m}_minus_A_Virchow2_linear"))
rows.append(crow("Plain FedAvg, CONCH, two-layer head, no projector", "plain_fedavg_Conch_v15", f"{m}_minus_plain_fedavg_Conch_v15"))
c = CON.get("abl_ce_only_minus_A_Conch_v15_ce_only")
rows.append(f"CONCH, one projector, cross-entropy only (vs.\\ cross-entropy-only protocol {pm('abl_ce_only')}) & {pm('A_Conch_v15_ce_only')} & {dci(c)} & {seeds_of(c)} & {pval(c)} \\\\")
c = CON.get("abl_ce_only_minus_A_Conch_v15_3group_ce_only")
rows.append(f"CONCH, three group-tied projectors, cross-entropy only (vs.\\ cross-entropy-only protocol) & {pm('A_Conch_v15_3group_ce_only')} & {dci(c)} & {seeds_of(c)} & {pval(c)} \\\\")
c = CON.get("method_1hidden_minus_A_1hidden")
rows.append(f"CONCH, one one-hidden-layer projector (vs.\\ one-hidden-layer protocol {pm('method_1hidden_balanced')}) & {pm('A_Conch_v15_1hidden')} & {dci(c)} & {seeds_of(c)} & {pval(c)} \\\\")
rows_write("tab_controls.tex", rows)

# ---- ablation table (balanced) --------------------------------------------------------------
rows = []
def arow(label, arm):
    c = CON.get(f"{m}_minus_{arm}")   # method minus variant -> flip sign for variant minus full
    return f"{label} & {pm(arm)} & {pm(arm, 'macro_f1')} & {dci(c, -1)} & {seeds_of(c, -1)} & {pval(c)} \\\\"
rows.append(f"Full reference protocol (CE + matching + contrastive, averaged head, linear) & {pm(m)} & {pm(m, 'macro_f1')} & -- & -- & -- \\\\")
rows.append("\\multicolumn{6}{l}{\\emph{Loss terms (averaged head, linear projector)}} \\\\")
rows.append(arow("Cross-entropy only (no anchor terms)", "abl_ce_only"))
rows.append(arow("Cross-entropy + prototype matching", "abl_ce_proto"))
rows.append(arow("Cross-entropy + contrastive", "abl_ce_con"))
rows.append("\\multicolumn{6}{l}{\\emph{Classifier head (both anchor terms kept)}} \\\\")
rows.append(arow("No head: nearest global prototype", "head_none_proto"))
rows.append(arow("Local heads, never averaged; nearest-prototype inference", "head_local"))
rows.append(arow("Server-trained head on class means, no anchors (tied FedGH)", "fedgh_tied_balanced"))
rows.append("\\multicolumn{6}{l}{\\emph{Projector depth (all terms, averaged head)}} \\\\")
rows.append(arow("One hidden layer", "method_1hidden_balanced"))
rows.append(arow("Two hidden layers", "method_2hidden_balanced"))
c1 = CON.get("method_1hidden_balanced_minus_abl_ce_only_1hidden_balanced")
rows.append(f"One hidden layer, cross-entropy only (vs.\\ one-hidden-layer full) & {pm('abl_ce_only_1hidden_balanced')} & {pm('abl_ce_only_1hidden_balanced', 'macro_f1')} & {dci(c1, -1)} & {seeds_of(c1, -1)} & {pval(c1)} \\\\")
rows_write("tab_ablation.tex", rows)

# ---- heterogeneous-FL methods per scheme ------------------------------------------------------
rows = []
faith = {}
for sc in ORDER:
    mm = f"method_linear_{sc}"
    rp = sf = oc = None
    if has(f"fedgh_faithful_{sc}"):
        rr = [jload(f"fedgh_faithful_{sc}", s) for s in A[f"fedgh_faithful_{sc}"]["seeds"]]
        rr = [r for r in rr if r and r.get("cross_site_probe")]
        if rr:
            rp = np.mean([r["cross_site_probe"]["random_partner_acc"] for r in rr])
            sf = np.mean([r["cross_site_probe"]["random_partner_same_class_frac"] for r in rr])
            oc = np.mean([r["cross_site_probe"]["other_class_partner_acc"] for r in rr])
    faith[sc] = (rp, sf, oc)
    cf = CON.get(f"{mm}_minus_fpkd_B_{sc}"); cg = CON.get(f"{mm}_minus_fedgh_tied_{sc}")
    rows.append(" & ".join([DISP[sc], pm(mm), pm(f"fpkd_B_{sc}"), dci(cf), pm(f"fedgh_tied_{sc}"), dci(cg),
                            pm(f"fedgh_faithful_{sc}"),
                            f"${rp:.3f}$ (${sf:.2f}$)" if rp is not None else NA,
                            f"${oc:.3f}$" if oc is not None else NA]) + " \\\\")
rows_write("tab_hetfl.tex", rows)

# ---- diagnostics table: per scheme (reference protocol) + variants (balanced) -----------------
def cell(arm, key):
    if not has(arm) or key not in A[arm]: return NA
    xs = np.array(A[arm][key]["per_seed"]); rng = np.random.default_rng(0)
    b = rng.choice(xs, size=(10000, len(xs)), replace=True).mean(1)
    return f"${xs.mean():+.3f}\\pm{xs.std():.3f}$ $[{np.percentile(b, 2.5):+.3f},{np.percentile(b, 97.5):+.3f}]$"
rows = []
for sc in ORDER:
    arm = f"method_linear_{sc}"
    n = A[arm]["macro_acc"]["n"] if has(arm) else 0
    rows.append(f"{DISP[sc]} & {n} & {cell(arm, 'silhouette_class')} & {cell(arm, 'xfm_prototype_cosine')} & {cell(arm, 'fm_probe_balanced_acc')} \\\\")
rows_write("tab_diag.tex", rows)
# supplementary S6: variants on the balanced scheme
rows = []
for label, arm in [("Full reference protocol", "method_linear_balanced"), ("Cross-entropy only", "abl_ce_only"),
                   ("Cross-entropy + matching", "abl_ce_proto"), ("Cross-entropy + contrastive", "abl_ce_con"),
                   ("No head, nearest prototype", "head_none_proto"), ("Local heads", "head_local"),
                   ("One hidden layer", "method_1hidden_balanced"), ("Two hidden layers", "method_2hidden_balanced"),
                   ("Homog.\\ CONCH, three group projectors", "A_Conch_v15_3group_linear"),
                   ("Homog.\\ CONCH, one projector", "A_Conch_v15_linear")]:
    rows.append(f"{label} & {pm(arm)} & {cell(arm, 'silhouette_class')} & {cell(arm, 'xfm_prototype_cosine')} & {cell(arm, 'fm_probe_balanced_acc')} \\\\")
rows_write("supp_S6_diag_variants.tex", rows)

# ---- supplementary S3: faithful degeneracy for FedGH -------------------------------------------
rows = []
for sc in ORDER:
    rp, sf, oc = faith[sc]
    if rp is None: continue
    rows.append(f"{DISP[sc]} & {pm(f'fedgh_faithful_{sc}')} & ${rp:.3f}$ & ${sf:.2f}$ & ${oc:.3f}$ & {pm(f'fedgh_tied_{sc}')} \\\\")
rows_write("supp_S3_fedgh_faithful.tex", rows)

# ---- supplementary S4: per-scheme summary ---------------------------------------------------------
rows = []
for sc in ORDER:
    mm = f"method_linear_{sc}"; bn = f"base_{sc}"
    best = NA; bval = NA
    if bn in A:
        bl = A[bn]; bname = max(bl, key=lambda b: bl[b]["macro_acc"]["mean"]); best = bname; bval = f"${bl[bname]['macro_acc']['mean']:.3f}$"
        cb = CON.get(f"{mm}_minus_{bname}")
    c = CON.get(f"{mm}_minus_A_Conch_v15_linear")
    rows.append(f"{DISP[sc]} & {pm(mm)} & {best} ({bval}) & {dci(c)} & {pval(c)} ({seeds_of(c)}) \\\\")
rows_write("supp_S4_scheme.tex", rows)

# ---- supplementary S7: per-cancer vs the capacity-matched control ------------------------------------
from scipy import stats
names = ['BRCA', 'COAD', 'STAD', 'LGG', 'LUAD', 'HNSC', 'SKCM', 'CESC', 'PAAD']
def perclass(arm):
    return {s: jload(arm, s)["per_class_recall"] for s in SEEDS if jload(arm, s)}
M = perclass("method_linear_balanced"); G = perclass("A_Conch_v15_3group_linear")
common = [s for s in M if s in G]
ps, rws = [], []
rng = np.random.default_rng(0)
for j, n in enumerate(names):
    d = np.array([M[s][j] - G[s][j] for s in common]); a = np.array([G[s][j] for s in common]); mm_ = np.array([M[s][j] for s in common])
    try: p = stats.wilcoxon(d).pvalue
    except Exception: p = 1.0
    b = rng.choice(d, size=(10000, len(d)), replace=True).mean(1)
    ps.append(p); rws.append((n, mm_.mean(), a.mean(), d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), int((d > 0).sum())))
order = np.argsort(ps); holm = [0.0] * 9
for rank, i in enumerate(order): holm[i] = min(1.0, ps[i] * (9 - rank))
for k in range(1, 9): holm[order[k]] = max(holm[order[k]], holm[order[k - 1]])
rows = []
for (n, mm_, a, d, lo, hi, ng), h in zip(rws, holm):
    sig = "$+$" if (h < 0.05 and d > 0) else ("$-$" if (h < 0.05 and d < 0) else "n.s.")
    rows.append(f"{n} & ${mm_:.3f}$ & ${a:.3f}$ & ${d:+.3f}$ & $[{lo:+.3f},\\,{hi:+.3f}]$ & {ng}/{len(common)} & ${h:.2g}$ & {sig} \\\\")
rows_write("supp_S7_perclass_matched.tex", rows)
(OUT / "supp_S7_n.txt").write_text(str(len(common)))

# ---- backend rows (E8) ----------------------------------------------------------------------------
bk = R / "r2_batch" / "backend"; rows = []
if bk.exists():
    gpu = {s: jload("method_linear_balanced", s)["macro_acc"] for s in [62, 63, 64, 65, 66] if jload("method_linear_balanced", s)}
    for arm, lab in [("cpu_t24", "CPU (24 threads) vs GPU, same seed"), ("cpu_t2", "CPU (2 threads) vs GPU, same seed")]:
        d = [json.loads((bk / f"{arm}_s{s}.json").read_text())["macro_acc"] - gpu[s] for s in gpu if (bk / f"{arm}_s{s}.json").exists()]
        if d:
            d = np.array(d)
            rows.append(f"{lab} & {len(d)} & ${d.mean():+.3f}$ & ${d.std(ddof=1) if len(d) > 1 else 0:.3f}$ & ${np.abs(d).max():.3f}$ \\\\")
rows_write("tab_backend_rows.tex", rows)

# ---- cost table ------------------------------------------------------------------------------------
an = json.loads((REV / "cost_analytic.json").read_text())
cost = json.loads((R / "r2_cost.json").read_text())["arms"] if (R / "r2_cost.json").exists() else {}
def meas(name):
    c = cost.get(name)
    return f"${c['per_round_s']:.2f}$ & ${c['total_min']:.1f}$ & ${c['peak_gpu_mem_mb']/1024:.2f}$" if c else "-- & -- & --"
P = an["protocol"]
rows = [
 f"Reference protocol, linear projectors & ${P['trainable_params_total']/1e6:.2f}$ & $6.3$ / $12.6$ / $21.0$ & ${P['per_round_total_up_MB']/1024:.2f}$ & ${P['total_150_rounds_GB']:.0f}$ & {meas('method_linear_balanced')} \\\\",
 f"Cross-entropy only (same communication) & ${P['trainable_params_total']/1e6:.2f}$ & $6.3$ / $12.6$ / $21.0$ & ${P['per_round_total_up_MB']/1024:.2f}$ & ${P['total_150_rounds_GB']:.0f}$ & {meas('abl_ce_only')} \\\\",
 f"Tied FedGH (server-trained head) & ${an['fedgh_tied']['trainable_params_total']/1e6:.2f}$ & $6.3$ / $12.6$ / $21.0$ & ${P['per_round_total_up_MB']/1024:.2f}$ & ${an['fedgh_tied']['total_150_rounds_GB']:.0f}$ & {meas('fedgh_tied_balanced')} \\\\",
 f"Protocol, one-hidden-layer projectors & ${an['protocol_1hidden']['trainable_params_total']/1e6:.2f}$ & $13.1$ / $25.7$ / $42.5$ & -- & -- & {meas('method_1hidden_balanced')} \\\\",
 f"Tuned FedProtoKD & ${an['fedprotokd_B']['trainable_params_total']/1e6:.2f}$ & $6.3$ / $12.6$ / $21.0$ & -- & ${an['fedprotokd_B']['total_150_rounds_GB']:.0f}$ & {meas('fpkd_B_balanced')} \\\\",
 f"Plain FedAvg, CONCH, two-layer head & ${an['plain_fedavg_conch']['trainable_params_total']/1e6:.2f}$ & $12.6$ & ${107*an['plain_fedavg_conch']['per_site_each_direction_MB']/1024:.2f}$ & ${an['plain_fedavg_conch']['total_150_rounds_GB']:.0f}$ & {meas('plain_fedavg_Conch_v15')} \\\\",
 f"Zero-padding + FedAvg & ${an['zeropad_fedavg']['trainable_params_total']/1e6:.2f}$ & $42.0$ & ${an['zeropad_fedavg']['per_round_total_up_MB']/1024:.2f}$ & ${an['zeropad_fedavg']['total_150_rounds_GB']:.0f}$ & {meas('zeropad')} \\\\",
 f"Per-model PCA / Procrustes + FedAvg & ${an['pca_or_procrustes_fedavg']['trainable_params_total']/1e6:.2f}$ & $0.5$ & ${107*an['pca_or_procrustes_fedavg']['per_site_each_direction_MB']/1024:.3f}$ & ${an['pca_or_procrustes_fedavg']['total_150_rounds_GB']:.0f}$ & {meas('procrustes')} \\\\",
]
rows_write("tab_cost.tex", rows)

# ---- copy into the build directory -------------------------------------------------------------------
dst = V2 / "access" / "tables"; dst.mkdir(exist_ok=True)
for f in OUT.glob("tab_*.tex"): shutil.copy(f, dst / f.name)
print("copied to", dst)
