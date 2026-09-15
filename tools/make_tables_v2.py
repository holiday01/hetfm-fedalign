"""LaTeX table bodies from results/r2_summary.json and the
frozen analysis files.  Nothing hand-typed.  Writes <out-dir>/tables/*.tex
(main text) and supp_*.tex (supplement).
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


# ---- main table (rows = configurations, columns = schemes) ------------------------------------------
rows = []
def sch(fn): return " & ".join(fn(sc) for sc in ORDER)
rows.append("Homogeneous CONCH, one projector (control) & " + sch(lambda sc: pm("A_Conch_v15_linear")) + " \\\\")
rows.append("Zero-padding + FedAvg & " + sch(lambda sc: base_pm(sc, "zeropad")) + " \\\\")
rows.append("Per-model PCA + FedAvg & " + sch(lambda sc: base_pm(sc, "localpca")) + " \\\\")
rows.append("PCA + Procrustes + FedAvg & " + sch(lambda sc: base_pm(sc, "procrustes")) + " \\\\")
rows.append("FedProtoKD (tuned) & " + sch(lambda sc: pm(f"fpkd_B_{sc}")) + " \\\\")
rows.append("FedGH (tied) & " + sch(lambda sc: pm(f"fedgh_tied_{sc}")) + " \\\\")
rows.append("Protocol, cross-entropy only & " + sch(lambda sc: pm("abl_ce_only" if sc == "balanced" else f"abl_ce_only_{sc}")) + " \\\\")
rows.append("Protocol (reference) & " + sch(lambda sc: pm(f"method_linear_{sc}")) + " \\\\")
rows.append("Protocol (reference), macro-F1 & " + sch(lambda sc: pm(f"method_linear_{sc}", "macro_f1")) + " \\\\")
rows.append("Reference $-$ CONCH control [95\\% CI] & " + sch(lambda sc: dci(CON.get(f"method_linear_{sc}_minus_A_Conch_v15_linear"))) + " \\\\")
rows_write("tab_main.tex", rows)

# ---- controls table (balanced) -----------------------------------------------------------
m = "method_linear_balanced"; rows = []
def crow(label, arm, key):
    c = CON.get(key); return f"{label} & {pm(arm)} & {dci(c)} & {seeds_of(c)} & {pval(c)} \\\\"
rows.append(crow("CONCH, one linear projector shared by all sites (submitted control)", "A_Conch_v15_linear", f"{m}_minus_A_Conch_v15_linear"))
rows.append(crow("CONCH, three group-tied linear projectors ($4.72$\\,M projector parameters)", "A_Conch_v15_3group_linear", f"{m}_minus_A_Conch_v15_3group_linear"))
rows.append(crow("UNI v2, one linear projector", "A_UNI_v2_linear", f"{m}_minus_A_UNI_v2_linear"))
rows.append(crow("UNI v2, three group-tied linear projectors ($9.44$\\,M)", "A_UNI_v2_3group_linear", f"{m}_minus_A_UNI_v2_3group_linear"))
rows.append(crow("Virchow2, one linear projector", "A_Virchow2_linear", f"{m}_minus_A_Virchow2_linear"))
rows.append(crow("Virchow2, three group-tied linear projectors ($15.73$\\,M)", "A_Virchow2_3group_linear", f"{m}_minus_A_Virchow2_3group_linear"))
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
rows.append("\\multicolumn{6}{l}{\\emph{Classifier head (linear projector, both prototype terms kept; scored by nearest global prototype)}} \\\\")
rows.append(arow("No head (no cross-entropy term): nearest global prototype", "head_none_proto"))
rows.append(arow("Local heads, never averaged; nearest-prototype inference", "head_local"))
rows.append("\\multicolumn{6}{l}{\\emph{Head-training rule (linear projector, no anchor terms)}} \\\\")
rows.append(arow("Head trained at the server on uploaded class means (tied FedGH)", "fedgh_tied_balanced"))
rows.append("\\multicolumn{6}{l}{\\emph{Projector depth (all terms, averaged head)}} \\\\")
rows.append(arow("One hidden layer", "method_1hidden_balanced"))
rows.append(arow("Two hidden layers", "method_2hidden_balanced"))
rows.append("\\multicolumn{6}{l}{\\emph{Loss terms with the one-hidden-layer projector (differences versus the one-hidden-layer full protocol)}} \\\\")
for lab, arm in [("One hidden layer, cross-entropy only", "abl_ce_only_1hidden_balanced"),
                 ("One hidden layer, cross-entropy + contrastive", "abl_ce_con_1hidden_balanced")]:
    c1 = CON.get(f"method_1hidden_balanced_minus_{arm}")
    rows.append(f"{lab} & {pm(arm)} & {pm(arm, 'macro_f1')} & {dci(c1, -1)} & {seeds_of(c1, -1)} & {pval(c1)} \\\\")
rows_write("tab_ablation.tex", rows)

# ---- inference-rule table: shared head versus nearest global prototype ----
rows = []
def hp(arm, key):
    if not has(arm): return NA
    if key in A[arm]: return pm(arm, key)
    if key == "acc_head" and arm.startswith("fedgh_tied"): return pm(arm, "macro_acc")   # FedGH: primary classifier is the head
    return NA                                              # prototype-primary arms (no head) have no head score
for lab, bal, adv in [("Reference protocol (anchors, averaged head)", "method_linear_balanced", "method_linear_adversarial"),
                      ("Cross-entropy only (averaged head)", "abl_ce_only", "abl_ce_only_adversarial"),
                      ("Tied FedGH (server-trained head)", "fedgh_tied_balanced", "fedgh_tied_adversarial"),
                      ("No head (anchors only)", "head_none_proto", None),
                      ("Homogeneous CONCH, one projector", "A_Conch_v15_linear", None),
                      ("Homogeneous CONCH, three group projectors", "A_Conch_v15_3group_linear", None),
                      ("Homogeneous UNI v2, one projector", "A_UNI_v2_linear", None),
                      ("Homogeneous UNI v2, three group projectors", "A_UNI_v2_3group_linear", None),
                      ("Homogeneous Virchow2, one projector", "A_Virchow2_linear", None),
                      ("Homogeneous Virchow2, three group projectors", "A_Virchow2_3group_linear", None),
                      ("Homogeneous CONCH, three groups, server-trained head (tied FedGH)", "fedgh_tied_homog3_Conch_v15", None)]:
    rows.append(" & ".join([lab, hp(bal, "acc_head"), hp(bal, "acc_proto"),
                            hp(adv, "acc_head") if adv else NA, hp(adv, "acc_proto") if adv else NA]) + " \\\\")
rows_write("tab_inference.tex", rows)

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
    rows.append(" & ".join([DISP[sc], pm(mm), pm(f"fpkd_B_{sc}"), dci(cf), pm(f"fedgh_tied_{sc}"), dci(cg)]) + " \\\\")
rows_write("tab_hetfl.tex", rows)

# ---- diagnostics table: per scheme (reference protocol) + variants (balanced) -----------------
def cell(arm, key):
    if not has(arm) or key not in A[arm]: return NA
    xs = np.array(A[arm][key]["per_seed"]); rng = np.random.default_rng(0)
    b = rng.choice(xs, size=(10000, len(xs)), replace=True).mean(1)
    return f"${xs.mean():+.3f}$ $[{np.percentile(b, 2.5):+.3f},{np.percentile(b, 97.5):+.3f}]$"
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
                   ("One hidden layer, cross-entropy only", "abl_ce_only_1hidden_balanced"),
                   ("One hidden layer, cross-entropy + contrastive", "abl_ce_con_1hidden_balanced"),
                   ("Homog.\\ CONCH, three group projectors", "A_Conch_v15_3group_linear"),
                   ("Homog.\\ UNI v2, three group projectors", "A_UNI_v2_3group_linear"),
                   ("Homog.\\ Virchow2, three group projectors", "A_Virchow2_3group_linear"),
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

# ---- supplementary S2: per-cancer vs the single-projector control (raw 30x9 arrays) ----------------
from scipy import stats
names = ['BRCA', 'COAD', 'STAD', 'LGG', 'LUAD', 'HNSC', 'SKCM', 'CESC', 'PAAD']
def pfmt(h):
    return "$<10^{-4}$" if h < 1e-4 else (f"${h:.2g}$" if h < 0.01 else f"${h:.2f}$")
def perclass_rows(Mm, Aa):
    ps, rws = [], []
    rng = np.random.default_rng(0)
    for j, n in enumerate(names):
        d = Mm[:, j] - Aa[:, j]
        try: p = stats.wilcoxon(d).pvalue
        except Exception: p = 1.0
        b = rng.choice(d, size=(10000, len(d)), replace=True).mean(1)
        ps.append(p); rws.append((n, Mm[:, j].mean(), Aa[:, j].mean(), d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), int((d > 0).sum())))
    order = np.argsort(ps); holm = [0.0] * 9
    for rank, i in enumerate(order): holm[i] = min(1.0, ps[i] * (9 - rank))
    for k in range(1, 9): holm[order[k]] = max(holm[order[k]], holm[order[k - 1]])
    out = []
    for (n, mm_, a, d, lo, hi, ng), h in zip(rws, holm):
        sig = "$+$" if (h < 0.05 and d > 0) else ("$-$" if (h < 0.05 and d < 0) else "n.s.")
        out.append(f"{n} & ${mm_:.3f}$ & ${a:.3f}$ & ${d:+.3f}$ & $[{lo:+.3f},\\,{hi:+.3f}]$ & {ng}/{Mm.shape[0]} & {pfmt(h)} & {sig} \\\\")
    return out
rows_write("supp_S2_perclass_single.tex", perclass_rows(np.load(R / "_perclass_M.npy"), np.load(R / "_perclass_A.npy")))

# ---- supplementary S3(a): faithful FedProtoKD at the headline seeds ------------------------------------
rows = []
for sc in ORDER:
    arm = f"fpkd_A_{sc}"
    if arm in A and "proto_per_site_routed" in A[arm]:
        e = A[arm]
        rows.append(f"{DISP[sc]} & {len(e['seeds'])} & ${e['head_selfeval_per_site']['mean']:.3f}\\pm{e['head_selfeval_per_site']['std']:.3f}$ & ${e['proto_per_site_routed']['mean']:.3f}\\pm{e['proto_per_site_routed']['std']:.3f}$ \\\\")
    else:
        rows.append(f"{DISP[sc]} & -- & -- & -- \\\\")
rows_write("supp_S3_fpkd_faithful.tex", rows)

# ---- supplementary S7: per-cancer vs the grouping-matched control ------------------------------------
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
    rows.append(f"{n} & ${mm_:.3f}$ & ${a:.3f}$ & ${d:+.3f}$ & $[{lo:+.3f},\\,{hi:+.3f}]$ & {ng}/{len(common)} & {pfmt(h)} & {sig} \\\\")
rows_write("supp_S7_perclass_matched.tex", rows)
(OUT / "supp_S7_n.txt").write_text(str(len(common)))

# ---- backend rows (E8) ----------------------------------------------------------------------------
bk = R / "r2_batch" / "backend"; rows = []
if bk.exists():
    gpu = {s: jload("method_linear_balanced", s)["macro_acc"] for s in [62, 63, 64, 65, 66] if jload("method_linear_balanced", s)}
    for arm, lab in [("cpu_t24", "CPU (24 threads) vs GPU, same seed, $m=8$"), ("cpu_t2", "CPU (2 threads) vs GPU, same seed, $m=8$")]:
        d = [json.loads((bk / f"{arm}_s{s}.json").read_text())["macro_acc"] - gpu[s] for s in gpu if (bk / f"{arm}_s{s}.json").exists()]
        if d:
            d = np.array(d)
            rows.append(f"{lab} & {len(d)} & ${d.mean():+.3f}$ & ${d.std(ddof=1) if len(d) > 1 else 0:.3f}$ & ${np.abs(d).max():.3f}$ \\\\")
rows_write("tab_backend_rows.tex", rows)
# seed-to-seed SD rows (population SD over the thirty seeds, as in Table 5): m=8 reference and the m=1 / m=16 gate sweeps
import statistics as _st
_sw = json.loads((R / "mincount_sweep_clean.json").read_text())["arms"]
_m8 = A["method_linear_balanced"]["macro_acc"]["std"]
_g = sorted(_st.pstdev(_sw[k]["per_seed"]) for k in ("m1", "m16"))
rows_write("tab_seed_rows.tex", [f"Seed to seed, one backend, $m=8$ (reference) & 30 & -- & ${_m8:.3f}$ & -- \\\\",
                                 f"Seed to seed, one backend, $m=1$ / $m=16$ gate sweeps & 30 & -- & ${_g[0]:.3f}$--${_g[1]:.3f}$ & -- \\\\"])

# ---- cost table ------------------------------------------------------------------------------------
an = json.loads((REV / "cost_analytic.json").read_text())
cost = json.loads((R / "r2_cost.json").read_text())["arms"] if (R / "r2_cost.json").exists() else {}
def meas(name):
    c = cost.get(name)
    return f"${c['per_round_s']:.2f}$ & ${c['total_min']:.1f}$ & ${c['peak_gpu_mem_mb']/1024:.2f}$" if c else "-- & -- & --"
P = an["protocol"]
SITES = {"UNI_v2": 36, "Conch_v15": 36, "Virchow2": 35}          # balanced scheme
def up_mb(params_by_fm, extra=2313 + 257):                       # head + one class mean with its count
    return {f: 4 * (params_by_fm[f] + extra) / 1e6 for f in params_by_fm}
h1 = up_mb(an["protocol_1hidden"]["projector_params"])
h1_round = sum(SITES[f] * h1[f] for f in SITES) / 1e3            # GB per round per direction
fp_round = an["fedprotokd_B"]["total_150_rounds_GB"] / 300
rows = [
 f"Reference protocol, linear projectors & ${P['trainable_params_total']/1e6:.2f}$ & $6.3$ / $12.6$ / $21.0$ & ${P['per_round_total_up_MB']/1000:.2f}$ & ${P['total_150_rounds_GB']:.0f}$ & {meas('method_linear_balanced')} \\\\",
 f"Cross-entropy only (same communication) & ${P['trainable_params_total']/1e6:.2f}$ & $6.3$ / $12.6$ / $21.0$ & ${P['per_round_total_up_MB']/1000:.2f}$ & ${P['total_150_rounds_GB']:.0f}$ & {meas('abl_ce_only')} \\\\",
 f"Tied FedGH (server-trained head) & ${an['fedgh_tied']['trainable_params_total']/1e6:.2f}$ & $6.3$ / $12.6$ / $21.0$ & ${an['fedgh_tied']['total_150_rounds_GB']/300:.2f}$ & ${an['fedgh_tied']['total_150_rounds_GB']:.0f}$ & {meas('fedgh_tied_balanced')} \\\\",
 f"Protocol, one-hidden-layer projectors & ${an['protocol_1hidden']['trainable_params_total']/1e6:.2f}$ & ${h1['Conch_v15']:.1f}$ / ${h1['UNI_v2']:.1f}$ / ${h1['Virchow2']:.1f}$ & ${h1_round:.2f}$ & ${2*150*h1_round:.0f}$ & {meas('method_1hidden_balanced')} \\\\",
 f"Tuned FedProtoKD & ${an['fedprotokd_B']['trainable_params_total']/1e6:.2f}$ & $6.3$ / $12.6$ / $21.0$ & ${fp_round:.2f}$ & ${an['fedprotokd_B']['total_150_rounds_GB']:.0f}$ & {meas('fpkd_B_balanced')} \\\\",
 f"Plain FedAvg, CONCH, two-layer head & ${an['plain_fedavg_conch']['trainable_params_total']/1e6:.2f}$ & $12.6$ & ${107*an['plain_fedavg_conch']['per_site_each_direction_MB']/1000:.2f}$ & ${an['plain_fedavg_conch']['total_150_rounds_GB']:.0f}$ & {meas('plain_fedavg_Conch_v15')} \\\\",
 f"Zero-padding + FedAvg & ${an['zeropad_fedavg']['trainable_params_total']/1e6:.2f}$ & $42.0$ & ${an['zeropad_fedavg']['per_round_total_up_MB']/1000:.2f}$ & ${an['zeropad_fedavg']['total_150_rounds_GB']:.0f}$ & {meas('zeropad')} \\\\",
 f"Per-model PCA / Procrustes + FedAvg & ${an['pca_or_procrustes_fedavg']['trainable_params_total']/1e6:.2f}$ & $0.5$ & ${107*an['pca_or_procrustes_fedavg']['per_site_each_direction_MB']/1000:.3f}$ & ${an['pca_or_procrustes_fedavg']['total_150_rounds_GB']:.0f}$ & {meas('procrustes')} \\\\",
]
rows_write("tab_cost.tex", rows)

# ---- numbers.tex: macros for text use; TBD until the arm exists --------------------------------------
TBD = "\\textcolor{red}{TBD}"
def mac(name, val): return f"\\newcommand{{\\{name}}}{{{val}}}"
def pmv(arm, key="macro_acc"):                    # keeps the $...$ so \pm works in text mode
    return pm(arm, key) if has(arm) else TBD
def dv(key):
    c = CON.get(key); return f"{c['mean_diff']:+.3f}" if c else TBD
def civ(key):
    c = CON.get(key); return f"[{c['boot_ci95'][0]:+.3f},{c['boot_ci95'][1]:+.3f}]" if c else TBD
def pv(key):
    c = CON.get(key); return pval(c).strip("$") if c else TBD
def sv(key):
    c = CON.get(key); return seeds_of(c) if c else TBD
lines = []
for sc in ORDER:
    tag = {"balanced": "Bal", "stratified": "Str", "cancer_correlated": "Cc", "adversarial": "Adv"}[sc]
    lines.append(mac(f"LocalOnly{tag}", pmv(f"localonly_{sc}")))
    lines.append(mac(f"LocalOnlyMean{tag}", f"{A[f'localonly_{sc}']['macro_acc']['mean']:.3f}" if has(f"localonly_{sc}") else TBD))
    fa = A.get(f"fpkd_A_{sc}", {})
    lines.append(mac(f"FpkdARouted{tag}", f"{fa['proto_per_site_routed']['mean']:.3f}" if "proto_per_site_routed" in fa else TBD))
    lines.append(mac(f"FpkdAHead{tag}", f"{fa['head_selfeval_per_site']['mean']:.3f}" if "head_selfeval_per_site" in fa else TBD))
    lines.append(mac(f"FpkdAN{tag}", str(len(fa["seeds"])) if "seeds" in fa else TBD))
for tag, arm in [("UniThree", "A_UNI_v2_3group_linear"), ("VirchowThree", "A_Virchow2_3group_linear"), ("FedghHomogThree", "fedgh_tied_homog3_Conch_v15")]:
    lines.append(mac(tag, pmv(arm)))
    lines.append(mac(tag + "Mean", f"{A[arm]['macro_acc']['mean']:.3f}" if has(arm) else TBD))
for tag, key in [("RefMinusUniThree", "method_linear_balanced_minus_A_UNI_v2_3group_linear"),
                 ("RefMinusVirchowThree", "method_linear_balanced_minus_A_Virchow2_3group_linear"),
                 ("UniThreeMinusOne", "A_UNI_v2_3group_linear_minus_A_UNI_v2_linear"),
                 ("VirchowThreeMinusOne", "A_Virchow2_3group_linear_minus_A_Virchow2_linear"),
                 ("FedghMinusFedghHomogThree", "fedgh_tied_balanced_minus_fedgh_tied_homog3_Conch_v15"),
                 ("FedghHomogThreeMinusCeThree", "fedgh_tied_homog3_Conch_v15_minus_A_Conch_v15_3group_ce_only"),
                 ("FedghMinusVirchowThree", "fedgh_tied_balanced_minus_A_Virchow2_3group_linear"),
                 ("CeOnlyMinusVirchowThree", "abl_ce_only_minus_A_Virchow2_3group_linear"),
                 ("VirchowThreeMinusConchThree", "A_Virchow2_3group_linear_minus_A_Conch_v15_3group_linear")]:
    lines.append(mac(tag + "D", dv(key))); lines.append(mac(tag + "CI", civ(key))); lines.append(mac(tag + "P", pv(key))); lines.append(mac(tag + "S", sv(key)))
_c = CON.get("method_linear_balanced_minus_A_Virchow2_3group_linear")
lines.append(mac("VirchowThreeMinusRef", f"{-_c['mean_diff']:.3f}" if _c else TBD))
(OUT / "numbers.tex").write_text("\n".join(lines) + "\n"); print("wrote", OUT / "numbers.tex", len(lines), "macros")

# ---- copy into the build directory -------------------------------------------------------------------
dst = V2 / "access" / "tables"; dst.mkdir(exist_ok=True)
for f in list(OUT.glob("tab_*.tex")) + [OUT / "numbers.tex"]: shutil.copy(f, dst / f.name)
print("copied to", dst)


# ---- refresh the supplementary bodies between markers ------------------------------------------
supp = V2 / "supplementary.tex"; ss = supp.read_text(); nrep = 0
import re as _re
for name in ["supp_S1_counts", "supp_S2_perclass_single", "supp_S3_fpkd_faithful", "supp_S3_fedgh_faithful", "supp_S4_scheme", "supp_S6_diag_variants", "supp_S7_perclass_matched"]:
    body = (OUT / f"{name}.tex").read_text()
    pat = _re.compile(r"%<" + name + r">\n.*?%</" + name + r">\n", _re.S)
    if pat.search(ss):
        ss = pat.sub(lambda m: f"%<{name}>\n{body}%</{name}>\n", ss); nrep += 1
supp.write_text(ss); print("supplementary bodies refreshed:", nrep)
