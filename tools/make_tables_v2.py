"""LaTeX table bodies from results/r2_summary.json and the
frozen analysis files.  Nothing hand-typed.  Writes <out-dir>/tables/*.tex
(main text) and supp_*.tex (supplement).
Missing arms render as '--' so the document builds while the batch runs.
Wilcoxon p-values are recomputed exactly from the per-seed values (the
summary stores three significant figures, and printing that to two would
round twice); Tables 6 and 7 carry a Holm-adjusted p column over the
comparisons of each table; Table 7 includes the three head settings under a
common nearest-prototype rule."""
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
    return f"${p:#.2g}$" if p < 0.1 else f"${p:.2f}$"     # two significant figures below 0.1, trailing zero kept (0.070, not 0.07)
# ---- exact Wilcoxon p-values from the per-seed values (six decimals, as in the summary) --------
from scipy import stats as _wst
def _ps(n, sub=None):
    if sub is None: return dict(zip(A[n]["seeds"], A[n]["macro_acc"]["per_seed"]))
    return dict(zip(SEEDS, A[n][sub]["macro_acc"]["per_seed"]))          # base_* arms: all thirty seeds
def _arms_of(key):
    if key == "method_1hidden_minus_A_1hidden": return ("method_1hidden_balanced", None), ("A_Conch_v15_1hidden", None)
    a, b = key.split("_minus_", 1)
    if b in ("zeropad", "localpca", "procrustes"): return (a, None), ("base_" + a.replace("method_linear_", ""), b)
    return (a, None), (b, None)
for _k, _c in CON.items():
    if _c.get("wilcoxon_p") is None: continue
    (_a, _sa), (_b, _sb) = _arms_of(_k); _da, _db = _ps(_a, _sa), _ps(_b, _sb)
    _cm = [x for x in SEEDS if x in _da and x in _db]
    _c["wilcoxon_p"] = float(_wst.wilcoxon([_da[x] for x in _cm], [_db[x] for x in _cm]).pvalue)
def paired6(a, b):
    """Same statistics as run_r2_batch._paired, for per-seed lists a, b (paired by position)."""
    import statistics as _s
    d = [x - y for x, y in zip(a, b)]; rng = np.random.default_rng(0); arr = np.asarray(d)
    means = rng.choice(arr, size=(10000, arr.size), replace=True).mean(axis=1)
    return {"n": len(d), "mean_diff": round(_s.mean(d), 6), "n_a_gt_b": f"{sum(1 for x in d if x > 0)}/{len(d)}",
            "boot_ci95": [round(float(np.percentile(means, 2.5)), 6), round(float(np.percentile(means, 97.5)), 6)],
            "wilcoxon_p": float(_wst.wilcoxon(a, b).pvalue)}
def raw6(arm, key):
    """per-seed values of a raw-file field, rounded to six decimals like the summary, keyed by seed"""
    return {x: round(jload(arm, x)[key], 6) for x in SEEDS if jload(arm, x) and jload(arm, x).get(key) is not None}
def pm_raw(arm, key, d=3):
    import statistics as _s
    v = list(raw6(arm, key).values()); return f"${_s.mean(v):.{d}f}\\pm{_s.pstdev(v):.{d}f}$" if v else NA
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
def ptex(p):
    return "$<10^{-4}$" if p < 1e-4 else (f"${p:#.2g}$" if p < 0.1 else f"${p:.2f}$")
HOLM = {}                                                    # contrast label -> Holm-adjusted p within its table
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

# ---- controls table (balanced): exploratory, p unadjusted + Holm over the ten comparisons ---------
m = "method_linear_balanced"
ctl = [("CONCH, one linear projector shared by all sites (pre-specified control)", "A_Conch_v15_linear", f"{m}_minus_A_Conch_v15_linear"),
       ("CONCH, three group-tied linear projectors ($4.72$\\,M projector parameters)", "A_Conch_v15_3group_linear", f"{m}_minus_A_Conch_v15_3group_linear"),
       ("UNI v2, one linear projector", "A_UNI_v2_linear", f"{m}_minus_A_UNI_v2_linear"),
       ("UNI v2, three group-tied linear projectors ($9.44$\\,M)", "A_UNI_v2_3group_linear", f"{m}_minus_A_UNI_v2_3group_linear"),
       ("Virchow2, one linear projector", "A_Virchow2_linear", f"{m}_minus_A_Virchow2_linear"),
       ("Virchow2, three group-tied linear projectors ($15.73$\\,M)", "A_Virchow2_3group_linear", f"{m}_minus_A_Virchow2_3group_linear"),
       ("Plain FedAvg, CONCH, two-layer head, no projector", "plain_fedavg_Conch_v15", f"{m}_minus_plain_fedavg_Conch_v15"),
       (f"CONCH, one projector, cross-entropy only (vs.\\ cross-entropy-only protocol {pm('abl_ce_only')})", "A_Conch_v15_ce_only", "abl_ce_only_minus_A_Conch_v15_ce_only"),
       ("CONCH, three group-tied projectors, cross-entropy only (vs.\\ cross-entropy-only protocol)", "A_Conch_v15_3group_ce_only", "abl_ce_only_minus_A_Conch_v15_3group_ce_only"),
       (f"CONCH, one one-hidden-layer projector (vs.\\ one-hidden-layer protocol {pm('method_1hidden_balanced')})", "A_Conch_v15_1hidden", "method_1hidden_minus_A_1hidden")]
_h = holm([CON[k]["wilcoxon_p"] for _, _, k in ctl])
rows = []
for (label, arm, key), h in zip(ctl, _h):
    c = CON[key]; HOLM[key] = h
    rows.append(f"{label} & {pm(arm)} & {dci(c)} & {seeds_of(c)} & {pval(c)} & {ptex(h)} \\\\")
rows_write("tab_controls.tex", rows)

# ---- ablation table (balanced): exploratory, p unadjusted + Holm over the twelve comparisons ------
# The common-rule block scores the full protocol, the local heads and the no-head variant all by
# nearest global prototype, so its differences isolate the training change from the inference change.
np_full = raw6(m, "acc_proto"); np_loc = raw6("head_local", "acc_proto"); np_non = raw6("head_none_proto", "acc_proto")
NP = {"local_minus_shared": paired6([np_loc[x] for x in SEEDS], [np_full[x] for x in SEEDS]),
      "none_minus_shared": paired6([np_non[x] for x in SEEDS], [np_full[x] for x in SEEDS]),
      "none_minus_local": paired6([np_non[x] for x in SEEDS], [np_loc[x] for x in SEEDS])}
abl = [("Cross-entropy only (no anchor terms)", "abl_ce_only", CON[f"{m}_minus_abl_ce_only"], -1),
       ("Cross-entropy + prototype matching", "abl_ce_proto", CON[f"{m}_minus_abl_ce_proto"], -1),
       ("Cross-entropy + contrastive", "abl_ce_con", CON[f"{m}_minus_abl_ce_con"], -1),
       ("No head (no cross-entropy term): nearest global prototype", "head_none_proto", CON[f"{m}_minus_head_none_proto"], -1),
       ("Local heads, never averaged; nearest-prototype inference", "head_local", CON[f"{m}_minus_head_local"], -1),
       ("Local heads, never averaged", "head_local", NP["local_minus_shared"], +1),
       ("No head (no cross-entropy term)", "head_none_proto", NP["none_minus_shared"], +1),
       ("Head trained at the server on uploaded class means (tied FedGH)", "fedgh_tied_balanced", CON[f"{m}_minus_fedgh_tied_balanced"], -1),
       ("One hidden layer", "method_1hidden_balanced", CON[f"{m}_minus_method_1hidden_balanced"], -1),
       ("Two hidden layers", "method_2hidden_balanced", CON[f"{m}_minus_method_2hidden_balanced"], -1),
       ("One hidden layer, cross-entropy only", "abl_ce_only_1hidden_balanced", CON["method_1hidden_balanced_minus_abl_ce_only_1hidden_balanced"], -1),
       ("One hidden layer, cross-entropy + contrastive", "abl_ce_con_1hidden_balanced", CON["method_1hidden_balanced_minus_abl_ce_con_1hidden_balanced"], -1)]
_h = holm([c["wilcoxon_p"] for _, _, c, _ in abl])
cells = {}
for (label, arm, c, sg), h in zip(abl, _h):
    cells[label + "|" + arm + "|" + str(sg)] = f"{label} & {pm(arm)} & {pm(arm, 'macro_f1')} & {dci(c, sg)} & {seeds_of(c, sg)} & {pval(c)} & {ptex(h)} \\\\"
    HOLM["abl:" + arm + ("" if sg < 0 else ":np")] = h
def cell(label, arm, sg=-1): return cells[label + "|" + arm + "|" + str(sg)]
rows = [f"Full reference protocol (CE + matching + contrastive, averaged head, linear) & {pm(m)} & {pm(m, 'macro_f1')} & -- & -- & -- & -- \\\\",
        "\\multicolumn{7}{l}{\\emph{Loss terms (averaged head, linear projector)}} \\\\",
        cell("Cross-entropy only (no anchor terms)", "abl_ce_only"),
        cell("Cross-entropy + prototype matching", "abl_ce_proto"),
        cell("Cross-entropy + contrastive", "abl_ce_con"),
        "\\multicolumn{7}{l}{\\emph{Classifier head (linear projector, both prototype terms kept; scored by nearest global prototype)}} \\\\",
        cell("No head (no cross-entropy term): nearest global prototype", "head_none_proto"),
        cell("Local heads, never averaged; nearest-prototype inference", "head_local"),
        "\\multicolumn{7}{l}{\\emph{Common nearest-prototype rule (all three head settings; differences versus the full protocol under the same rule)}} \\\\",
        f"Shared averaged head (full protocol) & {pm_raw(m, 'acc_proto')} & {pm_raw(m, 'f1_proto')} & -- & -- & -- & -- \\\\",
        cell("Local heads, never averaged", "head_local", +1),
        cell("No head (no cross-entropy term)", "head_none_proto", +1),
        "\\multicolumn{7}{l}{\\emph{Head-training rule (linear projector, no anchor terms)}} \\\\",
        cell("Head trained at the server on uploaded class means (tied FedGH)", "fedgh_tied_balanced"),
        "\\multicolumn{7}{l}{\\emph{Projector depth (all terms, averaged head)}} \\\\",
        cell("One hidden layer", "method_1hidden_balanced"),
        cell("Two hidden layers", "method_2hidden_balanced"),
        "\\multicolumn{7}{l}{\\emph{Loss terms with the one-hidden-layer projector (differences versus the one-hidden-layer full protocol)}} \\\\",
        cell("One hidden layer, cross-entropy only", "abl_ce_only_1hidden_balanced"),
        cell("One hidden layer, cross-entropy + contrastive", "abl_ce_con_1hidden_balanced")]
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
                      ("Homogeneous CONCH, one projector, cross-entropy only", "A_Conch_v15_ce_only", None),
                      ("Homogeneous CONCH, three group projectors", "A_Conch_v15_3group_linear", None),
                      ("Homogeneous CONCH, three group projectors, cross-entropy only", "A_Conch_v15_3group_ce_only", None),
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
        bl = A[bn]; bname = max(bl, key=lambda b: bl[b]["macro_acc"]["mean"]); best = {"zeropad": "zero-padding", "localpca": "per-model PCA", "procrustes": "PCA + Procrustes"}[bname]; bval = f"${bl[bname]['macro_acc']['mean']:.3f}$"
        cb = CON.get(f"{mm}_minus_{bname}")
    c = CON.get(f"{mm}_minus_A_Conch_v15_linear")
    rows.append(f"{DISP[sc]} & {pm(mm)} & {best} ({bval}) & {dci(c)} & {pval(c)} ({seeds_of(c)}) \\\\")
rows_write("supp_S4_scheme.tex", rows)

# ---- supplementary S2: per-cancer vs the single-projector control (raw 30x9 arrays) ----------------
from scipy import stats
names = ['BRCA', 'COAD', 'STAD', 'LGG', 'LUAD', 'HNSC', 'SKCM', 'CESC', 'PAAD']
def pfmt(h):
    return "$<10^{-4}$" if h < 1e-4 else (f"${h:#.2g}$" if h < 0.1 else f"${h:.2f}$")
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
# ---- supplementary S8: per-cancer vs the single-projector and three-group Virchow2 federations ----
for ctrl_arm, fname in [("A_Virchow2_linear", "supp_S8_perclass_virchow1.tex"), ("A_Virchow2_3group_linear", "supp_S8_perclass_virchow3.tex")]:
    G2 = perclass(ctrl_arm); common2 = [s for s in M if s in G2]
    Mm = np.array([M[s] for s in common2], dtype=float); Aa = np.array([G2[s] for s in common2], dtype=float)
    rows_write(fname, perclass_rows(Mm, Aa))

# ---- backend rows (E8) ----------------------------------------------------------------------------
bk = R / "r2_batch" / "backend"; rows = []
if bk.exists():
    gpu = {s: jload("method_linear_balanced", s)["macro_acc"] for s in [62, 63, 64, 65, 66] if jload("method_linear_balanced", s)}
    for arm, lab in [("cpu_t24", "CPU (24 threads) vs.\\ GPU, same seed, $m=8$"), ("cpu_t2", "CPU (2 threads) vs.\\ GPU, same seed, $m=8$")]:
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
for tag, key in [("UniThreeMinusConchThree", "A_UNI_v2_3group_linear_minus_A_Conch_v15_3group_linear")]:
    lines.append(mac(tag + "D", dv(key))); lines.append(mac(tag + "CI", civ(key))); lines.append(mac(tag + "P", pv(key))); lines.append(mac(tag + "S", sv(key)))
for tag, key in [("NpLocalMinusShared", "local_minus_shared"), ("NpNoneMinusShared", "none_minus_shared"), ("NpNoneMinusLocal", "none_minus_local")]:
    c = NP[key]
    lines.append(mac(tag + "D", f"{c['mean_diff']:+.3f}")); lines.append(mac(tag + "CI", f"[{c['boot_ci95'][0]:+.3f},{c['boot_ci95'][1]:+.3f}]"))
    lines.append(mac(tag + "P", pval(c).strip("$"))); lines.append(mac(tag + "S", seeds_of(c)))
lines.append(mac("NpSharedAcc", pm_raw(m, "acc_proto")))
for tag, key in [("HolmRefMinusUniThree", f"{m}_minus_A_UNI_v2_3group_linear"),
                 ("HolmCeOnlyMinusConchThreeCe", "abl_ce_only_minus_A_Conch_v15_3group_ce_only"),
                 ("HolmOneHiddenPair", "method_1hidden_minus_A_1hidden"),
                 ("HolmRefMinusVirchowThree", f"{m}_minus_A_Virchow2_3group_linear"),
                 ("HolmLocalHeads", "abl:head_local"), ("HolmCeCon", "abl:abl_ce_con"), ("HolmCeOnly", "abl:abl_ce_only"),
                 ("HolmNpLocal", "abl:head_local:np"), ("HolmNpNone", "abl:head_none_proto:np")]:
    lines.append(mac(tag, ptex(HOLM[key]).strip("$")))
(OUT / "numbers.tex").write_text("\n".join(lines) + "\n"); print("wrote", OUT / "numbers.tex", len(lines), "macros")

# ---- copy into the build directory -------------------------------------------------------------------
dst = V2 / "access" / "tables"; dst.mkdir(exist_ok=True)
for f in list(OUT.glob("tab_*.tex")) + [OUT / "numbers.tex"]: shutil.copy(f, dst / f.name)
print("copied to", dst)


# ---- refresh the supplementary bodies between markers ------------------------------------------
supp = V2 / "supplementary.tex"; ss = supp.read_text(); nrep = 0
import re as _re
for name in ["supp_S1_counts", "supp_S2_perclass_single", "supp_S8_perclass_virchow1", "supp_S8_perclass_virchow3", "supp_S3_fpkd_faithful", "supp_S3_fedgh_faithful", "supp_S4_scheme", "supp_S6_diag_variants", "supp_S7_perclass_matched"]:
    body = (OUT / f"{name}.tex").read_text()
    pat = _re.compile(r"%<" + name + r">\n.*?%</" + name + r">\n", _re.S)
    if pat.search(ss):
        ss = pat.sub(lambda m: f"%<{name}>\n{body}%</{name}>\n", ss); nrep += 1
supp.write_text(ss); print("supplementary bodies refreshed:", nrep)
