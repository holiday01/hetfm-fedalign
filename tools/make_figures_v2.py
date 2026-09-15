"""Figures 2-5 (v2) from the single-backend R2 batch summary
(<repo-root>/hetfm/results/r2_summary.json).  No hand-typed numbers.
Fig. 1 is drawn by fig1_overview_v2.py.  Error bars everywhere = SD over seeds."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("<repo-root>/hetfm/results")
OUT = Path(__file__).resolve().parents[1] / "figures"
S = json.loads((R / "r2_summary.json").read_text())
A = S["arms"]
LB = json.loads((R / "prereg_confirm.json").read_text())["LB"]
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False})
C = {"method": "#1b5e9c", "ctrl": "#7a7a7a", "base": "#c0732f",
     "lb": "#bbbbbb", "fpkd": "#a23b72", "fedgh": "#3a9d5d"}
ORDER = ["stratified", "balanced", "cancer_correlated", "adversarial"]
DISP = ["stratified", "balanced", "cancer-\ncorrelated", "adversarial"]


def ms(name, key="macro_acc"):
    return A[name][key]["mean"], A[name][key]["std"], A[name][key]["n"]


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig); print("wrote", OUT / name)


def fig2():
    names = [("plain_fedavg_Conch_v15", "Plain\nFedAvg"), ("A_Conch_v15_linear", "Homog.\nCONCH\n1 proj."),
             ("A_Conch_v15_3group_linear", "Homog.\nCONCH\n3 proj."), ("method_linear_balanced", "Protocol\n(ref.)"),
             ("abl_ce_only", "Protocol\nCE only"), ("fedgh_tied_balanced", "Tied\nFedGH")]
    if "localonly_balanced" in A and "macro_acc" in A["localonly_balanced"]:
        lo_m, lo_e = ms("localonly_balanced")[0], ms("localonly_balanced")[1]   # thirty-seed local-only floor
    else:
        lo_m, lo_e = LB, 0                              # historical Week-1 constant (no SD)
    vals = [lo_m] + [ms(n)[0] for n, _ in names]; errs = [lo_e] + [ms(n)[1] for n, _ in names]
    labels = ["Local-\nonly"] + [l for _, l in names]
    cols = [C["lb"], C["base"], C["ctrl"], C["ctrl"], C["method"], C["method"], C["fedgh"]]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = range(len(vals))
    ax.bar(x, vals, yerr=errs, capsize=3, color=cols, width=0.66)
    for i, v in enumerate(vals):
        ax.text(i, v + errs[i] + 0.015, f"{v:.3f}", ha="center", fontsize=6.3)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_ylabel("Macro accuracy (9-class)"); ax.set_ylim(0, 0.95)
    save(fig, "fig2_rq1_recovery.pdf")


def _series(pat):
    m, e = [], []
    for k in ORDER:
        n = pat(k)
        if n in A and "macro_acc" in A[n]:
            m.append(A[n]["macro_acc"]["mean"]); e.append(A[n]["macro_acc"]["std"])
        else:
            m.append(np.nan); e.append(0)
    return m, e


def fig3():
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(4)
    series = [(lambda k: f"method_linear_{k}", "Protocol (reference)", C["method"], "o", "-"),
              (lambda k: "abl_ce_only" if k == "balanced" else f"abl_ce_only_{k}", "Protocol, CE only", "#5a8fc4", "^", "-"),
              (lambda k: f"fedgh_tied_{k}", "Tied FedGH", C["fedgh"], "D", "-")]
    for pat, lab, col, mk, ls in series:
        m, e = _series(pat)
        ax.errorbar(x, m, yerr=e, marker=mk, ms=4.5, lw=1.5, ls=ls, color=col, label=lab, capsize=2)
    bb, be = [], []
    for k in ORDER:
        bl = A.get(f"base_{k}")
        if bl:
            best = max(bl, key=lambda b: bl[b]["macro_acc"]["mean"])
            bb.append(bl[best]["macro_acc"]["mean"]); be.append(bl[best]["macro_acc"]["std"])
        else:
            bb.append(np.nan); be.append(0)
    ax.errorbar(x, bb, yerr=be, marker="s", ms=4, lw=1.3, ls="--", color=C["base"],
                label="Best dimension-matching baseline", capsize=2)
    ax.set_xticks(list(x)); ax.set_xticklabels(DISP)
    ax.set_xlabel("Model-to-site assignment scheme")
    ax.set_ylabel("Macro accuracy (9-class)"); ax.set_ylim(0.3, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=6.3)
    save(fig, "fig3_rq4_inversion.pdf")


def fig4():
    arms = [(lambda k: f"method_linear_{k}", "Protocol (reference)", C["method"]),
            (lambda k: "abl_ce_only" if k == "balanced" else f"abl_ce_only_{k}", "Protocol, CE only", "#5a8fc4"),
            (lambda k: f"fedgh_tied_{k}", "Tied FedGH (server-trained head)", C["fedgh"]),
            (lambda k: f"fpkd_B_{k}", "Tuned FedProtoKD (local heads)", C["fpkd"])]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    w = 0.2; x = np.arange(4)
    for j, (pat, lab, col) in enumerate(arms):
        m, e = _series(pat)
        ax.bar(x + (j - 1.5) * w, m, w, yerr=e, capsize=1.5, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(DISP)
    ax.set_ylabel("Macro accuracy (9-class)"); ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=6.3)
    save(fig, "fig4_fedprotokd.pdf")


def fig5():
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 2.4))
    for a, key, lab, col in [(ax[0], "silhouette_class", "Class silhouette", C["method"]),
                             (ax[1], "xfm_prototype_cosine", "Cross-model prototype cosine", C["fpkd"])]:
        keys = [k for k in ORDER if key in A[f"method_linear_{k}"]]
        vals = [A[f"method_linear_{k}"][key]["per_seed"] for k in keys]
        parts = a.boxplot(vals, widths=0.55, showfliers=False, patch_artist=True)
        for b in parts["boxes"]:
            b.set(facecolor=col, alpha=0.25, edgecolor=col)
        for med in parts["medians"]:
            med.set(color=col, lw=1.4)
        for i, v in enumerate(vals):
            a.scatter(np.random.default_rng(i).normal(i + 1, 0.06, len(v)), v, s=5, color=col, alpha=0.7)
        a.axhline(0, color="#999", lw=0.6, ls=":")
        short = {"stratified": "str.", "balanced": "bal.", "cancer_correlated": "c-c.", "adversarial": "adv."}
        a.set_xticks(range(1, len(keys) + 1)); a.set_xticklabels([short[k] for k in keys])
        a.set_ylabel(lab)
    fig.tight_layout()
    save(fig, "fig5_diagnostics.pdf")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["2", "3", "4", "5"]
    for w in which:
        try:
            {"2": fig2, "3": fig3, "4": fig4, "5": fig5}[w]()
        except KeyError as e:
            print(f"fig{w}: missing arm {e}")
