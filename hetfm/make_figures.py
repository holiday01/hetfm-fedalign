"""Generate publication figures for manuscript_hetfm from REAL result JSONs.
No hardcoded results: every number is read from hetfm/results/*.json.
Output: manuscript_hetfm/figures/*.pdf (300 dpi, pub font sizes).
"""
import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

R = Path(os.environ.get("HETFM_RESULTS", "hetfm/results"))
OUT = Path(os.environ.get("HETFM_FIGDIR", "figures"))
OUT.mkdir(parents=True, exist_ok=True)
J = {f.stem: json.loads(f.read_text())
     for f in R.glob("*.json")}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
})
C = {"method": "#1b5e9c", "ctrl": "#7a7a7a", "base": "#c0732f",
     "lb": "#bbbbbb", "fpkd": "#a23b72"}


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---- Fig 1: protocol overview (schematic, no data) --------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    # One projector per foundation-model type, each with its own input
    # dimension: the subscripts must differ or the schematic reads as a
    # single shared projector.
    sites = [("Site A\nUNI v2\n12288-d", 4.2, "$P_{\\tau_1}$", "$12288\\!\\to\\!k$"),
             ("Site B\nCONCH v1.5\n6144-d", 2.4, "$P_{\\tau_2}$", "$6144\\!\\to\\!k$"),
             ("Site C\nVirchow2\n20480-d", 0.6, "$P_{\\tau_3}$", "$20480\\!\\to\\!k$")]
    for txt, y, psym, pdim in sites:
        ax.add_patch(FancyBboxPatch((0.3, y), 2.2, 1.5,
                     boxstyle="round,pad=0.06", fc="#eaf1f8", ec=C["method"]))
        ax.text(1.4, y + 0.75, txt, ha="center", va="center", fontsize=8.5)
        ax.add_patch(FancyBboxPatch((3.3, y + 0.15), 1.7, 1.2,
                     boxstyle="round,pad=0.05", fc="#fff", ec=C["base"]))
        ax.text(4.15, y + 0.75, f"projector\n{psym}\n{pdim}",
                ha="center", va="center", fontsize=7.5)
        ax.add_patch(FancyArrowPatch((2.5, y + 0.75), (3.3, y + 0.75),
                     arrowstyle="-|>", mutation_scale=12, color="#444"))
        ax.add_patch(FancyArrowPatch((5.0, y + 0.75), (6.2, 3.0),
                     arrowstyle="-|>", mutation_scale=12, color="#444"))
    ax.add_patch(FancyBboxPatch((6.2, 2.0), 3.3, 2.2,
                 boxstyle="round,pad=0.08", fc="#f3e9f1", ec=C["fpkd"]))
    ax.text(7.85, 3.1, "Server\nshared head $g_\\phi$\n+ global class\n"
            "prototypes $\\mu_c$", ha="center", va="center", fontsize=8.5)
    save(fig, "fig1_overview.pdf")


# ---- Fig 2: RQ1 recovery (pre-registered n=30) ------------------------------
def fig2():
    pr = J["prereg_confirm"]
    lb = pr["LB"]
    bplain = J["week3_rq1_linear"]["B_reference_plain_fedavg_REUSED"]
    a = pr["A_linear_homog_CONCH"]; m = pr["method_linear_balanced"]
    clo = pr["H1_RQ1_closure"]["mean"]
    labels = ["Local-only\nfloor", "Plain\nFedAvg", "Homog.\ncontrol",
              "Hetero.\nprotocol"]
    vals = [lb, bplain["mean"], a["mean"], m["mean"]]
    errs = [0, bplain["std"], a["std"], m["std"]]
    cols = [C["lb"], C["base"], C["ctrl"], C["method"]]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    x = range(4)
    ax.bar(x, vals, yerr=errs, capsize=4, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + (errs[i] or 0) + 0.02, f"{v:.3f}",
                ha="center", fontsize=7.5)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("Macro accuracy (9-class)")
    ax.set_ylim(0, 0.8)
    ax.annotate(f"gap-closure {clo:.2f}\n(parity bar $0.90$)",
                xy=(3, m["mean"]), xytext=(0.15, 0.71), fontsize=7,
                arrowprops=dict(arrowstyle="->", color="#444"))
    ax.set_title(f"Recovery over $n=${pr['n']} seeds", fontsize=8.5)
    save(fig, "fig2_rq1_recovery.pdf")


# ---- Fig 3: RQ4 assignment-skew inversion (n=30) -----------------
def fig3():
    s = J["seedext_rq4_n30"]["schemes"]
    order = ["stratified", "balanced", "cancer_correlated", "adversarial"]
    disp = ["stratified", "balanced", "cancer-\ncorrelated", "adversarial"]
    mm = [s[k]["method_linear"]["mean"] for k in order]
    me = [s[k]["method_linear"]["std"] for k in order]
    bb = []
    for k in order:
        bl = s[k]["baselines"]
        bb.append(max(v["mean"] for v in bl.values()))
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    x = range(4)
    ax.errorbar(x, mm, yerr=me, marker="o", ms=7, lw=2,
                color=C["method"], label="Heterogeneous protocol", capsize=4)
    ax.plot(x, bb, marker="s", ms=6, lw=1.6, ls="--",
            color=C["base"], label="Best dimension-matching baseline")
    for i, v in enumerate(mm):
        ax.text(i, v + me[i] + 0.015, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(list(x)); ax.set_xticklabels(disp)
    ax.set_xlabel("Model-to-site assignment scheme "
                  "(intuitive easy $\\rightarrow$ hard)")
    ax.set_ylabel("Macro accuracy (9-class)")
    ax.set_ylim(0, 0.95)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Thirty seeds per scheme", fontsize=8.5)
    save(fig, "fig3_rq4_inversion.pdf")


# ---- Fig 4: incrementality vs FedProtoKD ------------------------------------
def fig4():
    sc = J["seedext_rq4_n30"]["schemes"]
    schemes = ["balanced", "cancer_correlated", "adversarial"]
    disp = ["balanced", "cancer-correlated", "adversarial"]
    meth = [sc[k]["method_linear"]["mean"] for k in schemes]
    fp = [sc[k]["fedprotokd_B_tuned"]["mean"] for k in schemes]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    x = range(3); w = 0.36
    ax.bar([i - w / 2 for i in x], meth, w, color=C["method"],
           label="Heterogeneous protocol")
    ax.bar([i + w / 2 for i in x], fp, w, color=C["fpkd"],
           label="FedProtoKD (tuned, fair eval)")
    for i in x:
        ax.text(i - w / 2, meth[i] + 0.015, f"{meth[i]:.2f}",
                ha="center", fontsize=6.5)
        ax.text(i + w / 2, fp[i] + 0.015, f"{fp[i]:.2f}",
                ha="center", fontsize=6.5)
    ax.set_xticks(list(x)); ax.set_xticklabels(disp)
    ax.set_ylabel("Macro accuracy (9-class)")
    ax.set_ylim(0, 0.9)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Faithful FedProtoKD has no valid global classifier here "
                 "(see text)", fontsize=7.5)
    save(fig, "fig4_fedprotokd.pdf")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("done ->", OUT)
