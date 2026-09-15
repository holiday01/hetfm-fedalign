"""Fig. 1 (v2): protocol overview WITH the federated communication loop.
Schematic only, no data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
                     "font.size": 7.5})
BLUE, ORANGE, PURPLE, GREY = "#1b5e9c", "#c0732f", "#a23b72", "#555555"

fig, ax = plt.subplots(figsize=(7.16, 3.9))
ax.set_xlim(0, 10); ax.set_ylim(0, 7.6); ax.axis("off")

sites = [("Site A\nUNI v2\n12288-d", 4.0, "$P_{\\tau_1}$", "$12288\\!\\to\\!k$"),
         ("Site B\nCONCH v1.5\n6144-d", 2.2, "$P_{\\tau_2}$", "$6144\\!\\to\\!k$"),
         ("Site C\nVirchow2\n20480-d", 0.4, "$P_{\\tau_3}$", "$20480\\!\\to\\!k$")]
for txt, y, psym, pdim in sites:
    ax.add_patch(FancyBboxPatch((0.25, y), 1.55, 1.45, boxstyle="round,pad=0.05",
                                fc="#eaf1f8", ec=BLUE, lw=1))
    ax.text(1.02, y + 0.72, txt, ha="center", va="center", fontsize=7)
    ax.add_patch(FancyBboxPatch((2.1, y + 0.12), 1.55, 1.2, boxstyle="round,pad=0.05",
                                fc="#fff", ec=ORANGE, lw=1))
    ax.text(2.87, y + 0.72, f"projector\n{psym}\n{pdim}", ha="center", va="center",
            fontsize=6.8)
    ax.add_patch(FancyArrowPatch((1.8, y + 0.72), (2.1, y + 0.72), arrowstyle="-|>",
                                 mutation_scale=9, color=GREY))
    # local head copy + loss
    ax.add_patch(FancyBboxPatch((3.85, y + 0.12), 1.35, 1.2, boxstyle="round,pad=0.05",
                                fc="#fff", ec=PURPLE, lw=0.8, ls="--"))
    ax.text(4.52, y + 0.72, "local copies\n$P_i,\\,g_i$\n$\\mathcal{L}_i$: CE +\nanchors",
            ha="center", va="center", fontsize=6.2)
    ax.add_patch(FancyArrowPatch((3.65, y + 0.72), (3.85, y + 0.72), arrowstyle="-|>",
                                 mutation_scale=9, color=GREY))
    # upload arrow (site -> server)
    ax.add_patch(FancyArrowPatch((5.2, y + 0.95), (6.55, 3.4), arrowstyle="-|>",
                                 mutation_scale=9, color=BLUE, lw=1,
                                 connectionstyle="arc3,rad=0.0"))
    # download arrow (server -> site)
    ax.add_patch(FancyArrowPatch((6.55, 2.75), (5.2, y + 0.45), arrowstyle="-|>",
                                 mutation_scale=9, color=ORANGE, lw=1, ls="--"))

# server box
ax.add_patch(FancyBboxPatch((6.6, 1.65), 3.2, 2.8, boxstyle="round,pad=0.08",
                            fc="#f3e9f1", ec=PURPLE, lw=1.2))
ax.text(8.2, 4.05, "Server", ha="center", va="center", fontsize=8, weight="bold")
ax.text(8.2, 2.95, "shared head $g_\\phi$: average over all sites\n"
        "projectors $P_\\tau$: average only within\n   the sites holding architecture $\\tau$\n"
        "prototypes $\\mu_c$: count-weighted class\n   means, EMA $\\rho$, gate $m$",
        ha="center", va="center", fontsize=6.4)

# step legend (the loop)
ax.text(0.25, 7.45, "One communication round", fontsize=7.5, weight="bold", va="top")
steps = ["1  server $\\to$ site $i$: $g_\\phi$, $P_{\\tau(i)}$, $\\{\\mu_c\\}$  (dashed orange)",
         "2  local update of copies $P_i, g_i$ on $\\mathcal{L}_i$ for $E$ epochs",
         "3  site $i$ $\\to$ server: $P_i$, $g_i$, class means $(\\bar z_{i,c}, n_{i,c})$  (solid blue)",
         "4  server: average $g_\\phi$; average each $P_\\tau$ within its architecture;\n"
         "    update $\\mu_c$; then step 1 again.  Encoders, embeddings, slides never move."]
ax.text(0.25, 7.1, "\n".join(steps), fontsize=6.3, va="top", linespacing=1.35)
fig.savefig(OUT / "fig1_overview.pdf", bbox_inches="tight")
print("wrote", OUT / "fig1_overview.pdf")
