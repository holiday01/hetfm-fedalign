"""hetfm.diagnostics — P2: alignment-quality diagnostics (proposal
contribution #4). Two mechanistic metrics on the trained headline model:

  silhouette : silhouette score of projected test embeddings w.r.t. the
               9 classes in the shared latent space (higher = classes are
               separable after alignment).
  xfm_proto  : cross-FM prototype agreement = mean pairwise cosine, per
               class, between the per-FM class-mean projected embeddings
               (higher = different frozen FMs are mapped to a *shared*
               geometry, i.e. alignment, not mere co-location).

Run on a few seeds at the linear headline config. Output diagnostics.json
+ a figure. Gives the evidence the Discussion currently asserts.
"""
import os
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                                       # noqa: E402
from hetfm.run_hetfm import probe_fm_dims                       # noqa: E402
from hetfm.proto_anchor import C, train                        # noqa: E402

RESULTS = Path(__file__).parent / "results"
SEEDS = [62, 63, 64]
ROUNDS = 150
CFG = dict(depth="linear", k=256, tie="fm", lam_proto=1.0, lam_con=1.0)


@torch.no_grad()
def _diag(internals, client_fm):
    bank = internals["bank"]; tc = internals["test_cache"]
    bank.eval()
    Z, Y, FM = [], [], []
    for cid, (X, y) in tc.items():
        if cid not in bank.client_fm:
            continue
        z = bank.get(cid)(X).cpu().numpy()
        Z.append(z); Y.append(y.cpu().numpy())
        FM += [bank.client_fm[cid]] * len(z)
    Z = np.concatenate(Z); Y = np.concatenate(Y); FM = np.array(FM)

    # (1) silhouette w.r.t. class (subsample for speed)
    try:
        from sklearn.metrics import silhouette_score
        n = len(Z)
        idx = (np.random.default_rng(0).choice(n, 3000, replace=False)
               if n > 3000 else np.arange(n))
        sil = float(silhouette_score(Z[idx], Y[idx]))
    except Exception as e:                                      # noqa: BLE001
        sil = float("nan"); print("silhouette err:", e, flush=True)

    # (2) cross-FM prototype agreement
    fms = sorted(set(FM))
    cos_per_class = []
    for c in range(C):
        means = []
        for f in fms:
            m = (Y == c) & (FM == f)
            if m.sum() >= 5:
                v = Z[m].mean(0)
                means.append(v / (np.linalg.norm(v) + 1e-9))
        if len(means) >= 2:
            cos_per_class.append(float(np.mean(
                [a @ b for a, b in combinations(means, 2)])))
    xfm = float(np.mean(cos_per_class)) if cos_per_class else float("nan")
    return sil, xfm, cos_per_class


if __name__ == "__main__":
    print("##### P2 diagnostics #####", flush=True)
    fmd = probe_fm_dims(assign.FEAT_ROOT, assign.FMS)
    rows = []
    for sd in SEEDS:
        print(f"\n--- diagnostics seed {sd} ---", flush=True)
        _, sp, gt, _ = assign.build_canonical_partition(seed=sd)
        clab = assign.client_labels_from_splits(sp)
        asn = assign.assign_fms(list(sp), "balanced", sd, client_label=clab)
        rsp, rgt = assign.route(sp, gt, asn)
        r = train(rsp, rgt, asn, fmd, rounds=ROUNDS, seed=sd,
                  log_every=ROUNDS, return_internals=True, **CFG)
        sil, xfm, cpc = _diag(r["_internals"], asn)
        print(f"  seed {sd}: silhouette={sil:.4f} xfm_proto_cos={xfm:.4f}",
              flush=True)
        rows.append({"seed": sd, "macro_acc": round(r["macro_acc"], 4),
                     "silhouette_class": round(sil, 4),
                     "xfm_prototype_cosine": round(xfm, 4),
                     "per_class_xfm_cosine": [round(x, 4) for x in cpc]})

    def _ms(key):
        xs = [r[key] for r in rows]
        return {"mean": round(float(np.mean(xs)), 4),
                "std": round(float(np.std(xs)), 4)}
    out = {"config": CFG, "seeds": SEEDS, "rows": rows,
           "silhouette_class": _ms("silhouette_class"),
           "xfm_prototype_cosine": _ms("xfm_prototype_cosine"),
           "interpretation": (
               "Positive silhouette => classes are separable in the shared "
               "latent space after alignment. High cross-FM prototype "
               "cosine => the three frozen FMs are mapped into a shared "
               "geometry (alignment), not merely co-located; this is the "
               "mechanistic evidence for the alignment claims (contribution "
               "#4).")}
    (RESULTS / "diagnostics.json").write_text(json.dumps(out, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"savefig.dpi": 300, "font.size": 8,
                             "axes.labelsize": 8.5,
                             "xtick.labelsize": 7.5,
                             "ytick.labelsize": 7.5})
        fig, ax = plt.subplots(1, 2, figsize=(3.5, 2.2))
        s = [r["silhouette_class"] for r in rows]
        x = [r["xfm_prototype_cosine"] for r in rows]
        ax[0].bar(range(len(rows)), s, color="#1b5e9c")
        ax[0].set_xticks(range(len(rows)))
        ax[0].set_xticklabels([r["seed"] for r in rows])
        ax[0].set_ylabel("Silhouette (class)"); ax[0].set_xlabel("Seed")
        ax[1].bar(range(len(rows)), x, color="#a23b72")
        ax[1].set_xticks(range(len(rows)))
        ax[1].set_xticklabels([r["seed"] for r in rows])
        ax[1].set_ylabel("Cross-FM prototype cosine")
        ax[1].set_xlabel("Seed")
        fig.tight_layout()
        fig.savefig(os.environ.get("HETFM_FIGDIR", "figures") + "/"
                    "figures/fig5_diagnostics.pdf", bbox_inches="tight")
        print("wrote fig5_diagnostics.pdf", flush=True)
    except Exception as e:                                      # noqa: BLE001
        print("figure err:", e, flush=True)
    print("\n=== diagnostics ==="); print(json.dumps(out, indent=2))
    print("\n##### P2 COMPLETE #####", flush=True)
