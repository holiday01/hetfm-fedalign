"""hetfm.run_perclass — P1: pre-registered per-cancer RQ2 (H2).

Proposal §6.5 pre-registered: "Per-class complementarity (H2) tested per
cancer with Holm-Bonferroni correction across the 9 classes." §7.2: even
one significant per-class gain is a defensible contribution. This was never
computed; here it is, on the SAME pre-registered seed set as the headline
confirmation (62-91, n=30), headline linear config.

Per cancer c: paired difference (method_recall_c - A_recall_c) over the 30
seeds; two-sided Wilcoxon signed-rank; Holm-Bonferroni across the 9
classes; 'significant_help' = Holm-adjusted p<0.05 AND mean diff>0. All
outcomes reported regardless of direction. Output perclass_rq2.json.
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                                       # noqa: E402
from hetfm.run_hetfm import probe_fm_dims                       # noqa: E402
from hetfm.proto_anchor import train                           # noqa: E402
from data.partition import CANCER_TYPE_MAP                      # noqa: E402

RESULTS = Path(__file__).parent / "results"
SEEDS = list(range(62, 92))                 # n=30, pre-registered set
ROUNDS = 150
BEST_FM = "Conch_v15"
CFG = dict(depth="linear", k=256, tie="fm", lam_proto=1.0, lam_con=1.0)
NAME = {v: k.replace("TCGA-", "") for k, v in CANCER_TYPE_MAP.items()}  # 0..8


def _run(scheme, sd):
    _, sp, gt, _ = assign.build_canonical_partition(seed=sd)
    clab = assign.client_labels_from_splits(sp)
    asn = assign.assign_fms(list(sp), scheme, sd, client_label=clab)
    rsp, rgt = assign.route(sp, gt, asn)
    fmd = probe_fm_dims(assign.FEAT_ROOT, assign.FMS)
    r = train(rsp, rgt, asn, fmd, rounds=ROUNDS, seed=sd, log_every=ROUNDS,
              per_class=True, **CFG)
    return r["per_class_recall"]                 # length-9 (NaN if absent)


def _holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    run_max = 0.0
    for rank, i in enumerate(idx):
        a = min(1.0, (m - rank) * pvals[i])
        run_max = max(run_max, a)
        adj[i] = run_max
    return adj


if __name__ == "__main__":
    print(f"##### P1 per-cancer RQ2, n={len(SEEDS)} seeds 62-91 #####",
          flush=True)
    M, A = [], []
    for sd in SEEDS:
        print(f"\n--- perclass seed {sd} ---", flush=True)
        M.append(_run("balanced", sd))
        A.append(_run(f"homogeneous:{BEST_FM}", sd))
        np.save(RESULTS / "_perclass_M.npy", np.array(M))
        np.save(RESULTS / "_perclass_A.npy", np.array(A))
    M, A = np.array(M), np.array(A)              # [30, 9]

    from scipy import stats
    rows, praw = [], []
    for c in range(9):
        mc, ac = M[:, c], A[:, c]
        ok = ~(np.isnan(mc) | np.isnan(ac))
        mc, ac = mc[ok], ac[ok]
        d = mc - ac
        if np.allclose(d, 0):
            p = 1.0
        else:
            p = float(stats.wilcoxon(mc, ac).pvalue)
        praw.append(p)
        rows.append({"class": c, "cancer": NAME.get(c, str(c)),
                     "n_seeds": int(ok.sum()),
                     "mean_method_recall": round(float(mc.mean()), 4),
                     "mean_A_recall": round(float(ac.mean()), 4),
                     "mean_diff": round(float(d.mean()), 4),
                     "n_seeds_method_gt": f"{int((d > 0).sum())}/{len(d)}",
                     "wilcoxon_p": round(p, 5)})
    holm = _holm(praw)
    nsig = 0
    for r, hp in zip(rows, holm):
        r["holm_p"] = round(hp, 5)
        r["significant_help"] = bool(hp < 0.05 and r["mean_diff"] > 0)
        nsig += r["significant_help"]
    out = {
        "analysis": "pre-registered per-cancer RQ2 (H2), Holm-Bonferroni "
                    "across 9 classes",
        "config": CFG, "seeds": SEEDS, "n": len(SEEDS),
        "per_cancer": rows,
        "n_cancers_significant_positive": nsig,
        "interpretation": (
            f"{nsig}/9 cancers show a Holm-significant positive "
            "heterogeneity effect (method recall > homogeneous control). "
            "Per proposal §7.2 even one significant per-class gain is a "
            "defensible contribution; reported regardless of count. "
            "Aggregate macro RQ2 remains the headline (prereg_confirm.json); "
            "this resolves the pre-registered per-class H2."),
    }
    (RESULTS / "perclass_rq2.json").write_text(json.dumps(out, indent=2))
    print("\n=== per-cancer RQ2 ===")
    print(json.dumps(out, indent=2))
    print("\n##### P1 COMPLETE #####", flush=True)
