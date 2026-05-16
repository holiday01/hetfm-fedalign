"""hetfm.run_prereg — executes EXACTLY hetfm/PREREGISTRATION.md (frozen
2026-05-16T04:21:54Z). No knobs beyond what the plan fixes. Seeds 62-91,
fresh. Reports every outcome regardless of direction.
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                                       # noqa: E402
from hetfm.run_hetfm import run, probe_fm_dims                 # noqa: E402
from hetfm.fedprotokd import train as fpkd                     # noqa: E402

RESULTS = Path(__file__).parent / "results"
SEEDS = list(range(62, 92))                 # 30 fresh, frozen
ROUNDS = 150
LB = 0.1283                                 # frozen (Week-1)
CFG = dict(depth="linear", k=256, tie="fm", lam_proto=1.0, lam_con=1.0)
BEST_FM = "Conch_v15"


def _ms(xs):
    return {"mean": round(st.mean(xs), 4), "std": round(st.pstdev(xs), 4),
            "n": len(xs), "per_seed": [round(x, 4) for x in xs]}


def _test(a, b):
    """Primary Wilcoxon two-sided + secondary paired-t + bootstrap CI."""
    d = [x - y for x, y in zip(a, b)]
    o = {"mean_diff": round(st.mean(d), 4),
         "n_a_gt_b": f"{sum(1 for x in d if x > 0)}/{len(d)}",
         "all_positive": bool(all(x > 0 for x in d))}
    try:
        import numpy as np
        from scipy import stats
        o["wilcoxon_p"] = round(float(stats.wilcoxon(a, b).pvalue), 5)
        o["paired_t_p"] = round(float(stats.ttest_rel(a, b).pvalue), 5)
        rng = np.random.default_rng(0)
        da = np.array(d)
        bs = [rng.choice(da, da.size, replace=True).mean()
              for _ in range(10000)]
        o["mean_diff_ci95"] = [round(float(np.percentile(bs, 2.5)), 4),
                               round(float(np.percentile(bs, 97.5)), 4)]
        o["reject_H0_wilcoxon_p<0.05_and_meanΔ>0"] = bool(
            o["wilcoxon_p"] < 0.05 and o["mean_diff"] > 0)
    except Exception as e:                                      # noqa: BLE001
        o["stat_error"] = str(e)
    return o


if __name__ == "__main__":
    print("##### PRE-REGISTERED CONFIRM (frozen 2026-05-16T04:21:54Z) "
          "seeds 62-91 #####", flush=True)
    fmd = probe_fm_dims(assign.FEAT_ROOT, assign.FMS)
    m, a, fb = [], [], []
    for sd in SEEDS:
        print(f"\n--- prereg seed {sd} ---", flush=True)
        rm = run(scheme="balanced", seed=sd, rounds=ROUNDS,
                 log_every=ROUNDS, **CFG)
        ra = run(scheme=f"homogeneous:{BEST_FM}", seed=sd, rounds=ROUNDS,
                 log_every=ROUNDS, **CFG)
        _, sp, gt, _ = assign.build_canonical_partition(seed=sd)
        clab = assign.client_labels_from_splits(sp)
        asn = assign.assign_fms(list(sp), "balanced", sd, client_label=clab)
        rsp, rgt = assign.route(sp, gt, asn)
        # fpkd() takes only k & depth (tie is derived from
        # aggregate_projector; it uses lam_kd not lam_con) — do NOT splat
        # the run()-shaped CFG into it.
        rb = fpkd(rsp, rgt, asn, fmd, rounds=ROUNDS, seed=sd,
                  log_every=ROUNDS, aggregate_projector=True,
                  k=CFG["k"], depth=CFG["depth"])
        m.append(rm["macro_acc"]); a.append(ra["macro_acc"])
        fb.append(rb["macro_acc"])
        clos = [(mi - LB) / max(ai - LB, 1e-6) for mi, ai in zip(m, a)]
        out = {
            "frozen": "2026-05-16T04:21:54Z", "seeds": SEEDS[:len(m)],
            "n": len(m), "config": CFG, "LB": LB, "partial": len(m) < 30,
            "method_linear_balanced": _ms(m),
            "A_linear_homog_CONCH": _ms(a),
            "fedprotokd_B_balanced": _ms(fb),
            "H1_RQ1_closure": {**_ms(clos), "prereg_bar": 0.90,
                               "H1_pass": bool(st.mean(clos) >= 0.90),
                               "saturated": bool(any(c > 1 for c in clos))},
            "H2_RQ2_method_vs_A": _test(m, a),
            "H3_incrementality_method_vs_fedprotokdB": _test(m, fb),
        }
        (RESULTS / "prereg_confirm.json").write_text(
            json.dumps(out, indent=2))
    print("\n=== PRE-REGISTERED RESULT (n=30) ===")
    print(json.dumps(out, indent=2))
    print("\n##### PREREG COMPLETE #####", flush=True)
