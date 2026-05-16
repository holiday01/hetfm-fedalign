"""hetfm.run_linear_rebaseline — re-baseline at the NEW headline config.

User decision 2026-05-16: headline method = LINEAR projector, BOTH anchor
terms kept (lam_proto=1, lam_con=1), k=256, tie=fm. The ablation showed
linear ≫ 1hidden (+0.094); lam_proto is ~inert but the user keeps it for
robustness (stated honestly in the paper, not hidden).

All prior headline numbers were at the suboptimal 1hidden config. This
recomputes, at linear, n=20:
  method-linear : balanced, depth=linear  (seeds 42-46 REUSED from
                  ablation_grid.json['depth=linear']; 47-61 run here)
  A-linear      : homogeneous:Conch_v15, depth=linear (RQ1 control, all 20)
  B             : plain FedAvg, projector-independent -> REUSED from
                  week3_rq1_ext.json (config-independent, not recomputed)
Then RQ3-linear: reuse Phase-3 baseline per-seed from week4_rq3.json
(alignment operators are projector-independent), swap the method point to
method-linear (seeds 42-46), recompute the paired verdict.

Integrity unchanged: closure>1 = saturated (say 'recovers/matches');
rq2/rq3 'significant' requires p<0.05 AND every seed positive; n=20 here is
a re-baseline at the chosen config, the n=5 forking-paths caveat still
applies to any RQ2 *headline* claim (note carried in the JSON).
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                                       # noqa: E402
from hetfm.run_hetfm import run                                # noqa: E402

RESULTS = Path(__file__).parent / "results"
BEST_FM = "Conch_v15"
ROUNDS = 150
BASE_SEEDS = [42, 43, 44, 45, 46]
EXT_SEEDS = list(range(47, 62))
ALL_SEEDS = BASE_SEEDS + EXT_SEEDS
LB = (0.125 + 0.131 + 0.129) / 3
CFG = dict(depth="linear", k=256, tie="fm", lam_proto=1.0, lam_con=1.0)


def _ms(xs):
    return {"mean": round(st.mean(xs), 4),
            "std": round(st.pstdev(xs), 4) if len(xs) > 1 else 0.0,
            "n": len(xs), "per_seed": [round(x, 4) for x in xs]}


def _stats(a, b):
    d = [x - y for x, y in zip(a, b)]
    out = {"mean_diff": round(st.mean(d), 4),
           "n_a_gt_b": f"{sum(1 for x in d if x > 0)}/{len(d)}",
           "per_seed_diff": [round(x, 4) for x in d]}
    try:
        from scipy import stats
        if any(x != 0 for x in d):
            out["wilcoxon_p"] = round(float(stats.wilcoxon(a, b).pvalue), 4)
        out["paired_t_p"] = round(float(stats.ttest_rel(a, b).pvalue), 4)
        out["significant"] = bool(out.get("wilcoxon_p", 1.0) < 0.05
                                  and all(x > 0 for x in d))
    except Exception as e:                                      # noqa: BLE001
        out["stat_error"] = str(e)
    return out


def _method_linear():
    g = json.loads((RESULTS / "ablation_grid.json").read_text())
    reused = list(g["depth=linear"]["macro_acc"]["per_seed"])   # 42-46
    print(f"[reuse] method-linear seeds 42-46 from grid: {reused}",
          flush=True)
    for sd in EXT_SEEDS:
        print(f"\n--- method-linear seed {sd} ---", flush=True)
        r = run(scheme="balanced", seed=sd, rounds=ROUNDS,
                log_every=ROUNDS, **CFG)
        reused.append(r["macro_acc"])
        _write(reused, None, partial=True)
    return reused


def _A_linear():
    out = []
    for sd in ALL_SEEDS:
        print(f"\n--- A-linear seed {sd} ---", flush=True)
        r = run(scheme=f"homogeneous:{BEST_FM}", seed=sd, rounds=ROUNDS,
                log_every=ROUNDS, **CFG)
        out.append(r["macro_acc"])
    return out


def _write(m, a, partial=False):
    n = len(m)
    obj = {"config": CFG, "headline": True,
           "note": ("Re-baseline at the user-chosen headline config (linear "
                    "projector, both anchor terms). n=20. The forking-paths "
                    "caveat for an RQ2 *headline* p-value still applies — "
                    "this confirms at the chosen config, it is not a fresh "
                    "pre-registration."),
           "seeds": ALL_SEEDS[:n], "n": n, "rounds": ROUNDS,
           "method_linear_balanced": _ms(m)}
    if a is not None:
        a = a[:n]
        closures = [(mi - LB) / max(ai - LB, 1e-6)
                    for mi, ai in zip(m, a)]
        obj["A_control_linear_homog_CONCH"] = _ms(a)
        b = json.loads((RESULTS / "week3_rq1_ext.json").read_text())
        obj["B_reference_plain_fedavg_REUSED"] = \
            b["B_reference_plain_fedavg_homog_CONCH"]
        obj["local_only_LB_mean"] = round(LB, 4)
        obj["rq1_gap_closure_vs_A"] = _ms(closures)
        obj["rq1_recovery_pass"] = bool(st.mean(closures) >= 0.90)
        obj["rq1_metric_saturated"] = bool(
            sum(1 for c in closures if c > 1.0) > 0)
        obj["rq2_method_vs_A"] = _stats(m, a)
    (RESULTS / "week3_rq1_linear.json").write_text(json.dumps(obj, indent=2))
    if not partial:
        print("\n=== RQ1/RQ2 @ linear (n=%d) ===" % n, flush=True)
        print(json.dumps(obj, indent=2), flush=True)


def _rq3_linear(method_lin):
    p = RESULTS / "week4_rq3.json"
    if not p.exists():
        print("[rq3-linear] week4_rq3.json not present yet — skipped; "
              "rerun this script after Phase 3 finishes.", flush=True)
        return
    w4 = json.loads(p.read_text())
    m5 = method_lin[:5]                                          # seeds 42-46
    base_ps = {n: v["per_seed"] for n, v in w4["baselines"].items()}
    cmp = {n: _stats(m5, ps) for n, ps in base_ps.items()}
    best = max(w4["baselines"], key=lambda n: w4["baselines"][n]["mean"])
    out = {"config": CFG, "seeds": BASE_SEEDS, "n": 5,
           "method_linear": _ms(m5),
           "baselines_REUSED_from_phase3": w4["baselines"],
           "best_baseline": best,
           "method_linear_vs_each_baseline": cmp,
           "rq3_anchoring_necessary": bool(
               _ms(m5)["mean"] > w4["baselines"][best]["mean"]
               and isinstance(cmp[best].get("wilcoxon_p", 1.0), float)
               and cmp[best].get("wilcoxon_p", 1.0) < 0.05),
           "note": ("Alignment-operator baselines are projector-independent "
                    "-> reused from Phase-3 week4_rq3.json; only the method "
                    "point is the linear-config re-baseline. n=5 -> "
                    "indicative; confirm with seed-extension.")}
    (RESULTS / "week4_rq3_linear.json").write_text(json.dumps(out, indent=2))
    print("\n=== RQ3 @ linear (n=5) ===", flush=True)
    print(json.dumps(out, indent=2), flush=True)


def main():
    print("##### LINEAR RE-BASELINE (headline config) #####", flush=True)
    m = _method_linear()
    a = _A_linear()
    _write(m, a, partial=False)
    _rq3_linear(m)
    print("\n##### LINEAR RE-BASELINE COMPLETE #####", flush=True)


if __name__ == "__main__":
    main()
