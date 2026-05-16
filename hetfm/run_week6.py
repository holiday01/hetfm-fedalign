"""hetfm.run_week6 — RQ4 robustness to FM-assignment skew (proposal §3,§6.1).

Headline config (linear, k256, λp=λc=1). Schemes (proposal §6.1):
  balanced  (reference; REUSED from week3_rq1_linear / week4_rq3_linear,
             not recomputed)
  stratified        (easiest control: every FM sees every class)
  cancer_correlated (skewed: each FM sees a skewed class subset)
  adversarial       (hardest: each FM sees ~3 disjoint classes; H4 stress)

A-control = same method, homogeneous CONCH — FM-HOMOGENEOUS, so it is
scheme-INDEPENDENT (one reference, reused from week3_rq1_linear, 42-46).
Baselines (zeropad/localpca/procrustes) ARE scheme-dependent -> run per
scheme per seed (balanced reused).

Hypotheses (pre-registered, §3):
  H4: method degradation balanced->adversarial < best-baseline degradation.
  H3: method - best non-anchored baseline > 0.05 under BOTH
      cancer_correlated AND adversarial.

Integrity: n=5 -> report paired-t + mean-diff + sign count; the Wilcoxon
n=5 two-sided floor is 0.0625 (cannot reach <0.05) so Wilcoxon is recorded
but NOT used as the significance gate; 'significant' needs paired-t<0.05
AND every seed same-sign. Nothing auto-celebrated.
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                                       # noqa: E402
from hetfm.run_hetfm import run                                # noqa: E402
from hetfm.baselines_het import BASELINES                      # noqa: E402

RESULTS = Path(__file__).parent / "results"
SEEDS = [42, 43, 44, 45, 46]
ROUNDS = 150
CFG = dict(depth="linear", k=256, tie="fm", lam_proto=1.0, lam_con=1.0)
NEW_SCHEMES = ["stratified", "cancer_correlated", "adversarial"]


def _ms(xs):
    return {"mean": round(st.mean(xs), 4),
            "std": round(st.pstdev(xs), 4) if len(xs) > 1 else 0.0,
            "n": len(xs), "per_seed": [round(x, 4) for x in xs]}


def _paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    o = {"mean_diff": round(st.mean(d), 4),
         "n_a_gt_b": f"{sum(1 for x in d if x > 0)}/{len(d)}",
         "per_seed_diff": [round(x, 4) for x in d]}
    try:
        from scipy import stats
        o["paired_t_p"] = round(float(stats.ttest_rel(a, b).pvalue), 4)
        if any(x != 0 for x in d):
            o["wilcoxon_p_floored"] = round(
                float(stats.wilcoxon(a, b).pvalue), 4)
        o["significant"] = bool(o["paired_t_p"] < 0.05
                                and (all(x > 0 for x in d)
                                     or all(x < 0 for x in d)))
    except Exception as e:                                      # noqa: BLE001
        o["stat_error"] = str(e)
    return o


def _run_scheme(scheme):
    """method-linear + 3 baselines for one scheme, 5 seeds (returns dicts
    of per-seed macro_acc)."""
    m, base = [], {n: [] for n in BASELINES}
    for sd in SEEDS:
        print(f"\n--- {scheme} seed {sd} : method-linear ---", flush=True)
        r = run(scheme=scheme, seed=sd, rounds=ROUNDS, log_every=ROUNDS,
                **CFG)
        m.append(r["macro_acc"])
        _, sp, gt, _ = assign.build_canonical_partition(seed=sd)
        clab = assign.client_labels_from_splits(sp)
        asn = assign.assign_fms(list(sp), scheme, sd, client_label=clab)
        rsp, rgt = assign.route(sp, gt, asn)
        for n, fn in BASELINES.items():
            rb = fn(rsp, rgt, rounds=ROUNDS, seed=sd, log_every=ROUNDS)
            base[n].append(rb["macro_acc"])
            print(f"  [{scheme}/{n} sd{sd}] acc={rb['macro_acc']:.4f}",
                  flush=True)
        _dump(partial=True)
    return m, base


_STATE = {}


def _dump(partial=False):
    out = {"config": CFG, "seeds": SEEDS, "rounds": ROUNDS,
           "schemes": _STATE, "partial": partial}
    (RESULTS / "week6_rq4.json").write_text(json.dumps(out, indent=2))


def main():
    # balanced reference (reuse — no recompute)
    lin = json.loads((RESULTS / "week3_rq1_linear.json").read_text())
    w4 = json.loads((RESULTS / "week4_rq3_linear.json").read_text())
    m_bal = lin["method_linear_balanced"]["per_seed"][:5]
    A_ref = lin["A_control_linear_homog_CONCH"]["per_seed"][:5]   # scheme-indep
    base_bal = {n: v["per_seed"]
                for n, v in w4["baselines_REUSED_from_phase3"].items()}
    _STATE["balanced"] = {
        "method_linear": _ms(m_bal),
        "A_ref_scheme_independent": _ms(A_ref),
        "baselines": {n: _ms(v) for n, v in base_bal.items()},
        "reused": True}
    _dump(partial=True)

    for scheme in NEW_SCHEMES:
        print(f"\n##### SCHEME: {scheme} #####", flush=True)
        m, base = _run_scheme(scheme)
        best = max(base, key=lambda n: st.mean(base[n]))
        _STATE[scheme] = {
            "method_linear": _ms(m),
            "baselines": {n: _ms(v) for n, v in base.items()},
            "best_baseline": best,
            "method_vs_best_baseline": _paired(m, base[best]),
            "method_vs_A_ref": _paired(m, A_ref),
        }
        _dump(partial=True)

    # ---- RQ4 / H3 / H4 verdicts ----
    mb = _STATE["balanced"]["method_linear"]["mean"]
    adv = _STATE["adversarial"]["method_linear"]["mean"]
    best_adv = _STATE["adversarial"]["best_baseline"]
    base_bal_best_mean = st.mean(base_bal[best_adv])
    base_adv_mean = _STATE["adversarial"]["baselines"][best_adv]["mean"]
    method_drop = mb - adv
    base_drop = base_bal_best_mean - base_adv_mean
    h3 = {sc: round(_STATE[sc]["method_linear"]["mean"]
                    - _STATE[sc]["baselines"][_STATE[sc]["best_baseline"]]
                    ["mean"], 4)
          for sc in ("cancer_correlated", "adversarial")}
    verdict = {
        "rq4_degradation_balanced_to_adversarial": {
            "method_linear": round(method_drop, 4),
            f"best_baseline({best_adv})": round(base_drop, 4),
            "H4_method_more_robust": bool(method_drop < base_drop),
        },
        "H3_method_minus_best_baseline_gt_0.05": {
            "cancer_correlated": h3["cancer_correlated"],
            "adversarial": h3["adversarial"],
            "H3_holds_both": bool(h3["cancer_correlated"] > 0.05
                                  and h3["adversarial"] > 0.05),
        },
        "scheme_means_method_linear": {
            sc: _STATE[sc]["method_linear"]["mean"]
            for sc in _STATE},
        "interpretation": (
            "RQ4: compare method_linear across schemes (stratified easiest "
            "-> adversarial hardest). H4 = method degrades LESS than the "
            "best dimension-matching baseline from balanced->adversarial. "
            "H3 = method beats best non-anchored baseline by >0.05 under "
            "BOTH skewed schemes. n=5: read paired_t + mean_diff + sign "
            "count; Wilcoxon floored at 0.0625 (recorded, not the gate). "
            "A_ref is FM-homogeneous hence scheme-independent (one ref)."),
    }
    _STATE["_verdict"] = verdict
    _dump(partial=False)
    print("\n=== Week-6 RQ4 verdict ===")
    print(json.dumps(verdict, indent=2))
    print("\n##### WEEK-6 COMPLETE #####", flush=True)


if __name__ == "__main__":
    main()
