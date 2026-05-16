"""hetfm.run_week4 — RQ3: is explicit prototype anchoring necessary?

Compares the method (prototype-anchored, balanced — reused from the Week-3
per-seed results, NOT recomputed) against the three alignment-necessity
controls (hetfm.baselines_het) on the SAME routed balanced splits, 5 seeds.

RQ3 verdict = paired test (method vs the BEST baseline). H3-style honest
framing: anchoring is "necessary" only if it beats the best alignment
control by a statistically reliable margin; otherwise the simpler control
suffices and that is reported plainly (no false-PASS headline — two prior
RQ1 false PASSes on this project).
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                                       # noqa: E402
from hetfm.baselines_het import BASELINES                      # noqa: E402

RESULTS = Path(__file__).parent / "results"
SEEDS = [42, 43, 44, 45, 46]
ROUNDS = 150


def _ms(xs):
    return {"mean": round(st.mean(xs), 4),
            "std": round(st.pstdev(xs), 4) if len(xs) > 1 else 0.0,
            "per_seed": [round(x, 4) for x in xs]}


def _paired(a, b):
    """method (a) vs baseline (b): mean diff + Wilcoxon signed-rank
    (primary per §6.5, n=5) + paired t fallback."""
    d = [x - y for x, y in zip(a, b)]
    md = st.mean(d)
    res = {"mean_diff": round(md, 4),
           "per_seed_diff": [round(x, 4) for x in d],
           "n_method_gt": f"{sum(1 for x in d if x > 0)}/{len(d)}"}
    try:
        from scipy import stats
        if any(x != 0 for x in d):
            w = stats.wilcoxon(a, b)
            res["wilcoxon_p"] = round(float(w.pvalue), 4)
        t = stats.ttest_rel(a, b)
        res["paired_t_p"] = round(float(t.pvalue), 4)
    except Exception as e:                                      # noqa: BLE001
        res["stat_error"] = str(e)
    return res


def _method_per_seed():
    """Headline method per-seed macro_acc from the locked Week-3 def-C
    summary (week3_rq1.json). NOT the per-seed hetfm_balanced_seed*_k256_fm
    files: run_hetfm's filename omits depth/lambda, so the Phase-2 ablation
    grid overwrites those with ablated configs — week3_rq1.json is the
    stable, headline-config source (seeds 42-46, k256/1hidden/λ=1)."""
    s = json.loads((RESULTS / "week3_rq1.json").read_text())
    return s["method_balanced_hetero"]["per_seed"]


def main():
    method = _method_per_seed()
    base_acc = {name: [] for name in BASELINES}
    for sd in SEEDS:
        print(f"\n########## RQ3 seed {sd} ##########", flush=True)
        _, sp, gt, _ = assign.build_canonical_partition(seed=sd)
        asn = assign.assign_fms(list(sp), "balanced", sd)
        rsp, rgt = assign.route(sp, gt, asn)
        for name, fn in BASELINES.items():
            r = fn(rsp, rgt, rounds=ROUNDS, seed=sd, log_every=ROUNDS)
            base_acc[name].append(r["macro_acc"])
            print(f"  [{name} seed{sd}] macro_acc={r['macro_acc']:.4f}")

    base_ms = {n: _ms(v) for n, v in base_acc.items()}
    best = max(base_ms, key=lambda n: base_ms[n]["mean"])
    cmp = {n: _paired(method, base_acc[n]) for n in BASELINES}
    method_ms = _ms(method)
    sig = cmp[best].get("wilcoxon_p", 1.0)
    necessary = bool(method_ms["mean"] > base_ms[best]["mean"]
                     and isinstance(sig, float) and sig < 0.05)
    summary = {
        "rounds": ROUNDS, "seeds": SEEDS, "scheme": "balanced",
        "method_balanced_hetero": method_ms,
        "baselines": base_ms,
        "best_baseline": best,
        "method_vs_each_baseline": cmp,
        "rq3_anchoring_necessary": necessary,
        "interpretation": (
            f"Best alignment control = '{best}' "
            f"({base_ms[best]['mean']:.4f}); method "
            f"{method_ms['mean']:.4f}. RQ3 'anchoring necessary' is True "
            "ONLY if method > best baseline with Wilcoxon p<0.05 across the "
            "5 pre-registered seeds; otherwise the simpler control suffices "
            "and that is the honest finding. n=5 -> treat p as indicative, "
            "confirm with the seed-extension before any headline."),
    }
    out = RESULTS / "week4_rq3.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== Week-4 RQ3 ===")
    print(json.dumps(summary, indent=2))
    print(f"saved -> {out}")
    return summary


if __name__ == "__main__":
    main()
