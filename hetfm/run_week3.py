"""hetfm.run_week3 — RQ1 (definition C, locked 2026-05-16), 5 seeds, cached.

Per seed:
  method = balanced heterogeneous, prototype-anchored          (proto_anchor.train)
  A      = SAME method, homogeneous best-FM (CONCH)             RQ1 CONTROL
           -> isolates the effect of FM heterogeneity alone
  B      = plain standard FedAvg of an MLP head, homogeneous    EXTERNAL REFERENCE
           best-FM (CONCH)                                       (absolute standing only)
LB       = real Week-1 local-only mean (mixed federation floor).

RQ1 (pre-registered, locked #2): gap-closure vs the CONTROL A,
  closure = (method - LB) / (A - LB),  H1 bar: closure >= 0.90.

Integrity note (two prior attempts produced false PASSes):
  * closure > 1 means the method EXCEEDS the control -> the ratio
    SATURATES; report it as "recovers/matches" not "closes X%".
  * "heterogeneity helps" (method > A) is an RQ2 complementarity
    SIGNAL only if a proper PAIRED test is significant AND positive
    on every seed. mean(method) > mean(A) alone is NOT a finding and
    is never headlined here. The recorded interpretation stays
    conservative so an unattended re-run cannot emit a false headline.

Usage:
  python -m hetfm.run_week3                 # full run (5 seeds x 150 rds)
  python -m hetfm.run_week3 --summarize     # rebuild JSON from on-disk
                                            # per-seed results (no recompute)
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                                    # noqa: E402
from hetfm.run_hetfm import run, probe_fm_dims              # noqa: E402
from hetfm.proto_anchor import homogeneous_fedavg           # noqa: E402

ROUNDS = 150
SEEDS = [42, 43, 44, 45, 46]                # locked: 5 seeds
BEST_FM = "Conch_v15"
LB = {"UNI_v2": 0.125, "Conch_v15": 0.131, "Virchow2": 0.129}   # Week-1 real
RESULTS = Path(__file__).parent / "results"


def _ms(xs):
    return {"mean": round(st.mean(xs), 4),
            "std": round(st.pstdev(xs), 4) if len(xs) > 1 else 0.0,
            "per_seed": [round(x, 4) for x in xs]}


def _paired(method, control):
    """Honest paired comparison method vs control A. Returns the diff
    distribution + a two-tailed paired t-test (scipy if available, else
    NaN p). 'significant_help' requires p<0.05 AND every seed positive —
    a single seed where the method loses kills the complementarity claim."""
    d = [m - a for m, a in zip(method, control)]
    n = len(d)
    md, sd = st.mean(d), (st.stdev(d) if n > 1 else 0.0)
    t = md / (sd / n ** 0.5) if sd > 0 else float("inf")
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    except Exception:
        p = float("nan")
    n_pos = sum(1 for x in d if x > 0)
    sig = (p == p) and p < 0.05 and n_pos == n     # p==p: not NaN
    return {
        "per_seed_diff": [round(x, 4) for x in d],
        "mean_diff": round(md, 4), "sd_diff": round(sd, 4),
        "paired_t": round(t, 3), "paired_p_two_tailed": round(p, 4),
        "n_seeds_method_gt_control": f"{n_pos}/{n}",
        "significant_help": bool(sig),
    }


def _load_per_seed(scheme):
    """Read macro_acc per seed from the on-disk run_hetfm JSONs."""
    out = []
    for sd in SEEDS:
        f = RESULTS / f"hetfm_{scheme}_seed{sd}_k256_fm.json"
        out.append(json.loads(f.read_text())["macro_acc"])
    return out


def _summarize(m_acc, a_acc, b_acc):
    lb = sum(LB.values()) / len(LB)
    closures = [(m - lb) / max(a - lb, 1e-6) for m, a in zip(m_acc, a_acc)]
    mm, aa = st.mean(m_acc), st.mean(a_acc)
    rq2 = _paired(m_acc, a_acc)
    recovery_pass = bool(st.mean(closures) >= 0.90)
    saturated = bool(sum(1 for c in closures if c > 1.0) > 0)
    return {
        "rounds": ROUNDS, "seeds": SEEDS, "rq1_upper_bound_def": "C",
        "method_balanced_hetero": _ms(m_acc),
        "A_control_same_method_homog_CONCH": _ms(a_acc),
        "B_reference_plain_fedavg_homog_CONCH": _ms(b_acc),
        "local_only_LB_mean": round(lb, 4),
        "rq1_gap_closure_vs_A": _ms(closures),
        "rq1_prereg_bar": 0.90,
        "rq1_recovery_pass": recovery_pass,
        "rq1_metric_saturated": saturated,
        "rq2_complementarity": rq2,
        "interpretation": (
            "RQ1: closure>=0.90 vs the CORRECT control A => the recovery "
            "hypothesis (H1) holds; closure>1 here means the method "
            "MATCHES/SLIGHTLY-EXCEEDS the same-method homogeneous control "
            "(ratio saturates -> report as 'recovers/matches', not 'closes "
            "N%'). RQ2 (heterogeneity helps): a directional trend ONLY -- "
            f"mean_diff={rq2['mean_diff']}, paired p="
            f"{rq2['paired_p_two_tailed']}, "
            f"{rq2['n_seeds_method_gt_control']} seeds positive; "
            f"significant_help={rq2['significant_help']}. NOT a headline "
            "claim at n=5; needs more seeds + the Week-6/7 confirmation. "
            "B is plain-FedAvg floor context only; FedFM-WSI's tuned ~0.64 "
            "is the external ceiling, not B."),
    }


def _run_all():
    m_acc, a_acc, b_acc = [], [], []
    for sd in SEEDS:
        print(f"\n########## seed {sd} ##########")
        m = run(scheme="balanced", seed=sd, rounds=ROUNDS, k=256,
                tie="fm", log_every=50)
        a = run(scheme=f"homogeneous:{BEST_FM}", seed=sd, rounds=ROUNDS,
                k=256, tie="fm", log_every=50)
        _, sp, gt, _ = assign.build_canonical_partition(seed=sd)
        asn = assign.assign_fms(list(sp), f"homogeneous:{BEST_FM}", sd)
        rsp, rgt = assign.route(sp, gt, asn)
        d_in = probe_fm_dims(assign.FEAT_ROOT, [BEST_FM])[BEST_FM]
        b = homogeneous_fedavg(rsp, rgt, d_in, rounds=ROUNDS, seed=sd,
                               log_every=50)
        m_acc.append(m["macro_acc"]); a_acc.append(a["macro_acc"])
        b_acc.append(b["macro_acc"])
    return m_acc, a_acc, b_acc


if __name__ == "__main__":
    out = RESULTS / "week3_rq1.json"
    if "--summarize" in sys.argv:
        # Rebuild the summary from on-disk per-seed results — no recompute.
        # method/A come from run_hetfm JSONs; B (homogeneous_fedavg writes
        # no per-seed file) is reused from the prior real run recorded in
        # week3_rq1.json (same run; values also in week3C_run.log).
        m_acc = _load_per_seed("balanced")
        a_acc = _load_per_seed(f"homogeneous:{BEST_FM}")
        prior = json.loads(out.read_text())
        b_acc = prior["B_reference_plain_fedavg_homog_CONCH"]["per_seed"]
        print(f"[summarize] reusing B per_seed from {out.name}: {b_acc}")
    else:
        m_acc, a_acc, b_acc = _run_all()

    summary = _summarize(m_acc, a_acc, b_acc)
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== Week-3 RQ1 (definition C, 5 seeds, honest stats) ===")
    print(json.dumps(summary, indent=2))
    print(f"saved -> {out}")
