"""hetfm.run_autonomous_batch — one detached driver for the autonomous batch.

Runs sequentially (single GPU), each phase writing JSON incrementally so a
crash never loses completed work:

  Phase 1  RQ1/RQ2 seed-extension to n=20 (labeled robustness SENSITIVITY;
           the pre-registered n=5 Wilcoxon stays the headline). Reuses the
           on-disk Week-3 per-seed results for seeds 42-46; runs 47-61.
  Phase 2  Week-7 OFAT ablation grid (hetfm.run_grid).
  Phase 3  Week-4 RQ3 alignment-necessity baselines (hetfm.run_week4).

A phase failing is logged and the batch continues with the next.
"""
import json
import statistics as st
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                                       # noqa: E402
from hetfm.run_hetfm import run, probe_fm_dims                 # noqa: E402
from hetfm.proto_anchor import homogeneous_fedavg              # noqa: E402

RESULTS = Path(__file__).parent / "results"
BEST_FM = "Conch_v15"
ROUNDS = 150
BASE_SEEDS = [42, 43, 44, 45, 46]          # pre-registered (Week-3)
EXT_SEEDS = list(range(47, 62))            # +15 -> n=20 sensitivity
LB = (0.125 + 0.131 + 0.129) / 3


def _ms(xs):
    return {"mean": round(st.mean(xs), 4),
            "std": round(st.pstdev(xs), 4) if len(xs) > 1 else 0.0,
            "n": len(xs), "per_seed": [round(x, 4) for x in xs]}


def _on_disk(scheme, sd):
    f = RESULTS / f"hetfm_{scheme}_seed{sd}_k256_fm.json"
    return json.loads(f.read_text())["macro_acc"] if f.exists() else None


def phase1_seed_extension():
    print("\n##### PHASE 1: RQ1/RQ2 seed-extension (n=20 sensitivity) #####",
          flush=True)
    prior = json.loads((RESULTS / "week3_rq1.json").read_text())
    b_base = prior["B_reference_plain_fedavg_homog_CONCH"]["per_seed"]
    m_acc, a_acc, b_acc = [], [], []
    for i, sd in enumerate(BASE_SEEDS):                 # reuse on disk
        m_acc.append(_on_disk("balanced", sd))
        a_acc.append(_on_disk(f"homogeneous:{BEST_FM}", sd))
        b_acc.append(b_base[i])
    for sd in EXT_SEEDS:                                # compute new
        print(f"\n--- ext seed {sd} ---", flush=True)
        m = run(scheme="balanced", seed=sd, rounds=ROUNDS, k=256, tie="fm",
                log_every=ROUNDS)
        a = run(scheme=f"homogeneous:{BEST_FM}", seed=sd, rounds=ROUNDS,
                k=256, tie="fm", log_every=ROUNDS)
        _, sp, gt, _ = assign.build_canonical_partition(seed=sd)
        asn = assign.assign_fms(list(sp), f"homogeneous:{BEST_FM}", sd)
        rsp, rgt = assign.route(sp, gt, asn)
        d_in = probe_fm_dims(assign.FEAT_ROOT, [BEST_FM])[BEST_FM]
        b = homogeneous_fedavg(rsp, rgt, d_in, rounds=ROUNDS, seed=sd,
                               log_every=ROUNDS)
        m_acc.append(m["macro_acc"]); a_acc.append(a["macro_acc"])
        b_acc.append(b["macro_acc"])
        # incremental write every seed
        _write_ext(m_acc, a_acc, b_acc)
    _write_ext(m_acc, a_acc, b_acc, final=True)


def _write_ext(m_acc, a_acc, b_acc, final=False):
    n = len(m_acc)
    seeds = (BASE_SEEDS + EXT_SEEDS)[:n]
    closures = [(m - LB) / max(a - LB, 1e-6) for m, a in zip(m_acc, a_acc)]
    d = [m - a for m, a in zip(m_acc, a_acc)]
    out = {
        "note": ("ROBUSTNESS SENSITIVITY ONLY. Pre-registered headline is "
                 "the n=5 (seeds 42-46) Wilcoxon in week3_rq1.json; this "
                 "extends to n=20 to see if the RQ2 trend (p=0.087 @ n=5) "
                 "firms up. Not a substitute for pre-registration."),
        "seeds": seeds, "n": n, "rounds": ROUNDS,
        "method_balanced_hetero": _ms(m_acc),
        "A_control_same_method_homog_CONCH": _ms(a_acc),
        "B_reference_plain_fedavg_homog_CONCH": _ms(b_acc),
        "local_only_LB_mean": round(LB, 4),
        "rq1_gap_closure_vs_A": _ms(closures),
        "rq2_mean_diff_method_minus_A": round(st.mean(d), 4),
        "rq2_n_seeds_method_gt_A": f"{sum(1 for x in d if x > 0)}/{n}",
    }
    if n >= 2:
        try:
            from scipy import stats
            if any(x != 0 for x in d):
                out["rq2_wilcoxon_p"] = round(
                    float(stats.wilcoxon(m_acc, a_acc).pvalue), 4)
            out["rq2_paired_t_p"] = round(
                float(stats.ttest_rel(m_acc, a_acc).pvalue), 4)
            out["rq2_significant_help"] = bool(
                out.get("rq2_wilcoxon_p", 1.0) < 0.05
                and all(x > 0 for x in d))
        except Exception as e:                              # noqa: BLE001
            out["stat_error"] = str(e)
    (RESULTS / "week3_rq1_ext.json").write_text(json.dumps(out, indent=2))
    if final:
        print("\n=== Phase 1 result (n=%d) ===" % n)
        print(json.dumps(out, indent=2), flush=True)


def main():
    for name, fn in [("phase1_seed_extension", phase1_seed_extension)]:
        try:
            fn()
        except Exception:                                   # noqa: BLE001
            print(f"!!! {name} FAILED:\n{traceback.format_exc()}", flush=True)

    try:
        from hetfm.run_grid import main as grid_main
        print("\n##### PHASE 2: Week-7 ablation grid #####", flush=True)
        grid_main()
    except Exception:                                       # noqa: BLE001
        print(f"!!! phase2 grid FAILED:\n{traceback.format_exc()}", flush=True)

    try:
        from hetfm.run_week4 import main as week4_main
        print("\n##### PHASE 3: Week-4 RQ3 baselines #####", flush=True)
        week4_main()
    except Exception:                                       # noqa: BLE001
        print(f"!!! phase3 week4 FAILED:\n{traceback.format_exc()}",
              flush=True)

    print("\n##### AUTONOMOUS BATCH COMPLETE #####", flush=True)


if __name__ == "__main__":
    main()
