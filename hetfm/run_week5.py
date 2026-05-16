"""hetfm.run_week5 — FedProtoKD baseline, BOTH variants (user choice C).

A (paper-default, faithful): per-site projector+head, no aggregation,
   prototypes only. Under 107/107 single-class sites there is NO valid
   global classifier -> per-site-routed scores are artifacts. We RECORD
   the degeneracy (head-selfeval, proto-per-site) as a QUALITATIVE finding
   (not a leaderboard number). 3 seeds × {balanced, adversarial} is enough
   to show it is consistent.

B (best-effort tuned): relax ONLY the projector to per-FM-tied FedAvg
   (still no shared head, prototype-KD kept) -> nearest-global-prototype
   eval is non-degenerate. Full benchmark: {balanced, cancer_correlated,
   adversarial} × 5 seeds, linear/k256/150r. Compared (paired) vs the
   reused method-linear and the reused alignment baselines from
   week6_rq4.json — no recompute of those.

Honest n=5: paired-t + mean-diff + sign count; Wilcoxon two-sided floor at
n=5 is 0.0625 (recorded, NOT the gate); 'significant' = paired-t<0.05 AND
all same-sign. Method is expected to WIN B; reported plainly either way.
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                                       # noqa: E402
from hetfm.run_hetfm import probe_fm_dims                       # noqa: E402
from hetfm.fedprotokd import train as fpkd                      # noqa: E402

RESULTS = Path(__file__).parent / "results"
SEEDS = [42, 43, 44, 45, 46]
ROUNDS = 150
CFG = dict(k=256, depth="linear")
B_SCHEMES = ["balanced", "cancer_correlated", "adversarial"]


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


def _route(scheme, sd):
    _, sp, gt, _ = assign.build_canonical_partition(seed=sd)
    clab = assign.client_labels_from_splits(sp)
    asn = assign.assign_fms(list(sp), scheme, sd, client_label=clab)
    rsp, rgt = assign.route(sp, gt, asn)
    return rsp, rgt, asn


def variant_A():
    print("\n##### WEEK-5 A: faithful FedProtoKD degeneracy demo #####",
          flush=True)
    fmd = probe_fm_dims(assign.FEAT_ROOT, assign.FMS)
    rows = []
    for scheme in ["balanced", "adversarial"]:
        for sd in [42, 43, 44]:
            print(f"--- A {scheme} seed {sd} ---", flush=True)
            rsp, rgt, asn = _route(scheme, sd)
            r = fpkd(rsp, rgt, asn, fmd, rounds=ROUNDS, seed=sd,
                     log_every=ROUNDS, aggregate_projector=False, **CFG)
            rows.append({"scheme": scheme, "seed": sd, **r["degeneracy"]})
    out = {"variant": "A_faithful_paper_default",
           "finding": ("Faithful FedProtoKD (no aggregation) has NO valid "
                       "global classifier under class≈site coupling: every "
                       "per-site-routed score is an artifact of single-class "
                       "sites. This is the documented failure mode (risk "
                       "register) and an argument FOR our per-FM-group "
                       "FedAvg + shared-head design. NOT a leaderboard row."),
           "rows": rows}
    (RESULTS / "week5_fedprotokd_A_degeneracy.json").write_text(
        json.dumps(out, indent=2))
    print("saved -> week5_fedprotokd_A_degeneracy.json", flush=True)


def variant_B():
    print("\n##### WEEK-5 B: tuned FedProtoKD benchmark #####", flush=True)
    fmd = probe_fm_dims(assign.FEAT_ROOT, assign.FMS)
    w6 = json.loads((RESULTS / "week6_rq4.json").read_text())["schemes"]
    state = {"variant": "B_tuned_aggregate_projector", "config": CFG,
             "seeds": SEEDS, "rounds": ROUNDS, "schemes": {}}
    for scheme in B_SCHEMES:
        fb = []
        for sd in SEEDS:
            print(f"--- B {scheme} seed {sd} ---", flush=True)
            rsp, rgt, asn = _route(scheme, sd)
            r = fpkd(rsp, rgt, asn, fmd, rounds=ROUNDS, seed=sd,
                     log_every=ROUNDS, aggregate_projector=True, **CFG)
            fb.append(r["macro_acc"])
        method = w6[scheme]["method_linear"]["per_seed"]
        # 'balanced' is the reused entry and has no 'best_baseline' key;
        # derive it from the baseline means in that case.
        best_b = w6[scheme].get("best_baseline") or max(
            w6[scheme]["baselines"],
            key=lambda n: w6[scheme]["baselines"][n]["mean"])
        align = w6[scheme]["baselines"][best_b]["per_seed"]
        state["schemes"][scheme] = {
            "fedprotokd_B": _ms(fb),
            "method_linear_REUSED": _ms(method),
            "best_align_baseline_REUSED": {best_b: _ms(align)},
            "method_minus_fedprotokdB": _paired(method, fb),
            "fedprotokdB_minus_best_align": _paired(fb, align),
        }
        (RESULTS / "week5_fedprotokd_B.json").write_text(
            json.dumps(state, indent=2))
    # verdict
    md = {sc: state["schemes"][sc]["method_minus_fedprotokdB"]["mean_diff"]
          for sc in B_SCHEMES}
    state["verdict"] = {
        "method_minus_fedprotokdB_per_scheme": md,
        "method_beats_fedprotokdB_all_schemes": bool(
            all(v > 0 for v in md.values())),
        "interpretation": (
            "B = FedProtoKD given a FAIR non-degenerate eval (per-FM FedAvg "
            "projector + nearest global prototype). If method−B > 0 on every "
            "scheme with paired-t<0.05, our FedAvg-of-aligned-components + "
            "shared head beats prototype-KD-without-shared-head. n=5 caveat: "
            "read paired-t + sign; Wilcoxon floored. A (faithful) is the "
            "qualitative incompatibility finding, separate file."),
    }
    (RESULTS / "week5_fedprotokd_B.json").write_text(
        json.dumps(state, indent=2))
    print("saved -> week5_fedprotokd_B.json", flush=True)


def main():
    if "--bonly" not in sys.argv:        # A already saved on a prior run
        variant_A()
    variant_B()
    print("\n##### WEEK-5 COMPLETE #####", flush=True)


if __name__ == "__main__":
    main()
