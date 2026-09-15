"""hetfm.run_seedext — extend the assignment-skew and alignment-baseline
analyses from 5 seeds to the 30 headline seeds (62-91).

Motivation: earlier revisions of the skew and ablation analyses
used 5 seeds while the main experiments used 30. Week-6 also ran on a
different seed set (42-46) from the headline analyses (62-91), so the paper
carried two seed sets as well as two sample sizes. This script puts every
scheme-level claim on the single headline seed set at n=30.

At n=5 the two-sided Wilcoxon floor is 0.0625, so week6 could not reach
p<0.05 by construction and used a paired-t-plus-all-same-sign gate instead.
At n=30 the Wilcoxon test is usable as the primary test, matching the
headline analyses (Section "Metrics and statistical analysis").

Reused, not recomputed (already n=30 on seeds 62-91, frozen in
prereg_confirm.json): balanced method-linear, the scheme-independent
A-control, and balanced tuned-FedProtoKD.

  cd <repository root>
  python -m hetfm.run_seedext --worker 0 --nworkers 6     # one shard
  python -m hetfm.run_seedext --summarize                 # aggregate
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                                       # noqa: E402
from hetfm.baselines_het import BASELINES                      # noqa: E402
from hetfm.fedprotokd import train as fpkd                     # noqa: E402
from hetfm.run_hetfm import probe_fm_dims, run                 # noqa: E402

RESULTS = Path(__file__).parent / "results"
CACHE = RESULTS / "seedext"
SEEDS = list(range(62, 92))
ROUNDS = 150
CFG = dict(depth="linear", k=256, tie="fm", lam_proto=1.0, lam_con=1.0)
SCHEMES = ["balanced", "stratified", "cancer_correlated", "adversarial"]
# balanced method / A-control / balanced fedprotokd-B are frozen at n=30
METHOD_SCHEMES = ["stratified", "cancer_correlated", "adversarial"]
FPKD_SCHEMES = ["cancer_correlated", "adversarial"]


def _route(scheme, seed):
    _, sp, gt, _ = assign.build_canonical_partition(seed=seed)
    clab = assign.client_labels_from_splits(sp)
    asn = assign.assign_fms(list(sp), scheme, seed, client_label=clab)
    rsp, rgt = assign.route(sp, gt, asn)
    return rsp, rgt, asn


def _tasks():
    """(kind, scheme, seed) work list, ordered so every scheme progresses
    together rather than finishing one scheme at a time."""
    out = []
    for seed in SEEDS:
        for scheme in METHOD_SCHEMES:
            out.append(("method", scheme, seed))
        for scheme in SCHEMES:
            out.append(("base", scheme, seed))
        for scheme in FPKD_SCHEMES:
            out.append(("fpkd", scheme, seed))
    return out


def _cache_path(kind, scheme, seed):
    return CACHE / f"{kind}_{scheme}_s{seed}.json"


def _do(kind, scheme, seed):
    """Run one task. run_hetfm.run writes its own file whose name omits
    depth, so ablated configs overwrite each other there; we keep our own
    depth-aware cache and never read that file."""
    if kind == "method":
        r = run(scheme=scheme, seed=seed, rounds=ROUNDS, log_every=ROUNDS,
                **CFG)
        return {"macro_acc": r["macro_acc"], "macro_f1": r.get("macro_f1")}
    if kind == "base":
        rsp, rgt, _ = _route(scheme, seed)
        out = {}
        for name, fn in BASELINES.items():
            rb = fn(rsp, rgt, rounds=ROUNDS, seed=seed, log_every=ROUNDS)
            out[name] = rb["macro_acc"]
        return out
    if kind == "fpkd":
        rsp, rgt, asn = _route(scheme, seed)
        fmd = probe_fm_dims(assign.FEAT_ROOT, assign.FMS)
        r = fpkd(rsp, rgt, asn, fmd, rounds=ROUNDS, seed=seed,
                 log_every=ROUNDS, aggregate_projector=True,
                 k=CFG["k"], depth=CFG["depth"])
        return {"macro_acc": r["macro_acc"]}
    raise ValueError(kind)


def work(worker, nworkers, reverse=False):
    """reverse=True walks the task list backwards, so a worker added after
    the first pass drains the tail while the original workers are still
    moving forward; the two meet only when the queue is nearly empty."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tasks = _tasks()
    mine = [t for i, t in enumerate(tasks) if i % nworkers == worker]
    if reverse:
        mine.reverse()
    print(f"[seedext w{worker}] {len(mine)}/{len(tasks)} tasks", flush=True)
    for n, (kind, scheme, seed) in enumerate(mine, 1):
        p = _cache_path(kind, scheme, seed)
        if p.exists():
            continue
        print(f"[seedext w{worker} {n}/{len(mine)}] {kind} {scheme} "
              f"seed {seed}", flush=True)
        # A CUDA OOM from a co-running worker must not kill the shard: the
        # baselines cache zero-padded 20480-d features and can transiently
        # need several GB. Retry once, then leave the task for a later pass.
        for attempt in (1, 2):
            try:
                res = _do(kind, scheme, seed)
                p.write_text(json.dumps(res, indent=2))
                break
            except Exception as e:                              # noqa: BLE001
                print(f"  ATTEMPT {attempt} FAILED {kind} {scheme} "
                      f"seed {seed}: {type(e).__name__}: {e}", flush=True)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:                               # noqa: BLE001
                    pass
                if attempt == 2:
                    print(f"  GIVING UP for now: {kind} {scheme} seed "
                          f"{seed}", flush=True)
                else:
                    time.sleep(30)
    print(f"[seedext w{worker}] done", flush=True)


# ---- summarize --------------------------------------------------------------
def _ms(xs):
    return {"mean": round(st.mean(xs), 4),
            "std": round(st.pstdev(xs), 4) if len(xs) > 1 else 0.0,
            "n": len(xs), "per_seed": [round(x, 4) for x in xs]}


def _boot_ci(d, iters=10000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = np.asarray(d, dtype=float)
    means = rng.choice(a, size=(iters, a.size), replace=True).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4)]


def _paired(a, b):
    """Wilcoxon signed-rank is primary at n=30, matching the headline
    analyses; paired t and a bootstrap CI of the mean difference are
    reported alongside."""
    d = [x - y for x, y in zip(a, b)]
    o = {"n": len(d), "mean_diff": round(st.mean(d), 4),
         "n_a_gt_b": f"{sum(1 for x in d if x > 0)}/{len(d)}",
         "boot_ci95": _boot_ci(d)}
    try:
        from scipy import stats
        o["wilcoxon_p"] = float(f"{stats.wilcoxon(a, b).pvalue:.3g}")
        o["paired_t_p"] = float(f"{stats.ttest_rel(a, b).pvalue:.3g}")
        o["significant"] = bool(o["wilcoxon_p"] < 0.05)
    except Exception as e:                                      # noqa: BLE001
        o["stat_error"] = str(e)
    return o


def _load(kind, scheme):
    """Per-seed values for one arm, or None if the shard is incomplete."""
    vals = []
    for seed in SEEDS:
        p = _cache_path(kind, scheme, seed)
        if not p.exists():
            return None
        vals.append(json.loads(p.read_text()))
    return vals


def summarize():
    pr = json.loads((RESULTS / "prereg_confirm.json").read_text())
    A_ref = pr["A_linear_homog_CONCH"]["per_seed"]          # scheme-indep
    method = {"balanced": pr["method_linear_balanced"]["per_seed"]}
    fpkd_b = {"balanced": pr["fedprotokd_B_balanced"]["per_seed"]}

    missing = []
    for scheme in METHOD_SCHEMES:
        v = _load("method", scheme)
        if v is None:
            missing.append(f"method/{scheme}")
        else:
            method[scheme] = [x["macro_acc"] for x in v]
    for scheme in FPKD_SCHEMES:
        v = _load("fpkd", scheme)
        if v is None:
            missing.append(f"fpkd/{scheme}")
        else:
            fpkd_b[scheme] = [x["macro_acc"] for x in v]
    base = {}
    for scheme in SCHEMES:
        v = _load("base", scheme)
        if v is None:
            missing.append(f"base/{scheme}")
        else:
            base[scheme] = {n: [x[n] for x in v] for n in BASELINES}

    out = {"seeds": SEEDS, "n": len(SEEDS), "rounds": ROUNDS, "config": CFG,
           "reused_frozen_n30": ["method/balanced", "A_control",
                                 "fpkd/balanced"],
           "incomplete": missing,
           "A_control_scheme_independent": _ms(A_ref),
           "schemes": {}}

    for scheme in SCHEMES:
        if scheme not in method or scheme not in base:
            continue
        e = {"method_linear": _ms(method[scheme]),
             "baselines": {n: _ms(v) for n, v in base[scheme].items()}}
        best = max(base[scheme], key=lambda n: st.mean(base[scheme][n]))
        e["best_baseline"] = best
        e["method_vs_best_baseline"] = _paired(method[scheme],
                                               base[scheme][best])
        e["method_vs_A_control"] = _paired(method[scheme], A_ref)
        if scheme in fpkd_b:
            e["fedprotokd_B_tuned"] = _ms(fpkd_b[scheme])
            e["method_vs_fedprotokd_B"] = _paired(method[scheme],
                                                  fpkd_b[scheme])
        out["schemes"][scheme] = e

    done = [s for s in SCHEMES if s in out["schemes"]]
    if len(done) == len(SCHEMES):
        mm = {s: out["schemes"][s]["method_linear"]["mean"] for s in SCHEMES}
        deg_m = mm["balanced"] - mm["adversarial"]
        bb = out["schemes"]["balanced"]["best_baseline"]
        deg_b = (out["schemes"]["balanced"]["baselines"][bb]["mean"]
                 - out["schemes"]["adversarial"]["baselines"][bb]["mean"])
        out["scheme_means_method"] = mm
        out["difficulty_order_easiest_first"] = sorted(
            mm, key=lambda s: -mm[s])
        out["H4_degradation"] = {
            "method_balanced_minus_adversarial": round(deg_m, 4),
            "best_baseline_same_contrast": round(deg_b, 4),
            "method_degrades_less": bool(deg_m < deg_b),
            "note": ("Negative degradation means the adversarial scheme is "
                     "EASIER than balanced for that arm.")}
    out["note"] = (
        "Every scheme-level comparison is on the 30 headline seeds (62-91), "
        "the same set as the recovery, complementarity and incrementality "
        "analyses. Wilcoxon signed-rank is primary; at the previous n=5 its "
        "two-sided floor was 0.0625 and could not reach 0.05.")

    p = RESULTS / "seedext_rq4_n30.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nsaved -> {p}")
    if missing:
        print(f"INCOMPLETE, still running: {missing}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        work(a.worker, a.nworkers, a.reverse)


if __name__ == "__main__":
    main()
