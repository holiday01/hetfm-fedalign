"""hetfm.run_mincount_clean — single-backend re-run of the minimum-count
gate sweep.

The earlier sweep compared m=1 and m=16 arms run in July against an m=8 arm
reused frozen from the confirmatory run in May, and the July arms themselves
straddled two execution environments. A controlled check showed that backend
and thread count alone move macro accuracy by 0.03-0.04 on this task, which is
several times the effect the sweep reports, so the comparison was not sound.

This runs all three arms (m = 1, 8, 16) over the same thirty seeds in one
batch on one backend, and summarises them against each other rather than
against a frozen external baseline.

  cd <repository root>
  python -m hetfm.run_mincount_clean --worker 0 --nworkers 2
  python -m hetfm.run_mincount_clean --summarize
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                                       # noqa: E402
from hetfm.proto_anchor import train                           # noqa: E402
from hetfm.run_hetfm import probe_fm_dims                      # noqa: E402

RESULTS = Path(__file__).parent / "results"
CACHE = RESULTS / "mincount_clean"
SEEDS = list(range(62, 92))
ROUNDS = 150
SCHEME = "balanced"
CFG = dict(k=256, depth="linear", tie="fm", lam_proto=1.0, lam_con=1.0)
M_VALUES = [1, 8, 16]


def _one(m, seed):
    _, sp, gt, _ = assign.build_canonical_partition(seed=seed)
    clab = assign.client_labels_from_splits(sp)
    asn = assign.assign_fms(list(sp), SCHEME, seed, client_label=clab)
    rsp, rgt = assign.route(sp, gt, asn)
    fmd = probe_fm_dims(assign.FEAT_ROOT, assign.FMS)
    r = train(rsp, rgt, asn, fmd, rounds=ROUNDS, seed=seed,
              log_every=ROUNDS, min_count=m, **CFG)
    return {"macro_acc": r["macro_acc"], "macro_f1": r.get("macro_f1"),
            "min_count": m, "seed": seed}


def work(worker, nworkers):
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s) for m in M_VALUES for s in SEEDS]
    mine = [j for i, j in enumerate(jobs) if i % nworkers == worker]
    for n, (m, seed) in enumerate(mine, 1):
        p = CACHE / f"m{m}_s{seed}.json"
        if p.exists():
            continue
        print(f"[clean w{worker} {n}/{len(mine)}] m={m} seed={seed}",
              flush=True)
        p.write_text(json.dumps(_one(m, seed), indent=2))
    print(f"[clean w{worker}] done", flush=True)


def _ms(xs):
    return {"mean": round(st.mean(xs), 4),
            "std": round(st.pstdev(xs), 4) if len(xs) > 1 else 0.0,
            "n": len(xs), "per_seed": [round(x, 6) for x in xs]}


def summarize():
    arms, missing = {}, []
    for m in M_VALUES:
        vals = []
        for seed in SEEDS:
            p = CACHE / f"m{m}_s{seed}.json"
            if not p.exists():
                missing.append(f"m{m}/s{seed}")
                continue
            vals.append(json.loads(p.read_text())["macro_acc"])
        arms[m] = vals
    out = {"scheme": SCHEME, "seeds": SEEDS, "rounds": ROUNDS, "config": CFG,
           "default_gate": 8, "single_batch_single_backend": True,
           "arms": {f"m{m}": _ms(v) for m, v in arms.items() if v},
           "vs_m8": {}, "incomplete": missing}
    base = arms.get(8, [])
    if len(base) == len(SEEDS):
        for m in (1, 16):
            v = arms.get(m, [])
            if len(v) != len(SEEDS):
                continue
            d = [x - y for x, y in zip(v, base)]
            e = {"mean_diff": round(st.mean(d), 4),
                 "n_gt_m8": f"{sum(1 for x in d if x > 0)}/{len(d)}"}
            try:
                from scipy import stats
                e["wilcoxon_p"] = float(
                    f"{stats.wilcoxon(v, base).pvalue:.3g}")
                e["paired_t_p"] = float(
                    f"{stats.ttest_rel(v, base).pvalue:.3g}")
                e["significant"] = bool(e["wilcoxon_p"] < 0.05)
            except Exception as err:                            # noqa: BLE001
                e["stat_error"] = str(err)
            out["vs_m8"][f"m{m}"] = e
    p = RESULTS / "mincount_sweep_clean.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nsaved -> {p}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        work(a.worker, a.nworkers)


if __name__ == "__main__":
    main()
