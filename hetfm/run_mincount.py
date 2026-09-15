"""hetfm.run_mincount — privacy/utility sensitivity to the per-class
minimum-count gate m.

Motivation: the protocol should not depend on an arbitrary choice of the
leakage risk of class means contributed by sites with few samples, and does
not justify a minimum sample threshold.

The protocol gates a site's class-mean contribution at m=8 slides. This
sweeps m over {1, 16} against that default (m=8 is already covered by the
headline runs) on the balanced scheme, over the same 30 headline seeds, so
the utility cost of a stricter gate — and the utility gained by removing it
— can be reported rather than asserted. m=1 is the no-gate worst case, in
which a site contributes a class mean computed from a single slide.

  cd <repository root>
  python -m hetfm.run_mincount --m 1 --worker 0 --nworkers 1
  python -m hetfm.run_mincount --summarize
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
CACHE = RESULTS / "mincount"
SEEDS = list(range(62, 92))
ROUNDS = 150
SCHEME = "balanced"
CFG = dict(k=256, depth="linear", tie="fm", lam_proto=1.0, lam_con=1.0)
M_VALUES = [1, 16]


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


def work(m, worker, nworkers, reverse=False):
    """reverse=True walks the seed list backwards so a worker added later
    drains the tail while the original workers move forward."""
    CACHE.mkdir(parents=True, exist_ok=True)
    mine = [s for i, s in enumerate(SEEDS) if i % nworkers == worker]
    if reverse:
        mine.reverse()
    for n, seed in enumerate(mine, 1):
        p = CACHE / f"m{m}_s{seed}.json"
        if p.exists():
            continue
        print(f"[mincount m={m} w{worker} {n}/{len(mine)}] seed {seed}",
              flush=True)
        p.write_text(json.dumps(_one(m, seed), indent=2))
    print(f"[mincount m={m} w{worker}] done", flush=True)


def _ms(xs):
    return {"mean": round(st.mean(xs), 4),
            "std": round(st.pstdev(xs), 4) if len(xs) > 1 else 0.0,
            "n": len(xs), "per_seed": [round(x, 4) for x in xs]}


def summarize():
    pr = json.loads((RESULTS / "prereg_confirm.json").read_text())
    base = pr["method_linear_balanced"]["per_seed"]      # m=8, n=30, frozen
    out = {"scheme": SCHEME, "seeds": SEEDS, "rounds": ROUNDS, "config": CFG,
           "default_gate": 8, "m8_reused_frozen": _ms(base), "sweep": {},
           "incomplete": []}
    for m in M_VALUES:
        vals = []
        for seed in SEEDS:
            p = CACHE / f"m{m}_s{seed}.json"
            if not p.exists():
                out["incomplete"].append(f"m{m}/s{seed}")
                continue
            vals.append(json.loads(p.read_text())["macro_acc"])
        if len(vals) != len(SEEDS):
            continue
        e = {"acc": _ms(vals)}
        d = [x - y for x, y in zip(vals, base)]
        e["vs_m8_mean_diff"] = round(st.mean(d), 4)
        e["n_gt_m8"] = f"{sum(1 for x in d if x > 0)}/{len(d)}"
        try:
            from scipy import stats
            e["wilcoxon_p"] = float(f"{stats.wilcoxon(vals, base).pvalue:.3g}")
            e["significant"] = bool(e["wilcoxon_p"] < 0.05)
        except Exception as err:                                # noqa: BLE001
            e["stat_error"] = str(err)
        out["sweep"][f"m{m}"] = e
    p = RESULTS / "mincount_sweep.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nsaved -> {p}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        work(a.m, a.worker, a.nworkers, a.reverse)


if __name__ == "__main__":
    main()
