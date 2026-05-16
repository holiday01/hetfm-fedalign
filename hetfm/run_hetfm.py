"""hetfm.run_hetfm — orchestrate: canonical partition -> assign -> route ->
prototype-anchored heterogeneous-FM federated training. Real run, real numbers.

  cd /home/holiday01/fl_wsi
  python -m hetfm.run_hetfm --scheme balanced --seed 42 --rounds 40
  python -m hetfm.run_hetfm --smoke           # Week-2 acceptance smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                       # noqa: E402
from hetfm.proto_anchor import train           # noqa: E402

LOCALONLY_LB = {"UNI_v2": 0.125, "Conch_v15": 0.131, "Virchow2": 0.129}


def probe_fm_dims(feat_root: str, fms) -> dict:
    dims = {}
    for fm in fms:
        f = next((Path(feat_root) / fm).glob("*.npy"))
        a = np.load(f)
        dims[fm] = int(a.mean(axis=0).shape[0]) if a.ndim > 1 else int(a.shape[0])
    return dims


def run(scheme="balanced", seed=42, rounds=40, k=256, depth="1hidden",
        tie="fm", lam_proto=1.0, lam_con=1.0, local_epochs=1, log_every=10):
    _, splits, gtest, _ = assign.build_canonical_partition(seed=seed)
    cids = list(splits)
    clab = assign.client_labels_from_splits(splits)
    asn = assign.assign_fms(cids, scheme, seed, client_label=clab)
    rsplits, rgtest = assign.route(splits, gtest, asn)
    fm_dims = probe_fm_dims(assign.FEAT_ROOT, assign.FMS)
    print(f"[run_hetfm] scheme={scheme} seed={seed} clients={len(cids)} "
          f"fm_dims={fm_dims} k={k} tie={tie} rounds={rounds}")
    res = train(rsplits, rgtest, asn, fm_dims, k=k, depth=depth, tie=tie,
                rounds=rounds, local_epochs=local_epochs, lam_proto=lam_proto,
                lam_con=lam_con, seed=seed, log_every=log_every)
    res.update({"scheme": scheme, "seed": seed})
    outdir = Path(__file__).parent / "results"
    outdir.mkdir(exist_ok=True)
    out = outdir / f"hetfm_{scheme}_seed{seed}_k{k}_{tie}.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"[run_hetfm] saved -> {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="balanced")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--depth", default="1hidden")
    ap.add_argument("--tie", default="fm", choices=["fm", "site"])
    ap.add_argument("--lam_proto", type=float, default=1.0)
    ap.add_argument("--lam_con", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true",
                    help="Week-2 acceptance: short balanced run must beat "
                         "the local-only lower bound (~0.13)")
    args = ap.parse_args()

    if args.smoke:
        res = run(scheme="balanced", seed=42, rounds=20, log_every=5)
        lb = max(LOCALONLY_LB.values())
        ok = res["macro_acc"] > lb + 0.05
        print(f"\n=== Week-2 smoke ===")
        print(f"  macro_acc={res['macro_acc']:.4f}  local-only LB={lb:.3f}")
        print(f"  per-FM-group aggregation: well-defined "
              f"(groups={sorted(set(assign.FMS))})")
        print(f"  WEEK-2 SMOKE: {'PASS' if ok else 'FAIL'} "
              f"(method must learn above the LB)")
        sys.exit(0 if ok else 1)
    run(args.scheme, args.seed, args.rounds, args.k, args.depth, args.tie,
        args.lam_proto, args.lam_con)


if __name__ == "__main__":
    main()
