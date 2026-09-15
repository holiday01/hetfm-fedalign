"""hetfm.r2_backend — controlled backend study.

Same code, same seed, same data; only the execution backend changes:
  gpu           : from the R2 batch (results/r2_batch/method_linear_balanced_s*.json)
  cpu_t24       : CPU, torch.set_num_threads(24) (the machine's default)
  cpu_t2        : CPU, torch.set_num_threads(2)
Seeds 62-66 (five), headline linear configuration, balanced scheme, 150 rounds.
Output: results/r2_batch/backend/{arm}_s{seed}.json

  python -m hetfm.r2_backend
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                                       # noqa: E402
from hetfm.r2_trainer import train_r2                          # noqa: E402
from hetfm.run_hetfm import probe_fm_dims                      # noqa: E402

OUT = Path(__file__).parent / "results" / "r2_batch" / "backend"
SEEDS = [62, 63, 64, 65, 66]
ARMS = {"cpu_t24": 24, "cpu_t2": 2}
HEAD = dict(k=256, depth="linear", tie="fm", lam_proto=1.0, lam_con=1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fmd = probe_fm_dims(assign.FEAT_ROOT, assign.FMS)
    for arm, nt in ARMS.items():
        torch.set_num_threads(nt)
        for sd in SEEDS:
            p = OUT / f"{arm}_s{sd}.json"
            if p.exists():
                continue
            print(f"[backend] {arm} seed {sd} threads={torch.get_num_threads()}",
                  flush=True)
            _, sp, gt, _ = assign.build_canonical_partition(seed=sd)
            clab = assign.client_labels_from_splits(sp)
            asn = assign.assign_fms(list(sp), "balanced", sd, client_label=clab)
            rsp, rgt = assign.route(sp, gt, asn)
            t0 = time.time()
            r = train_r2(rsp, rgt, asn, fmd, rounds=150, seed=sd, device="cpu",
                         log_every=150, run_diagnostics=False, **HEAD)
            r["_task"] = {"arm": arm, "threads": torch.get_num_threads(),
                          "seed": sd, "wall_s": round(time.time() - t0, 1)}
            p.write_text(json.dumps(r, indent=1, default=str))
            print(f"  -> macro_acc={r['macro_acc']:.6f} wall={r['_task']['wall_s']}s",
                  flush=True)


if __name__ == "__main__":
    main()
