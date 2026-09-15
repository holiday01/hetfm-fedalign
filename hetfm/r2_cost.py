"""hetfm.r2_cost — dedicated single-process timing and memory pass for the
computational-cost table.  Run with NOTHING else on the
GPU.  One seed (62), 150 rounds, balanced scheme, each arm once; records
wall-clock per round and in total, peak GPU memory, trainable parameters, and
the GPU name.  Output: results/r2_cost.json.

  python -m hetfm.r2_cost
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import run_r2_batch as B                            # noqa: E402
from hetfm.baselines_het import BASELINES                      # noqa: E402

OUT = Path(__file__).parent / "results" / "r2_cost.json"
SEED, ROUNDS = 62, 150


def _measure(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    r = fn()
    torch.cuda.synchronize()
    tot = time.time() - t0
    return r, tot, torch.cuda.max_memory_allocated() / 2**20


def main():
    R = B.registry()
    out = {"seed": SEED, "rounds": ROUNDS, "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "arms": {}}
    arms = ["method_linear_balanced", "method_1hidden_balanced", "abl_ce_only",
            "plain_fedavg_Conch_v15", "fpkd_B_balanced", "fedgh_tied_balanced",
            "fedgh_faithful_balanced"]
    for name in arms:
        print(f"[cost] {name}", flush=True)
        r, tot, mem = _measure(lambda: R[name][1](SEED, ROUNDS))
        train_s = r.get("timing", {}).get("train_s", tot)
        out["arms"][name] = {"total_s": round(tot, 1), "total_min": round(tot / 60, 2),
                             "train_s": round(train_s, 1),
                             "per_round_s": round(train_s / ROUNDS, 3),
                             "peak_gpu_mem_mb": round(mem, 1),
                             "n_params": r.get("n_params"), "macro_acc": r.get("macro_acc")}
        OUT.write_text(json.dumps(out, indent=1))
    # dimension-matching baselines, each separately
    rsp, rgt, _ = B._route("balanced", SEED)
    for bname, fn in BASELINES.items():
        print(f"[cost] {bname}", flush=True)
        r, tot, mem = _measure(lambda: fn(rsp, rgt, rounds=ROUNDS, seed=SEED, log_every=ROUNDS))
        out["arms"][bname] = {"total_s": round(tot, 1), "total_min": round(tot / 60, 2),
                              "train_s": round(tot, 1), "per_round_s": round(tot / ROUNDS, 3),
                              "peak_gpu_mem_mb": round(mem, 1), "macro_acc": r.get("macro_acc")}
        OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)); print("saved ->", OUT)


if __name__ == "__main__":
    main()
