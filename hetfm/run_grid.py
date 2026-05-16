"""hetfm.run_grid — Week-7 ablation grid, one-factor-at-a-time (OFAT).

Around the locked default (balanced, k=256, depth=1hidden, lam_proto=1,
lam_con=1, tie=fm, 150 rds, 5 seeds) vary ONE factor at a time
(proposal §6.4). Full cross-product is wasteful and not what §6.4 asks;
OFAT isolates each factor's effect on the recovery accuracy.

Writes results/ablation_grid.json incrementally (one cell at a time) so a
crash mid-grid still leaves every completed cell on disk.
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm.run_hetfm import run                                # noqa: E402

RESULTS = Path(__file__).parent / "results"
SEEDS = [42, 43, 44, 45, 46]
ROUNDS = 150
DEFAULT = dict(k=256, depth="1hidden", lam_proto=1.0, lam_con=1.0)
FACTORS = {
    "k": [64, 128, 256, 512],
    "depth": ["linear", "1hidden", "2hidden"],
    "lam_proto": [0.0, 0.1, 1.0, 10.0],
    "lam_con": [0.0, 0.1, 1.0],
}


def _cell(params):
    accs, f1s = [], []
    for sd in SEEDS:
        r = run(scheme="balanced", seed=sd, rounds=ROUNDS, k=params["k"],
                depth=params["depth"], tie="fm",
                lam_proto=params["lam_proto"], lam_con=params["lam_con"],
                log_every=ROUNDS)
        accs.append(r["macro_acc"]); f1s.append(r["macro_f1"])
    return {
        "params": params,
        "macro_acc": {"mean": round(st.mean(accs), 4),
                      "std": round(st.pstdev(accs), 4),
                      "per_seed": [round(x, 4) for x in accs]},
        "macro_f1": {"mean": round(st.mean(f1s), 4),
                     "std": round(st.pstdev(f1s), 4)},
    }


def main():
    out = RESULTS / "ablation_grid.json"
    grid = json.loads(out.read_text()) if out.exists() else {}
    grid.setdefault("_meta", {"default": DEFAULT, "seeds": SEEDS,
                              "rounds": ROUNDS, "scheme": "balanced"})

    if "default" not in grid:
        print("\n=== ablation cell: DEFAULT ===", flush=True)
        grid["default"] = _cell(dict(DEFAULT))
        out.write_text(json.dumps(grid, indent=2))

    for factor, values in FACTORS.items():
        for v in values:
            params = dict(DEFAULT); params[factor] = v
            if params == DEFAULT:
                continue                       # == the default cell
            key = f"{factor}={v}"
            if key in grid:
                continue                       # resume-safe
            print(f"\n=== ablation cell: {key} ===", flush=True)
            grid[key] = _cell(params)
            out.write_text(json.dumps(grid, indent=2))

    d = grid["default"]["macro_acc"]["mean"]
    print("\n=== ablation grid (Δ vs default macro_acc=%.4f) ===" % d)
    for k, v in grid.items():
        if k.startswith("_") or k == "default":
            continue
        m = v["macro_acc"]["mean"]
        print(f"  {k:<16} acc={m:.4f}  Δ={m - d:+.4f}")
    print(f"saved -> {out}")
    return grid


if __name__ == "__main__":
    main()
