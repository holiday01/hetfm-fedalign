"""Supplementary S1: per-cancer sites, cases, slides, and the seeded
train/val/test split sizes (mean over the thirty headline seeds) of the
canonical model-independent partition.  Writes tables/supp_S1_counts.tex."""
import sys, statistics as st
from pathlib import Path
sys.path.insert(0, "<repo-root>")
from hetfm import run_r2_batch as B                                   # noqa: E402
from hetfm.assign import build_canonical_partition                    # noqa: E402
V2 = Path(__file__).resolve().parents[1]; OUT = V2 / "tables"
SEEDS = list(range(62, 92))
NAMES = ['BRCA', 'COAD', 'STAD', 'LGG', 'LUAD', 'HNSC', 'SKCM', 'CESC', 'PAAD']
per = {}
for seed in SEEDS:
    _, sp, gt, _ = build_canonical_partition(seed=seed)
    for cid, d in sp.items():
        proj = cid.split("_")[0].replace("TCGA-", "")
        e = per.setdefault(proj, {"sites": set(), "cases": set(), "slides": set(), "train": [], "val": [], "test": []})
        e["sites"].add(cid)
        for split in ("train", "val", "test"):
            for s in d[split]:
                e["cases"].add(s["case_id"]); e["slides"].add(s["filename"])
        e["_tmp"] = e.get("_tmp", {})
        for split in ("train", "val", "test"):
            e["_tmp"].setdefault((seed, split), 0); e["_tmp"][(seed, split)] += len(d[split])
rows = []; tot = {"sites": 0, "cases": 0, "slides": 0, "train": 0, "val": 0, "test": 0}
for n in NAMES:
    e = per[n]
    tr = st.mean(e["_tmp"][(s, "train")] for s in SEEDS); va = st.mean(e["_tmp"][(s, "val")] for s in SEEDS); te = st.mean(e["_tmp"][(s, "test")] for s in SEEDS)
    rows.append(f"{n} & {len(e['sites'])} & {len(e['cases'])} & {len(e['slides'])} & {tr:.1f} & {va:.1f} & {te:.1f} \\\\")
    tot["sites"] += len(e["sites"]); tot["cases"] += len(e["cases"]); tot["slides"] += len(e["slides"]); tot["train"] += tr; tot["val"] += va; tot["test"] += te
rows.append(f"Total & {tot['sites']} & {tot['cases']} & {tot['slides']} & {tot['train']:.1f} & {tot['val']:.1f} & {tot['test']:.1f} \\\\")
(OUT / "supp_S1_counts.tex").write_text("\n".join(rows) + "\n"); print("\n".join(rows))
