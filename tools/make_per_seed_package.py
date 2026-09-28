"""Build the per-seed result package that accompanies the article.

Collects every per-seed result file behind the reported tables and figures,
flattens them into one tidy CSV (one row per arm x seed x backend), adds the
label-information and selection-grid tables, copies the raw JSON files, and
writes recompute_tables.py + README.md next to them.

Output: <out-dir>/per_seed_results/
Values are read from the raw result files unchanged.
"""
import csv, json, math, shutil
from pathlib import Path

R = Path("<repo-root>/hetfm/results")
REV1 = Path("<analysis-dir>")
V3 = Path(__file__).resolve().parents[1]
OUT = V3 / "per_seed_results"
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "raw_json").mkdir(parents=True)

SEEDS = list(range(62, 92))
SCHEMES = ["stratified", "balanced", "cancer_correlated", "adversarial"]
CANCERS = ["BRCA", "COAD", "STAD", "LGG", "LUAD", "HNSC", "SKCM", "CESC", "PAAD"]

# arm name in the raw files -> (configuration as described in the article, where it is used)
DESC = {
    "method_linear": ("Reference protocol: linear architecture-tied projectors, both prototype terms, averaged head", "Tables 5-11, Figs. 2-5"),
    "method_linear_balanced_r450": ("Reference protocol, 450 rounds", "Section V-C/V-D/V-E text"),
    "method_1hidden_balanced": ("Protocol with one-hidden-layer projectors (all terms, averaged head)", "Tables 6, 7"),
    "method_2hidden_balanced": ("Protocol with two-hidden-layer projectors (all terms, averaged head)", "Table 7"),
    "abl_ce_only": ("Cross-entropy-only protocol (no prototype terms, averaged head)", "Tables 5-9, Figs. 3, 4"),
    "abl_ce_only_r450": ("Cross-entropy-only protocol, 450 rounds", "Section V-D/V-E text"),
    "abl_ce_proto": ("Cross-entropy + prototype matching", "Table 7, Supp. Table S8"),
    "abl_ce_con": ("Cross-entropy + contrastive", "Table 7, Supp. Table S8"),
    "abl_ce_only_1hidden_balanced": ("One hidden layer, cross-entropy only", "Table 7"),
    "abl_ce_con_1hidden_balanced": ("One hidden layer, cross-entropy + contrastive", "Table 7"),
    "head_none_proto": ("No head (both prototype terms; nearest global prototype)", "Tables 7, 9"),
    "head_local": ("Local heads never averaged (both prototype terms; nearest global prototype)", "Table 7"),
    "A_Conch_v15_linear": ("Homogeneous CONCH v1.5, one linear projector (pre-specified control)", "Tables 5, 6, 9, Fig. 2"),
    "A_Conch_v15_linear_r450": ("Homogeneous CONCH v1.5, one projector, 450 rounds", "Section V-C text"),
    "A_Conch_v15_3group_linear": ("Homogeneous CONCH v1.5, three group-tied projectors", "Tables 6, 9, Fig. 2"),
    "A_Conch_v15_3group_linear_r450": ("Homogeneous CONCH v1.5, three group-tied projectors, 450 rounds", "Section V-C text"),
    "A_Conch_v15_ce_only": ("Homogeneous CONCH v1.5, one projector, cross-entropy only", "Tables 6, 9"),
    "A_Conch_v15_3group_ce_only": ("Homogeneous CONCH v1.5, three group-tied projectors, cross-entropy only", "Tables 6, 9"),
    "A_Conch_v15_1hidden": ("Homogeneous CONCH v1.5, one one-hidden-layer projector", "Table 6"),
    "A_UNI_v2_linear": ("Homogeneous UNI v2, one linear projector", "Tables 6, 9"),
    "A_UNI_v2_3group_linear": ("Homogeneous UNI v2, three group-tied projectors", "Tables 6, 9"),
    "A_Virchow2_linear": ("Homogeneous Virchow2, one linear projector", "Tables 6, 9"),
    "A_Virchow2_3group_linear": ("Homogeneous Virchow2, three group-tied projectors", "Tables 6, 9"),
    "plain_fedavg_Conch_v15": ("Plain homogeneous FedAvg, CONCH v1.5, two-layer head, no projector", "Tables 5, 6, Fig. 2"),
    "base:zeropad": ("Zero-padding + FedAvg", "Table 5, Fig. 3"),
    "base:localpca": ("Per-model PCA + FedAvg", "Table 5"),
    "base:procrustes": ("PCA + Procrustes + FedAvg", "Table 5"),
    "fpkd_B": ("FedProtoKD, tuned (per-model projector averaging)", "Tables 5, 8, Fig. 4"),
    "fpkd_A": ("FedProtoKD, faithful (no aggregation; per-site-routed scores only)", "Supp. Table S5(a)"),
    "fedgh_tied": ("FedGH, tied (per-model projector averaging, server-trained head)", "Tables 5, 7, 8, 9, Figs. 3, 4"),
    "fedgh_tied_balanced_r450": ("FedGH, tied, 450 rounds", "Section V-E text"),
    "fedgh_faithful": ("FedGH, faithful (projector per site; per-site-routed and cross-site probes)", "Supp. Table S5(b)"),
    "fedgh_tied_homog3_Conch_v15": ("FedGH head rule on the homogeneous three-group CONCH control", "Table 9, Section V-E text"),
    "localonly": ("Local-only floor (each site's own projector and head, scored on the global test set)", "Section V-A, Table 5 caption, Fig. 2"),
    "backend_cpu_t24": ("Reference protocol on the CPU, 24 threads (backend study)", "Table 11"),
    "backend_cpu_t2": ("Reference protocol on the CPU, 2 threads (backend study)", "Table 11"),
    "gate_sweep_mixed_m1": ("Reference protocol, gate m=1, mixed CPU/GPU batch", "Table 11"),
    "gate_sweep_mixed_m16": ("Reference protocol, gate m=16, mixed CPU/GPU batch", "Table 11"),
    "gate_sweep_m1": ("Reference protocol, gate m=1, single GPU batch", "Section III-D, Table 11"),
    "gate_sweep_m8": ("Reference protocol, gate m=8 (default), single GPU batch", "Section III-D"),
    "gate_sweep_m16": ("Reference protocol, gate m=16, single GPU batch", "Section III-D, Table 11"),
}

COLS = ["arm", "configuration", "scheme", "seed", "rounds", "backend", "gate_m", "primary_eval",
        "macro_acc", "macro_f1", "acc_head", "f1_head", "acc_proto", "f1_proto",
        "acc_local_routed", "fpkd_head_selfeval_per_site", "fpkd_proto_per_site_routed",
        "fedgh_random_partner_acc", "fedgh_random_partner_same_class_frac", "fedgh_other_class_partner_acc",
        "silhouette_class", "xfm_prototype_cosine", "fm_probe_balanced_acc"] + \
       [f"recall_{c}" for c in CANCERS] + ["used_in", "source_file"]
rows, copied = [], []


def num(x):
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return repr(float(x)) if isinstance(x, (int, float)) else str(x)


def add(arm_key, arm, scheme, seed, rec, src, rounds=None, backend=None, gate=None, **over):
    desc, used = DESC[arm_key]
    d = rec.get("diag", {}) if isinstance(rec.get("diag"), dict) else {}
    pcr = rec.get("per_class_recall") or [None] * 9
    row = {"arm": arm, "configuration": desc, "scheme": scheme, "seed": seed,
           "rounds": rounds if rounds is not None else rec.get("rounds", 150),
           "backend": backend or ("gpu" if rec.get("device", "cuda") == "cuda" else rec.get("device")),
           "gate_m": gate if gate is not None else rec.get("min_count", ""),
           "primary_eval": rec.get("primary_eval", ""),
           "macro_acc": num(rec.get("macro_acc")), "macro_f1": num(rec.get("macro_f1")),
           "acc_head": num(rec.get("acc_head")), "f1_head": num(rec.get("f1_head")),
           "acc_proto": num(rec.get("acc_proto")), "f1_proto": num(rec.get("f1_proto")),
           "acc_local_routed": num(rec.get("acc_local_routed_artefact")),
           "fpkd_head_selfeval_per_site": num(rec.get("degeneracy", {}).get("head_selfeval_per_site")),
           "fpkd_proto_per_site_routed": num(rec.get("degeneracy", {}).get("proto_per_site_routed")),
           "fedgh_random_partner_acc": num(rec.get("cross_site_probe", {}).get("random_partner_acc")),
           "fedgh_random_partner_same_class_frac": num(rec.get("cross_site_probe", {}).get("random_partner_same_class_frac")),
           "fedgh_other_class_partner_acc": num(rec.get("cross_site_probe", {}).get("other_class_partner_acc")),
           "silhouette_class": num(d.get("silhouette_class")),
           "xfm_prototype_cosine": num(d.get("xfm_prototype_cosine")),
           "fm_probe_balanced_acc": num(d.get("fm_probe_balanced_acc")),
           "used_in": used, "source_file": src}
    for c, v in zip(CANCERS, pcr):
        row[f"recall_{c}"] = num(v)
    row.update(over)
    rows.append(row)


def raw(sub, name):
    p = R / sub / name
    dst = OUT / "raw_json" / sub / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    copied.append(f"{sub}/{name}")
    return json.loads(p.read_text())


# ---- the single-GPU batch (r2_batch): every 150-round arm on 30 seeds, plus the 450-round checks ----
per_scheme = ["method_linear", "fpkd_B", "fpkd_A", "fedgh_tied", "fedgh_faithful", "localonly"]
for base in per_scheme:
    for sc in SCHEMES:
        for s in SEEDS:
            f = f"{base}_{sc}_s{s}.json"
            if (R / "r2_batch" / f).exists():
                add(base, f"{base}_{sc}", sc, s, raw("r2_batch", f), f"r2_batch/{f}")
for sc in SCHEMES:
    arm = "abl_ce_only" if sc == "balanced" else f"abl_ce_only_{sc}"
    for s in SEEDS:
        f = f"{arm}_s{s}.json"
        add("abl_ce_only", arm, sc, s, raw("r2_batch", f), f"r2_batch/{f}")
    for s in SEEDS:
        f = f"base_{sc}_s{s}.json"
        rec = raw("r2_batch", f)
        for b in ("zeropad", "localpca", "procrustes"):
            add(f"base:{b}", f"base_{sc}:{b}", sc, s, {"macro_acc": rec[b]["macro_acc"], "macro_f1": rec[b]["macro_f1"]},
                f"r2_batch/{f}", rounds=150, backend="gpu")
balanced_only = ["method_1hidden_balanced", "method_2hidden_balanced", "abl_ce_proto", "abl_ce_con",
                 "abl_ce_only_1hidden_balanced", "abl_ce_con_1hidden_balanced", "head_none_proto", "head_local",
                 "fedgh_tied_homog3_Conch_v15"]
homog = ["A_Conch_v15_linear", "A_Conch_v15_3group_linear", "A_Conch_v15_ce_only", "A_Conch_v15_3group_ce_only",
         "A_Conch_v15_1hidden", "A_UNI_v2_linear", "A_UNI_v2_3group_linear", "A_Virchow2_linear",
         "A_Virchow2_3group_linear", "plain_fedavg_Conch_v15"]
r450 = ["method_linear_balanced_r450", "abl_ce_only_r450", "A_Conch_v15_linear_r450",
        "A_Conch_v15_3group_linear_r450", "fedgh_tied_balanced_r450"]
for arm in balanced_only + homog + r450:
    for s in SEEDS:
        f = f"{arm}_s{s}.json"
        if (R / "r2_batch" / f).exists():
            rec = raw("r2_batch", f)
            scheme = "homogeneous" if arm in homog else "balanced"
            add(arm, arm, scheme, s, rec, f"r2_batch/{f}",
                rounds=450 if arm.endswith("_r450") else rec.get("rounds", 150))

# ---- backend study: the same seeds on the CPU (24 and 2 threads) ----
for t in ("cpu_t24", "cpu_t2"):
    for s in range(62, 67):
        f = f"{t}_s{s}.json"
        rec = raw("r2_batch/backend", f)
        add(f"backend_{t}", f"backend_{t}", "balanced", s, rec, f"r2_batch/backend/{f}",
            backend=f"cpu_{rec['_task']['threads']}_threads")

# ---- minimum-count gate sweeps: the mixed CPU/GPU batch and the single-GPU batch ----
# In the mixed batch, seeds 62-73 ran on CPU workers and seeds 74-91 on the GPU worker
# (worker logs mincount_m{1,16}_w{0,1,2}.log and mincount_gpu_m{1,16}.log); the files record no device.
for m in (1, 16):
    for s in SEEDS:
        f = f"m{m}_s{s}.json"
        rec = raw("mincount", f)
        add(f"gate_sweep_mixed_m{m}", f"gate_sweep_mixed_m{m}", "balanced", s, rec, f"mincount/{f}",
            rounds=150, backend="cpu_mixed_batch" if s <= 73 else "gpu", gate=m)
for m in (1, 8, 16):
    for s in SEEDS:
        f = f"m{m}_s{s}.json"
        rec = raw("mincount_clean", f)
        add(f"gate_sweep_m{m}", f"gate_sweep_m{m}", "balanced", s, rec, f"mincount_clean/{f}",
            rounds=150, backend="gpu", gate=m)

with open(OUT / "per_seed_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)

# ---- Table 3: label information carried by model identity, per seed ----
mi = json.loads((REV1 / "mi_fm_y.json").read_text())
with open(OUT / "label_information_per_seed.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["scheme", "seed", "I_FM_Y_bits", "H_Y_bits", "I_over_H_Y", "identity_only_macro_acc",
                "classes_seen_UNI_v2", "classes_seen_CONCH_v1.5", "classes_seen_Virchow2"])
    for sc in SCHEMES:
        for r in mi["per_seed"][sc]:
            c = r["classes_seen_per_fm"]
            w.writerow([sc, r["seed"], r["I_FM_Y_bits"], r["H_Y_bits"], r["normalized_I_over_HY"],
                        r["fm_only_macro_acc"], c.get("UNI_v2", 0), c.get("Conch_v15", 0), c.get("Virchow2", 0)])
shutil.copy2(REV1 / "mi_fm_y.json", OUT / "raw_json" / "mi_fm_y.json")

# ---- Supplementary Table S1: five-seed selection grid ----
g = json.loads((R / "ablation_grid.json").read_text())
with open(OUT / "selection_grid_per_seed.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["grid_row", "k", "depth", "lam_proto", "lam_con", "seed", "macro_acc"])
    for name, e in g.items():
        if name == "_meta":
            continue
        for s, v in zip(g["_meta"]["seeds"], e["macro_acc"]["per_seed"]):
            p = e["params"]
            w.writerow([name, p["k"], p["depth"], p["lam_proto"], p["lam_con"], s, v])
shutil.copy2(R / "ablation_grid.json", OUT / "raw_json" / "ablation_grid.json")

# ---- Table 4: cost (one measured seed + the analytic parameter/traffic counts) ----
shutil.copy2(R / "r2_cost.json", OUT / "raw_json" / "r2_cost.json")
shutil.copy2(REV1 / "cost_analytic.json", OUT / "raw_json" / "cost_analytic.json")

# ---- README and the recompute script (kept in tools/per_seed_package/) ----
for f in ("README.md", "recompute_tables.py"):
    shutil.copy2(Path(__file__).resolve().parent / "per_seed_package" / f, OUT / f)

print(f"per_seed_results.csv: {len(rows)} rows, {len({r['arm'] for r in rows})} arms")
print(f"raw JSON files copied: {len(copied)} (+ mi_fm_y, ablation_grid, r2_cost, cost_analytic)")
