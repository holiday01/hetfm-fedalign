"""hetfm.run_r2_batch — single-backend GPU batch.  Every arm below is run on the 30 headline seeds
(62-91), 150 rounds, in one batch on one GPU, so that the consolidated main
table has a single provenance.  Results are cached one JSON per (task, seed)
under results/r2_batch/; nothing in results/*.json from the submission is
touched or overwritten (run_hetfm.run is deliberately not used).

  cd /home/holiday01/fl_wsi
  python -m hetfm.run_r2_batch --worker 0 --nworkers 3      # one shard
  python -m hetfm.run_r2_batch --list                       # task inventory
  python -m hetfm.run_r2_batch --summarize                  # aggregate

Priority 1 (core arms), 2 (matched controls, FedGH),
3 (dimension-matching baselines re-run for single-backend provenance).
Tasks are ordered seed-major inside each priority so an interrupted batch
still leaves every arm at the same number of seeds.
"""
from __future__ import annotations

import argparse
import functools
import json
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm import assign                                       # noqa: E402
from hetfm.baselines_het import BASELINES                      # noqa: E402
from hetfm.fedprotokd import train as fpkd                     # noqa: E402
from hetfm.fedgh import train as fedgh                         # noqa: E402
from hetfm.proto_anchor import homogeneous_fedavg              # noqa: E402
from hetfm.r2_trainer import train_r2                          # noqa: E402
from hetfm.run_hetfm import probe_fm_dims                      # noqa: E402

RESULTS = Path(__file__).parent / "results"
CACHE = RESULTS / "r2_batch"
SEEDS = list(range(62, 92))
R450_SEEDS = [62, 63, 64, 65, 66]
ROUNDS = 150
SCHEMES = ["balanced", "stratified", "cancer_correlated", "adversarial"]
HEAD = dict(k=256, depth="linear", tie="fm", lam_proto=1.0, lam_con=1.0)
_FMD = {}


def _fmd():
    if not _FMD:
        _FMD.update(probe_fm_dims(assign.FEAT_ROOT, assign.FMS))
    return _FMD


@functools.lru_cache(maxsize=2)
def _partition(seed):
    """Memoised per seed: tasks are seed-major, so ~30 arms share one build.
    assign.route deep-copies, so the cached splits are never mutated."""
    _, sp, gt, _ = assign.build_canonical_partition(seed=seed)
    clab = assign.client_labels_from_splits(sp)
    return sp, gt, clab


def _route(scheme, seed):
    sp, gt, clab = _partition(seed)
    asn = assign.assign_fms(list(sp), scheme, seed, client_label=clab)
    rsp, rgt = assign.route(sp, gt, asn)
    return rsp, rgt, asn


# ---- task implementations ---------------------------------------------------
def t_method(scheme, seed, rounds, **over):
    rsp, rgt, asn = _route(scheme, seed)
    cfg = dict(HEAD); cfg.update(over)
    return train_r2(rsp, rgt, asn, _fmd(), rounds=rounds, seed=seed,
                    log_every=rounds, **cfg)


def t_homog(fm, seed, rounds, **over):
    """Same-method homogeneous control: every site holds `fm`."""
    rsp, rgt, asn = _route(f"homogeneous:{fm}", seed)
    cfg = dict(HEAD); cfg.update(over)
    return train_r2(rsp, rgt, asn, _fmd(), rounds=rounds, seed=seed,
                    log_every=rounds, **cfg)


def t_homog_3group(fm, seed, rounds, **over):
    """Capacity-matched homogeneous control: every site holds `fm`, but sites
    are split into three projector groups exactly as the balanced assignment
    splits them across models; each group owns its own projector and averages
    only within the group.  Identical structure, parameter count per group,
    aggregation and optimisation to the heterogeneous protocol; the only
    difference is that all three groups see the same foundation model."""
    sp, gt, clab = _partition(seed)
    bal = assign.assign_fms(list(sp), "balanced", seed, client_label=clab)
    gname = {f: f"G{i}" for i, f in enumerate(assign.FMS)}
    groups = {cid: gname[bal[cid]] for cid in bal}
    homog = {cid: fm for cid in sp}
    rsp, rgt = assign.route(sp, gt, homog)
    d = _fmd()[fm]
    fmd3 = {g: d for g in gname.values()}
    cfg = dict(HEAD); cfg.update(over)
    return train_r2(rsp, rgt, groups, fmd3, rounds=rounds, seed=seed,
                    log_every=rounds, **cfg)


def t_plain_fedavg(fm, seed, rounds):
    rsp, rgt, _ = _route(f"homogeneous:{fm}", seed)
    t0 = time.time()
    r = homogeneous_fedavg(rsp, rgt, _fmd()[fm], rounds=rounds, seed=seed,
                           log_every=rounds)
    r["timing"] = {"total_s": round(time.time() - t0, 2)}
    return r


def t_fpkd(scheme, seed, rounds):
    rsp, rgt, asn = _route(scheme, seed)
    t0 = time.time()
    r = fpkd(rsp, rgt, asn, _fmd(), rounds=rounds, seed=seed, log_every=rounds,
             aggregate_projector=True, k=HEAD["k"], depth=HEAD["depth"])
    r["timing"] = {"total_s": round(time.time() - t0, 2)}
    return r


def t_fedgh(scheme, seed, rounds, tied):
    rsp, rgt, asn = _route(scheme, seed)
    return fedgh(rsp, rgt, asn, _fmd(), rounds=rounds, seed=seed,
                 log_every=rounds, aggregate_projector=tied,
                 k=HEAD["k"], depth=HEAD["depth"])


def t_baselines(scheme, seed, rounds):
    rsp, rgt, _ = _route(scheme, seed)
    out = {}
    for name, fn in BASELINES.items():
        t0 = time.time()
        rb = fn(rsp, rgt, rounds=rounds, seed=seed, log_every=rounds)
        out[name] = {"macro_acc": rb["macro_acc"], "macro_f1": rb.get("macro_f1"),
                     "total_s": round(time.time() - t0, 2)}
    return out


def t_fpkd_faithful(scheme, seed, rounds):
    """Faithful FedProtoKD (variant A: per-site projector and head, nothing
    aggregated) at the headline seeds; returns the per-site-routed artefacts."""
    rsp, rgt, asn = _route(scheme, seed)
    t0 = time.time()
    r = fpkd(rsp, rgt, asn, _fmd(), rounds=rounds, seed=seed, log_every=rounds,
             aggregate_projector=False, k=HEAD["k"], depth=HEAD["depth"])
    r["timing"] = {"total_s": round(time.time() - t0, 2)}
    return r


def t_fedgh_homog_3group(fm, seed, rounds):
    """FedGH's server-trained head on the grouping-matched homogeneous
    federation: every site holds `fm`, three projector groups as in the
    balanced assignment, tied FedGH otherwise unchanged.  Isolates the
    head-training rule inside a single-model federation."""
    sp, gt, clab = _partition(seed)
    bal = assign.assign_fms(list(sp), "balanced", seed, client_label=clab)
    gname = {f: f"G{i}" for i, f in enumerate(assign.FMS)}
    groups = {cid: gname[bal[cid]] for cid in bal}
    homog = {cid: fm for cid in sp}
    rsp, rgt = assign.route(sp, gt, homog)
    d = _fmd()[fm]
    fmd3 = {g: d for g in gname.values()}
    return fedgh(rsp, rgt, groups, fmd3, rounds=rounds, seed=seed,
                 log_every=rounds, aggregate_projector=True,
                 k=HEAD["k"], depth=HEAD["depth"])


def t_localonly(scheme, seed, rounds):
    from hetfm.r2_localonly import train_local_only                 # noqa: E402
    rsp, rgt, asn = _route(scheme, seed)
    return train_local_only(rsp, rgt, asn, _fmd(), rounds=rounds, seed=seed,
                            k=HEAD["k"])


# ---- task registry: name -> (priority, callable(seed, rounds)) --------------
def registry():
    R = {}
    # priority 1
    for sc in SCHEMES:
        R[f"method_linear_{sc}"] = (1, lambda s, r, sc=sc: t_method(sc, s, r))
    R["A_Conch_v15_linear"] = (1, lambda s, r: t_homog("Conch_v15", s, r))
    R["abl_ce_only"] = (1, lambda s, r: t_method("balanced", s, r, lam_proto=0.0, lam_con=0.0))
    R["abl_ce_proto"] = (1, lambda s, r: t_method("balanced", s, r, lam_con=0.0))
    R["abl_ce_con"] = (1, lambda s, r: t_method("balanced", s, r, lam_proto=0.0))
    R["head_none_proto"] = (1, lambda s, r: t_method("balanced", s, r, head_mode="none"))
    R["head_local"] = (1, lambda s, r: t_method("balanced", s, r, head_mode="local"))
    R["plain_fedavg_Conch_v15"] = (1, lambda s, r: t_plain_fedavg("Conch_v15", s, r))
    for sc in SCHEMES:
        R[f"fpkd_B_{sc}"] = (1, lambda s, r, sc=sc: t_fpkd(sc, s, r))
    # priority 2
    R["A_Conch_v15_1hidden"] = (2, lambda s, r: t_homog("Conch_v15", s, r, depth="1hidden"))
    R["method_1hidden_balanced"] = (2, lambda s, r: t_method("balanced", s, r, depth="1hidden"))
    R["A_Conch_v15_3group_linear"] = (2, lambda s, r: t_homog_3group("Conch_v15", s, r))
    R["method_2hidden_balanced"] = (2, lambda s, r: t_method("balanced", s, r, depth="2hidden"))
    R["A_Conch_v15_ce_only"] = (2, lambda s, r: t_homog("Conch_v15", s, r, lam_proto=0.0, lam_con=0.0))
    R["A_Conch_v15_3group_ce_only"] = (2, lambda s, r: t_homog_3group("Conch_v15", s, r, lam_proto=0.0, lam_con=0.0))
    R["A_UNI_v2_linear"] = (2, lambda s, r: t_homog("UNI_v2", s, r))
    R["A_Virchow2_linear"] = (2, lambda s, r: t_homog("Virchow2", s, r))
    for sc in ["stratified", "cancer_correlated", "adversarial"]:
        R[f"abl_ce_only_{sc}"] = (2, lambda s, r, sc=sc: t_method(sc, s, r, lam_proto=0.0, lam_con=0.0))
    R["abl_ce_only_1hidden_balanced"] = (2, lambda s, r: t_method("balanced", s, r, depth="1hidden", lam_proto=0.0, lam_con=0.0))
    R["abl_ce_con_1hidden_balanced"] = (2, lambda s, r: t_method("balanced", s, r, depth="1hidden", lam_proto=0.0))
    for sc in SCHEMES:
        R[f"fedgh_tied_{sc}"] = (2, lambda s, r, sc=sc: t_fedgh(sc, s, r, True))
    for sc in SCHEMES:
        R[f"fedgh_faithful_{sc}"] = (2, lambda s, r, sc=sc: t_fedgh(sc, s, r, False))
    # under-training check: does the single-projector homogeneous control catch up
    # with more rounds?  (rounds argument ignored: fixed 450)
    R["A_Conch_v15_linear_r450"] = (2, lambda s, r: t_homog("Conch_v15", s, 450))
    R["method_linear_balanced_r450"] = (2, lambda s, r: t_method("balanced", s, 450))
    R["A_Conch_v15_3group_linear_r450"] = (2, lambda s, r: t_homog_3group("Conch_v15", s, 450))
    R["abl_ce_only_r450"] = (2, lambda s, r: t_method("balanced", s, 450, lam_proto=0.0, lam_con=0.0))
    R["fedgh_tied_balanced_r450"] = (2, lambda s, r: t_fedgh("balanced", s, 450, True))
    # priority 3
    for sc in SCHEMES:
        R[f"base_{sc}"] = (3, lambda s, r, sc=sc: t_baselines(sc, s, r))
    # priority 4 (added September 2026): capacity
    # bracket around the heterogeneous protocol, faithful FedProtoKD and the
    # local-only floor at the headline seeds, and FedGH's head rule inside a
    # homogeneous federation
    R["A_UNI_v2_3group_linear"] = (4, lambda s, r: t_homog_3group("UNI_v2", s, r))
    R["A_Virchow2_3group_linear"] = (4, lambda s, r: t_homog_3group("Virchow2", s, r))
    R["fedgh_tied_homog3_Conch_v15"] = (4, lambda s, r: t_fedgh_homog_3group("Conch_v15", s, r))
    for sc in SCHEMES:
        R[f"fpkd_A_{sc}"] = (4, lambda s, r, sc=sc: t_fpkd_faithful(sc, s, r))
    for sc in SCHEMES:
        R[f"localonly_{sc}"] = (4, lambda s, r, sc=sc: t_localonly(sc, s, r))
    return R


def task_list():
    R = registry()
    out = []
    for pr in (1, 2, 3, 4):
        names = [n for n, (p, _) in R.items() if p == pr]
        for seed in SEEDS:
            for n in names:
                if n.endswith("_r450") and seed not in R450_SEEDS:
                    continue            # the convergence check is a 5-seed study
                out.append((n, seed))
    return out, R


def _path(name, seed):
    return CACHE / f"{name}_s{seed}.json"


def work(worker, nworkers, rounds=ROUNDS, only=None, seeds=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    tasks, R = task_list()
    if only:
        tasks = [t for t in tasks if t[0] in only]
    if seeds:
        tasks = [t for t in tasks if t[1] in seeds]
    mine = [t for i, t in enumerate(tasks) if i % nworkers == worker]
    print(f"[r2 w{worker}] {len(mine)}/{len(tasks)} tasks host={socket.gethostname()}",
          flush=True)
    for i, (name, seed) in enumerate(mine, 1):
        p = _path(name, seed)
        if p.exists():
            continue
        print(f"[r2 w{worker} {i}/{len(mine)}] {name} seed {seed} "
              f"{datetime.now().isoformat(timespec='seconds')}", flush=True)
        for attempt in (1, 2):
            try:
                t0 = time.time()
                res = R[name][1](seed, rounds)
                res["_task"] = {"name": name, "seed": seed, "rounds": rounds,
                                "worker": worker, "host": socket.gethostname(),
                                "wall_s": round(time.time() - t0, 2),
                                "finished": datetime.now().isoformat(timespec="seconds")}
                p.write_text(json.dumps(res, indent=1, default=str))
                del res
                try:
                    import gc, torch
                    gc.collect(); torch.cuda.empty_cache()
                except Exception:                               # noqa: BLE001
                    pass
                break
            except Exception as e:                              # noqa: BLE001
                print(f"  ATTEMPT {attempt} FAILED {name} seed {seed}: "
                      f"{type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:                               # noqa: BLE001
                    pass
                if attempt == 1:
                    time.sleep(20)
    print(f"[r2 w{worker}] done", flush=True)


# ---- summarize ---------------------------------------------------------------
def _ms(xs):
    import statistics as st
    return {"mean": round(st.mean(xs), 4),
            "std": round(st.pstdev(xs), 4) if len(xs) > 1 else 0.0,
            "n": len(xs), "per_seed": [round(x, 4) for x in xs]}


def _paired(a, b):
    import statistics as st
    import numpy as np
    d = [x - y for x, y in zip(a, b)]
    rng = np.random.default_rng(0)
    arr = np.asarray(d)
    means = rng.choice(arr, size=(10000, arr.size), replace=True).mean(axis=1)
    o = {"n": len(d), "mean_diff": round(st.mean(d), 4),
         "n_a_gt_b": f"{sum(1 for x in d if x > 0)}/{len(d)}",
         "boot_ci95": [round(float(np.percentile(means, 2.5)), 4),
                       round(float(np.percentile(means, 97.5)), 4)]}
    try:
        from scipy import stats
        o["wilcoxon_p"] = float(f"{stats.wilcoxon(a, b).pvalue:.3g}")
        o["paired_t_p"] = float(f"{stats.ttest_rel(a, b).pvalue:.3g}")
    except Exception as e:                                      # noqa: BLE001
        o["stat_error"] = str(e)
    return o


def _seeds_done(name):
    return [s for s in SEEDS if _path(name, s).exists()]


def load(name, key="macro_acc"):
    """Per-seed values for the seeds that exist (partial arms are summarised
    with their own n so the generators can be tested before the batch ends)."""
    seeds = _seeds_done(name)
    if not seeds:
        return None
    vals = []
    for seed in seeds:
        r = json.loads(_path(name, seed).read_text())
        vals.append(r[key] if key in r else None)
    return vals


def summarize():
    tasks, R = task_list()
    names = list(dict.fromkeys(n for n, _ in tasks))
    done = {n: sum(1 for s in SEEDS if _path(n, s).exists()) for n in names}
    out = {"seeds": SEEDS, "rounds": ROUNDS, "config": HEAD,
           "completed_seeds_per_task": done, "arms": {}}
    for n in names:
        if n.startswith("base_"):
            vals = load(n, key="zeropad")
            if vals is None:
                continue
            rows = [json.loads(_path(n, s).read_text()) for s in _seeds_done(n)]
            out["arms"][n] = {b: {"macro_acc": _ms([r[b]["macro_acc"] for r in rows]),
                                  "macro_f1": _ms([r[b]["macro_f1"] for r in rows
                                                   if r[b]["macro_f1"] is not None])}
                              for b in BASELINES}
            continue
        if n.startswith("fpkd_A_"):
            rows = [json.loads(_path(n, s).read_text()) for s in _seeds_done(n)]
            if not rows:
                continue
            out["arms"][n] = {"seeds": _seeds_done(n),
                              "proto_per_site_routed": _ms([r["degeneracy"]["proto_per_site_routed"] for r in rows]),
                              "head_selfeval_per_site": _ms([r["degeneracy"]["head_selfeval_per_site"] for r in rows])}
            continue
        vals = load(n)
        if vals is None or any(v is None for v in vals):
            continue
        rows = [json.loads(_path(n, s).read_text()) for s in _seeds_done(n)]
        e = {"macro_acc": _ms(vals), "seeds": _seeds_done(n)}
        for k in ("macro_acc_weighted", "macro_acc_best_site"):
            if all(k in r for r in rows):
                e[k] = _ms([r[k] for r in rows])
        f1 = [r.get("macro_f1") for r in rows]
        if all(v is not None for v in f1):
            e["macro_f1"] = _ms(f1)
        for k in ("acc_proto", "acc_head", "acc_local_routed_artefact"):
            if all(k in r for r in rows):
                e[k] = _ms([r[k] for r in rows])
        if all("per_class_recall" in r for r in rows):
            import numpy as np
            M = np.array([r["per_class_recall"] for r in rows], dtype=float)
            e["per_class_recall_mean"] = [round(float(x), 4) for x in np.nanmean(M, 0)]
        if all("diag" in r for r in rows):
            for k in ("silhouette_class", "xfm_prototype_cosine", "fm_probe_balanced_acc"):
                xs = [r["diag"].get(k) for r in rows]
                if all(isinstance(x, float) for x in xs):
                    import math
                    xs2 = [x for x in xs if not math.isnan(x)]
                    if xs2:
                        e[k] = _ms(xs2)
        if all("timing" in r for r in rows):
            e["total_s_mean"] = round(sum(r["timing"]["total_s"] for r in rows) / len(rows), 1)
        if all("peak_gpu_mem_mb" in r for r in rows):
            e["peak_gpu_mem_mb_max"] = max(r["peak_gpu_mem_mb"] for r in rows)
        out["arms"][n] = e
    A = out["arms"]

    def has(*ns):
        return all(n in A and "macro_acc" in A[n] for n in ns)

    def ps(n, sub=None):
        """per-seed values of arm n (sub-baseline for base_* arms), keyed by seed"""
        if sub is None:
            return dict(zip(A[n]["seeds"], A[n]["macro_acc"]["per_seed"]))
        return dict(zip(_seeds_done(n), A[n][sub]["macro_acc"]["per_seed"]))

    def _paired_arms(a, b, suba=None, subb=None):
        da, db = ps(a, suba), ps(b, subb)
        common = [s for s in SEEDS if s in da and s in db]
        return _paired([da[s] for s in common], [db[s] for s in common])

    con = {}
    m = "method_linear_balanced"
    for other in ["A_Conch_v15_linear", "A_UNI_v2_linear", "A_Virchow2_linear",
                  "A_Conch_v15_3group_linear", "abl_ce_only", "abl_ce_proto",
                  "abl_ce_con", "head_none_proto", "head_local",
                  "plain_fedavg_Conch_v15", "fpkd_B_balanced", "fedgh_tied_balanced",
                  "fedgh_faithful_balanced", "method_1hidden_balanced",
                  "method_2hidden_balanced"]:
        if has(m, other):
            con[f"{m}_minus_{other}"] = _paired_arms(m, other)
    if has("method_1hidden_balanced", "A_Conch_v15_1hidden"):
        con["method_1hidden_minus_A_1hidden"] = _paired_arms(
            "method_1hidden_balanced", "A_Conch_v15_1hidden")
    for sc in SCHEMES:
        mm = f"method_linear_{sc}"
        for other in ["A_Conch_v15_linear", f"fpkd_B_{sc}", f"fedgh_tied_{sc}",
                      f"fedgh_faithful_{sc}"]:
            if has(mm, other):
                con[f"{mm}_minus_{other}"] = _paired_arms(mm, other)
        bn = f"base_{sc}"
        if bn in A and has(mm):
            for b in BASELINES:
                con[f"{mm}_minus_{b}"] = _paired_arms(mm, bn, None, b)
    for sc in ["stratified", "cancer_correlated", "adversarial"]:
        if has(f"method_linear_{sc}", f"abl_ce_only_{sc}"):
            con[f"method_linear_{sc}_minus_abl_ce_only_{sc}"] = _paired_arms(
                f"method_linear_{sc}", f"abl_ce_only_{sc}")
    for other in ["A_Conch_v15_ce_only", "A_Conch_v15_3group_ce_only", "A_Conch_v15_linear"]:
        if has("abl_ce_only", other):
            con[f"abl_ce_only_minus_{other}"] = _paired_arms("abl_ce_only", other)
    for sc in ["stratified", "cancer_correlated", "adversarial"]:
        if has(f"abl_ce_only_{sc}", "A_Conch_v15_ce_only"):
            con[f"abl_ce_only_{sc}_minus_A_Conch_v15_ce_only"] = _paired_arms(f"abl_ce_only_{sc}", "A_Conch_v15_ce_only")
    for other in ["abl_ce_only_1hidden_balanced", "abl_ce_con_1hidden_balanced"]:
        if has("method_1hidden_balanced", other):
            con[f"method_1hidden_balanced_minus_{other}"] = _paired_arms(
                "method_1hidden_balanced", other)
    for sc in SCHEMES:
        ce = "abl_ce_only" if sc == "balanced" else f"abl_ce_only_{sc}"
        for a, b in [(f"fedgh_tied_{sc}", ce), (f"fedgh_tied_{sc}", "A_Conch_v15_linear"),
                     (f"fedgh_tied_{sc}", f"fpkd_B_{sc}"), (ce, "A_Conch_v15_linear")]:
            if has(a, b):
                con[f"{a}_minus_{b}"] = _paired_arms(a, b)
    for a, b in [("fedgh_tied_balanced", "A_Conch_v15_3group_ce_only"),
                 ("fedgh_tied_balanced", "A_Virchow2_linear"),
                 ("abl_ce_only", "A_Virchow2_linear"),
                 ("A_Conch_v15_3group_linear", "A_Conch_v15_linear"),
                 ("A_Conch_v15_3group_ce_only", "A_Conch_v15_ce_only"),
                 ("A_Conch_v15_linear_r450", "A_Conch_v15_linear"),
                 ("method_linear_balanced_r450", "method_linear_balanced"),
                 ("A_Conch_v15_3group_linear_r450", "A_Conch_v15_3group_linear"),
                 ("abl_ce_only_r450", "abl_ce_only"), ("fedgh_tied_balanced_r450", "fedgh_tied_balanced"),
                 ("method_linear_balanced_r450", "A_Conch_v15_3group_linear_r450"),
                 ("method_linear_balanced_r450", "A_Conch_v15_linear_r450"),
                 ("fedgh_tied_balanced_r450", "method_linear_balanced_r450"),
                 ("fedgh_tied_balanced_r450", "abl_ce_only_r450"),
                 ("abl_ce_only_r450", "method_linear_balanced_r450"),
                 ("method_linear_balanced", "A_UNI_v2_3group_linear"),
                 ("method_linear_balanced", "A_Virchow2_3group_linear"),
                 ("A_UNI_v2_3group_linear", "A_UNI_v2_linear"),
                 ("A_Virchow2_3group_linear", "A_Virchow2_linear"),
                 ("A_Virchow2_3group_linear", "A_Conch_v15_3group_linear"),
                 ("A_UNI_v2_3group_linear", "A_Conch_v15_3group_linear"),
                 ("fedgh_tied_balanced", "fedgh_tied_homog3_Conch_v15"),
                 ("fedgh_tied_homog3_Conch_v15", "A_Conch_v15_3group_ce_only"),
                 ("fedgh_tied_homog3_Conch_v15", "A_Conch_v15_3group_linear")]:
        if has(a, b):
            con[f"{a}_minus_{b}"] = _paired_arms(a, b)
    for sc in SCHEMES:
        if has(f"method_linear_{sc}", f"localonly_{sc}"):
            con[f"method_linear_{sc}_minus_localonly_{sc}"] = _paired_arms(f"method_linear_{sc}", f"localonly_{sc}")
    out["contrasts"] = con
    # reproduction check against the submitted (frozen) per-seed values
    try:
        pr = json.loads((RESULTS / "prereg_confirm.json").read_text())
        se = json.loads((RESULTS / "seedext_rq4_n30.json").read_text())
        rep = {}
        def _sub(name, ref):
            d = ps(name); return [d[s] for s in SEEDS if s in d], [ref[i] for i, s in enumerate(SEEDS) if s in d]
        if has("method_linear_balanced"):
            rep["method_linear_balanced_vs_prereg"] = _maxabs(
                *_sub("method_linear_balanced", pr["method_linear_balanced"]["per_seed"]))
        if has("A_Conch_v15_linear"):
            rep["A_control_vs_prereg"] = _maxabs(
                *_sub("A_Conch_v15_linear", pr["A_linear_homog_CONCH"]["per_seed"]))
        if has("fpkd_B_balanced"):
            rep["fpkd_B_balanced_vs_prereg"] = _maxabs(
                *_sub("fpkd_B_balanced", pr["fedprotokd_B_balanced"]["per_seed"]))
        for sc in ["stratified", "cancer_correlated", "adversarial"]:
            if has(f"method_linear_{sc}"):
                rep[f"method_linear_{sc}_vs_seedext(mixed backend)"] = _maxabs(
                    *_sub(f"method_linear_{sc}", se["schemes"][sc]["method_linear"]["per_seed"]))
        out["reproduction_vs_submitted"] = rep
    except Exception as e:                                      # noqa: BLE001
        out["reproduction_error"] = str(e)
    p = RESULTS / "r2_summary.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1)[:6000])
    print("arms:", {n: (A[n]["macro_acc"]["mean"] if "macro_acc" in A[n] else "base")
                    for n in A})
    print("saved ->", p)
    return out


def _maxabs(a, b):
    import numpy as np
    d = np.abs(np.asarray(a) - np.asarray(b))
    return {"max_abs_diff": round(float(d.max()), 5),
            "mean_diff": round(float((np.asarray(a) - np.asarray(b)).mean()), 5),
            "n_identical_4dp": int((d < 5e-5).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        tasks, R = task_list()
        names = list(dict.fromkeys(n for n, _ in tasks))
        for n in names:
            print(R[n][0], n)
        print(len(tasks), "tasks")
    elif a.summarize:
        summarize()
    else:
        work(a.worker, a.nworkers, a.rounds, a.only, a.seeds)


if __name__ == "__main__":
    main()
