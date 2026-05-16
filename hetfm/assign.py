"""hetfm.assign — FM-independent canonical partition + FM-to-site routing.

Design rule (week1_impl_spec.md §1): build the client partition and the
case-level split ONCE, on the slide set common to all FMs, independent of FM.
The FM assignment then only rewrites which FM subdirectory each sample's
feature `path` points to. Mandatory order:
  intersect slides -> build partition -> case-level split -> min_samples -> FM-route
so every scheme/baseline/method differs ONLY on the FM axis with identical
clients and identical splits per seed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.partition import TCGAFLPartitioner  # noqa: E402

FEAT_ROOT = "/mnt/10t/cached_slide_features_npy"
FMS = ["UNI_v2", "Conch_v15", "Virchow2"]
REF_FM = "UNI_v2"


def slide_intersection(feat_root: str, fms: List[str]) -> set:
    """Filenames (e.g. 'TCGA-..-DX1.<uuid>.npy') present in *every* FM dir."""
    sets = []
    for fm in fms:
        d = Path(feat_root) / fm
        sets.append({p.name for p in d.glob("*.npy")})
    inter = set.intersection(*sets) if sets else set()
    return inter


def build_canonical_partition(feat_root: str = FEAT_ROOT,
                              ref_fm: str = REF_FM,
                              fms: List[str] = None,
                              seed: int = 42,
                              min_samples: int = 10):
    """Return (clients, splits, global_test, stats) on the FM-common slide set.

    Paths still point at `ref_fm`; call route() to retarget to an assignment.
    """
    fms = fms or FMS
    inter = slide_intersection(feat_root, fms)

    # scan with min_samples=1 so the partitioner's own filter does NOT run
    # before we intersect (order matters — see module docstring).
    p = TCGAFLPartitioner(feat_root, ref_fm, min_samples=1, seed=seed)
    p.scan_features()

    stages = {"clients_pre_intersect": len(p.clients)}
    # intersect-filter, then re-apply min_samples manually
    filtered: Dict[str, list] = {}
    for cid, samples in p.clients.items():
        keep = [s for s in samples if s["filename"] in inter]
        if len(keep) >= min_samples:
            filtered[cid] = keep
    p.clients = filtered
    stages["clients_post_intersect_minsamples"] = len(p.clients)

    splits = p.split_clients()
    global_test = p.build_global_test_set(splits)
    stats = p.summary()
    stats["intersection_size"] = len(inter)
    stats["stages"] = stages
    return p.clients, splits, global_test, stats


def client_labels_from_splits(splits: dict) -> Dict[str, int]:
    """client_id -> dominant 9-class cancer label, from the canonical
    partition (FM-independent — labels are a property of the TSS site, not
    the FM). Each TSS client is single-project so any sample's label works.
    """
    out: Dict[str, int] = {}
    for cid, d in splits.items():
        for sp in ("train", "val", "test"):
            if d.get(sp):
                out[cid] = int(d[sp][0]["cancer_label"])
                break
    return out


def assign_fms(client_ids: List[str], scheme: str, seed: int,
               fms: List[str] = None,
               client_label: Dict[str, int] = None) -> Dict[str, str]:
    """Deterministic client_id -> FM map (proposal §6.1, 4 schemes).

    'homogeneous:<FM>', 'balanced' need no labels. 'stratified',
    'cancer_correlated', 'adversarial' need `client_label` (a
    cancer-label map; FM-independent so it never breaks the canonical-
    partition-before-assignment rule). Seeds give 5 distinct assignments.
    """
    fms = fms or FMS
    nf = len(fms)
    cids = sorted(client_ids)  # sort first => order independent of dict order
    if scheme.startswith("homogeneous:"):
        fm = scheme.split(":", 1)[1]
        if fm not in fms:
            raise ValueError(f"unknown FM {fm!r}; known {fms}")
        return {c: fm for c in cids}
    if scheme == "balanced":
        rng = np.random.default_rng(seed)
        order = list(cids)
        rng.shuffle(order)
        return {c: fms[i % nf] for i, c in enumerate(order)}

    if scheme not in ("stratified", "cancer_correlated", "adversarial"):
        raise ValueError(f"unknown scheme {scheme!r}")
    if client_label is None:
        raise ValueError(f"scheme {scheme!r} requires client_label "
                         f"(use assign.client_labels_from_splits)")
    rng = np.random.default_rng(seed)
    classes = sorted({client_label[c] for c in cids})

    if scheme == "stratified":
        # Equalize per-FM class coverage: within each class, round-robin
        # that class's sites across FMs (seeded). Every FM sees every
        # class ~equally -> easiest control.
        out: Dict[str, str] = {}
        for cl in classes:
            grp = [c for c in cids if client_label[c] == cl]
            rng.shuffle(grp)
            off = int(rng.integers(nf))          # rotate start per class
            for i, c in enumerate(grp):
                out[c] = fms[(i + off) % nf]
        return out

    if scheme == "cancer_correlated":
        # Each class has a preferred FM (seed-permuted); a site gets its
        # class's preferred FM w.p. P, else a uniform other FM. Skewed,
        # stochastic -> each FM sees a skewed class subset.
        P = 0.7
        perm = list(rng.permutation(nf))
        pref = {cl: fms[perm[i % nf]] for i, cl in enumerate(classes)}
        out = {}
        for c in cids:
            p = pref[client_label[c]]
            if rng.random() < P:
                out[c] = p
            else:
                others = [f for f in fms if f != p]
                out[c] = others[int(rng.integers(len(others)))]
        return out

    # adversarial: FM is a deterministic function of class -> group of
    # ~3 classes per FM (seed-permuted class order & FM order). Several
    # FMs NEVER see some classes -> hardest; alignment must propagate
    # through prototypes shared by other sites. Stress test for H4.
    cls = list(rng.permutation(classes))
    fm_order = list(rng.permutation(nf))
    per = -(-len(cls) // nf)                       # ceil split
    cls_fm = {}
    for gi in range(nf):
        for cl in cls[gi * per:(gi + 1) * per]:
            cls_fm[cl] = fms[fm_order[gi]]
    return {c: cls_fm[client_label[c]] for c in cids}


def route(splits: dict, global_test: list, assignment: Dict[str, str],
          feat_root: str = FEAT_ROOT):
    """Deep-copy and retarget every sample's `path` to its client's assigned FM.

    Adds sample['fm'] (consumed by the Week-2 per-FM-type-tied projector).
    Rebuilds global_test from the routed splits so its paths are consistent.
    """
    rsplits = copy.deepcopy(splits)
    for cid, data in rsplits.items():
        fm = assignment[cid]
        for split_name in ("train", "val", "test"):
            for s in data[split_name]:
                s["fm"] = fm
                s["path"] = str(Path(feat_root) / fm / s["filename"])
    rglobal = [s for d in rsplits.values() for s in d["test"]]
    return rsplits, rglobal


def partition_sha(splits: dict, assignment: Dict[str, str]) -> str:
    """Stable SHA over (client -> sorted filenames per split) + assignment."""
    payload = {}
    for cid in sorted(splits):
        payload[cid] = {
            sp: sorted(s["filename"] for s in splits[cid][sp])
            for sp in ("train", "val", "test")
        }
    blob = json.dumps({"splits": payload,
                       "assignment": dict(sorted(assignment.items()))},
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def save_assignment(assignment: Dict[str, str], scheme: str, seed: int,
                    sha: str, out_dir: str = None) -> str:
    out_dir = out_dir or str(Path(__file__).parent / "assignments")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(out_dir) / f"{scheme.replace(':', '_')}_seed{seed}.json"
    out.write_text(json.dumps(
        {"scheme": scheme, "seed": seed, "sha": sha,
         "assignment": dict(sorted(assignment.items()))}, indent=2))
    return str(out)
