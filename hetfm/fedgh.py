"""hetfm.fedgh — FedGH baseline (Yi, Wang, Liu, Shi, Yu, "FedGH: Heterogeneous
Federated Learning with Generalized Global Header", ACM MM 2023,
arXiv:2303.13137), adapted to frozen pathology foundation models.

FedGH: every client keeps a heterogeneous feature extractor and a header of a
common shape.  Each round the server broadcasts the global header; clients
train extractor + header locally, then upload only per-class average
representations; the server trains the generalized global header on the
uploaded (representation, label) pairs and broadcasts it again.  FedGH thus
provides a genuine global classifier and is well defined when every site holds a
single class, because the server sees class means from all sites.

Adaptation to this testbed (documented for the paper's fairness note):
  * extractor = frozen FM + trainable projector d_fm -> k (the only trainable
    part of the extractor); header = linear k -> C, identical to the protocol's
    shared head.
  * aggregate_projector=False -> FAITHFUL: per-site projectors, never averaged.
  * aggregate_projector=True  -> TIED: projectors FedAvg-averaged within the
    FM group (mirrors the charitable FedProtoKD variant and the protocol's own
    per-FM tying); the header is still learnt only at the server from class
    means, never FedAvg-averaged, which is what distinguishes FedGH from the
    protocol (no prototype anchoring, no head averaging).
  * class means are uploaded only for classes with >= min_count slides at the
    site (the same gate the protocol uses), so both methods see the same set of
    contributions.
  * global evaluation: each test client's own projector + the global header.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm.proto_anchor import (C, Head, _avg_state, _cache_by_client,   # noqa: E402
                                _macro_metrics)
from hetfm.projector import ProjectorBank                               # noqa: E402
from hetfm.r2_trainer import _eval_head, _metrics, _n_params            # noqa: E402


def train(splits: dict, global_test: list, client_fm: Dict[str, str],
          fm_dims: Dict[str, int], *, k=256, depth="linear", rounds=150,
          local_epochs=1, lr=3e-4, header_epochs=5, header_lr=1e-3,
          min_count=8, aggregate_projector=False, device=None, seed=42,
          log_every=150):
    t0 = time.time()
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    tie = "fm" if aggregate_projector else "site"
    bank = ProjectorBank(fm_dims, client_fm, k=k, depth=depth, tie=tie).to(device)
    ghead = Head(k).to(device)
    opt_h = torch.optim.Adam(ghead.parameters(), lr=header_lr)
    client_ids = [c for c in splits if splits[c]["train"]]
    train_cache = _cache_by_client(
        {c: splits[c]["train"] for c in client_ids}, device)
    tbc: Dict[str, list] = {}
    for s in global_test:
        tbc.setdefault(s["client_id"], []).append(s)
    test_cache = _cache_by_client(tbc, device)
    history = []
    n_up = []

    for rnd in range(1, rounds + 1):
        proj_states: Dict[str, list] = {}
        proj_w: Dict[str, list] = {}
        reps, labs = [], []
        for cid in client_ids:
            cached = train_cache.get(cid)
            if cached is None:
                continue
            X, y = cached
            proj = copy.deepcopy(bank.get(cid)).to(device)
            h = copy.deepcopy(ghead).to(device)          # local copy of global header
            opt = torch.optim.Adam(list(proj.parameters()) + list(h.parameters()),
                                   lr=lr)
            proj.train(); h.train()
            for _ in range(local_epochs):
                opt.zero_grad()
                loss = F.cross_entropy(h(proj(X)), y)
                loss.backward()
                opt.step()
            key = bank.key_for(cid)
            proj_states.setdefault(key, []).append(
                {kk: v.detach().clone() for kk, v in proj.state_dict().items()})
            proj_w.setdefault(key, []).append(X.size(0))
            with torch.no_grad():
                z = proj(X)
            for c in y.unique():
                m = (y == c)
                if int(m.sum()) >= min_count:
                    reps.append(z[m].mean(0)); labs.append(int(c))
        for key, states in proj_states.items():       # site: write-back; fm: FedAvg
            bank.load_for(key, _avg_state(states, proj_w[key]))
        # ---- server: train the generalized global header on uploaded class means
        n_up.append(len(reps))
        if reps:
            R = torch.stack(reps).detach(); L = torch.tensor(labs, device=device)
            ghead.train()
            for _ in range(header_epochs):
                opt_h.zero_grad()
                lh = F.cross_entropy(ghead(R), L)
                lh.backward()
                opt_h.step()
        if rnd % log_every == 0 or rnd == rounds:
            yt, yp = _eval_head(bank, ghead, test_cache)
            acc, f1 = _macro_metrics(yt, yp)
            history.append({"round": rnd, "macro_acc": acc, "macro_f1": f1})
            print(f"  [fedgh/{tie} r{rnd}] macro_acc={acc:.4f} macro_f1={f1:.4f}",
                  flush=True)

    yt, yp = _eval_head(bank, ghead, test_cache)
    acc, f1, rec = _metrics(yt, yp)
    cross = {}
    if not aggregate_projector:
        # Degeneracy probes for per-site extractors under single-class sites.
        # A valid global classifier is a function of (model, feature) only, not
        # of the site; with one class per site any site-dependent function can
        # score perfectly on its own site.  So each test client's slides are
        # also routed through the projector of ANOTHER site holding the same
        # foundation model, then classified by the same global header:
        #   random_partner : seeded derangement within the model group
        #                    (class-agnostic; the site-agnostic score)
        #   other_class    : the next site in the group whose class differs
        #                    (strict probe; a collapsed projector predicts the
        #                    partner's class and falls to zero)
        rng = np.random.default_rng(seed)
        by_fm = {}
        lab = {cid: int(splits[cid]["train"][0]["cancer_label"])
               for cid in splits if splits[cid]["train"]}
        for cid in sorted(bank.client_fm):
            if cid in lab:
                by_fm.setdefault(bank.client_fm[cid], []).append(cid)
        rand_p, other_p = {}, {}
        for fm, cids in by_fm.items():
            perm = list(rng.permutation(len(cids)))
            for i, cid in enumerate(cids):
                j = perm[i]
                if cids[j] == cid:
                    j = (j + 1) % len(cids)
                rand_p[cid] = cids[j]
                for step in range(1, len(cids)):
                    cand = cids[(i + step) % len(cids)]
                    if lab[cand] != lab[cid]:
                        other_p[cid] = cand
                        break
        bank.eval(); ghead.eval()

        def _routed(partner):
            yt_, yp_ = [], []
            with torch.no_grad():
                for cid, (X, y) in test_cache.items():
                    if cid not in partner:
                        continue
                    z = bank.get(partner[cid])(X)
                    yt_.extend(y.cpu().tolist())
                    yp_.extend(ghead(z).argmax(1).cpu().tolist())
            return _metrics(np.array(yt_), np.array(yp_))[0] if yt_ else None

        cross = {"random_partner_acc": _routed(rand_p),
                 "random_partner_same_class_frac": float(np.mean(
                     [lab[rand_p[c]] == lab[c] for c in rand_p])),
                 "other_class_partner_acc": _routed(other_p)}
    out = {"macro_acc": acc, "macro_f1": f1, "per_class_recall": rec,
           "cross_site_probe": cross,
           "history": history, "rounds": rounds, "k": k, "depth": depth,
           "tie": tie, "header_epochs": header_epochs, "header_lr": header_lr,
           "min_count": min_count, "seed": seed, "device": str(device),
           "mode": "fedgh_tied" if aggregate_projector else "fedgh_faithful",
           "uploaded_means_per_round_mean": float(np.mean(n_up)),
           "n_params": {"projectors_total": _n_params(bank),
                        "n_projectors": len(bank._mods),
                        "header": _n_params(ghead)},
           "timing": {"total_s": round(time.time() - t0, 2)}}
    if str(device).startswith("cuda"):
        out["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return out
