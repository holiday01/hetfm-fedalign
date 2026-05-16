"""hetfm.proto_anchor — prototype-anchored heterogeneous-FM federated training.

Per round: each client trains a copy of its FM-type projector + the shared
head on L_i = CE + lambda_proto*||z-mu_y||^2 + lambda_con*proto-contrastive.
Server: FedAvg the shared head over all clients; FedAvg each projector ONLY
within its FM group (well-defined: same input dim); update prototypes as
count-weighted class means with EMA decay rho and a min-count gate m.

Reuses fl_agent.federated_learning.load_features_batch (mean-pooled slide
features), identical to the Week-1 local-only lower bound -> comparable.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from fl_agent.federated_learning import load_features_batch  # noqa: E402
from hetfm.projector import ProjectorBank                     # noqa: E402

C = 9  # cancer classes


class Head(nn.Module):
    def __init__(self, k: int, n_classes: int = C):
        super().__init__()
        self.fc = nn.Linear(k, n_classes)

    def forward(self, z):
        return self.fc(z)


def _avg_state(states: List[dict], weights: List[float]) -> dict:
    tot = float(sum(weights))
    out = {}
    for key in states[0]:
        out[key] = sum(s[key] * (w / tot) for s, w in zip(states, weights))
    return out


def _cache_by_client(items_by_client: Dict[str, list], device) -> Dict[str, tuple]:
    """Load (X,y) ONCE per client and keep on device. Features are frozen
    (frozen FMs) so reloading them every round was pure wasted I/O — this
    cuts ~1M np.load calls in a 150-round run to ~1 pass."""
    cache = {}
    for cid, samples in items_by_client.items():
        if not samples:
            continue
        X, y = load_features_batch(samples, device=device)
        if X is not None:
            cache[cid] = (X, y)
    return cache


def train(splits: dict, global_test: list, client_fm: Dict[str, str],
          fm_dims: Dict[str, int], *, k=256, depth="1hidden", tie="fm",
          rounds=40, local_epochs=1, lr=3e-4, lam_proto=0.5, lam_con=0.5,
          tau=0.2, rho=0.5, min_count=8, warmup=5, device=None, seed=42,
          log_every=10, per_class=False, return_internals=False):
    """Returns dict with global-test macro acc/F1 + history. Real training.
    per_class=True also adds 'per_class_recall' (length-C list, NaN for a
    class absent from the global test) for the pre-registered per-cancer
    H2 analysis. Backward-compatible: default off, training path unchanged."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    bank = ProjectorBank(fm_dims, client_fm, k=k, depth=depth, tie=tie).to(device)
    head = Head(k).to(device)
    prototypes = torch.zeros(C, k, device=device)
    proto_seen = torch.zeros(C, device=device)
    client_ids = [c for c in splits if splits[c]["train"]]

    # ---- load every feature ONCE (frozen) -> cache on device --------------
    train_cache = _cache_by_client(
        {c: splits[c]["train"] for c in client_ids}, device)
    test_by_client: Dict[str, list] = {}
    for s in global_test:
        test_by_client.setdefault(s["client_id"], []).append(s)
    test_cache = _cache_by_client(test_by_client, device)
    history = []

    for rnd in range(1, rounds + 1):
        anchor_on = (rnd > warmup) and (float(proto_seen.sum()) >= C)
        head_states, head_w = [], []
        proj_updates: Dict[str, list] = {}
        proj_w: Dict[str, list] = {}
        cls_sum = torch.zeros(C, k, device=device)
        cls_cnt = torch.zeros(C, device=device)

        for cid in client_ids:
            cached = train_cache.get(cid)
            if cached is None:
                continue
            X, y = cached
            proj = copy.deepcopy(bank.get(cid)).to(device)
            h = copy.deepcopy(head).to(device)
            opt = torch.optim.Adam(list(proj.parameters()) +
                                   list(h.parameters()), lr=lr)
            proj.train(); h.train()
            for _ in range(local_epochs):
                opt.zero_grad()
                z = proj(X)
                logits = h(z)
                loss = F.cross_entropy(logits, y)
                # Anchor terms activate only once every class prototype has
                # been observed (proto_seen all 1) AND past warmup -> avoids
                # cold-start zero-prototype noise. Both terms are BOUNDED
                # (cosine-based), so they cannot dwarf CE at lambda~0.5.
                if anchor_on:
                    zc = F.normalize(z, dim=1)
                    mc = F.normalize(prototypes, dim=1)
                    # (b) prototype match: 1 - cos(z, mu_y)  in [0, 2]
                    cos_y = (zc * mc[y]).sum(1)
                    loss = loss + lam_proto * (1.0 - cos_y).mean()
                    # (c) prototype-contrastive over cosine logits
                    sim = zc @ mc.t() / tau            # [N, C]
                    loss = loss + lam_con * F.cross_entropy(sim, y)
                loss.backward()
                opt.step()

            n = X.size(0)
            key = bank.key_for(cid)
            proj_updates.setdefault(key, []).append(
                {kk: v.detach().clone() for kk, v in
                 proj.state_dict().items()})
            proj_w.setdefault(key, []).append(n)
            head_states.append({kk: v.detach().clone()
                                for kk, v in h.state_dict().items()})
            head_w.append(n)
            # class-mean projected embeddings (post-update), min-count gated
            with torch.no_grad():
                z = proj(X)
            for c in y.unique():
                m = (y == c)
                if int(m.sum()) >= min_count:
                    cls_sum[c] += z[m].sum(0)
                    cls_cnt[c] += int(m.sum())

        # ---- server aggregation ----
        if head_states:
            head.load_state_dict(_avg_state(head_states, head_w))
        for key, ups in proj_updates.items():          # per-FM-group FedAvg
            bank.load_for(key, _avg_state(ups, proj_w[key]))
        new_mu = torch.where(cls_cnt[:, None] > 0,
                             cls_sum / cls_cnt.clamp(min=1)[:, None],
                             prototypes)
        upd = cls_cnt > 0
        prototypes[upd] = rho * prototypes[upd] + (1 - rho) * new_mu[upd]
        proto_seen[upd] = 1.0

        if rnd % log_every == 0 or rnd == rounds:
            acc, f1 = _evaluate(bank, head, test_cache)
            history.append({"round": rnd, "macro_acc": acc, "macro_f1": f1})
            print(f"  [hetfm r{rnd}] macro_acc={acc:.4f} macro_f1={f1:.4f}")

    acc, f1 = _evaluate(bank, head, test_cache)
    out = {"macro_acc": acc, "macro_f1": f1, "history": history,
           "rounds": rounds, "k": k, "tie": tie}
    if per_class:
        out["per_class_recall"] = _per_class_recall(bank, head, test_cache)
    if return_internals:
        out["_internals"] = {"bank": bank, "head": head,
                             "prototypes": prototypes,
                             "test_cache": test_cache}
    return out


@torch.no_grad()
def _per_class_recall(bank: ProjectorBank, head: Head,
                      test_cache: Dict[str, tuple]):
    """Length-C per-class recall on the global test (NaN if a class is
    absent). Same routing as _evaluate; used only for per-cancer H2."""
    bank.eval(); head.eval()
    yt, yp = [], []
    for cid, (X, y) in test_cache.items():
        if cid not in bank.client_fm:
            continue
        pred = head(bank.get(cid)(X)).argmax(1)
        yt.extend(y.cpu().tolist()); yp.extend(pred.cpu().tolist())
    yt, yp = np.array(yt), np.array(yp)
    out = []
    for c in range(C):
        m = yt == c
        out.append(float((yp[m] == c).mean()) if m.sum() else float("nan"))
    return out


def _macro_metrics(yt: np.ndarray, yp: np.ndarray):
    accs, f1s = [], []
    for c in range(C):
        m = yt == c
        if m.sum() == 0:
            continue
        accs.append((yp[m] == c).mean())
        tp = ((yp == c) & (yt == c)).sum()
        prec = tp / max((yp == c).sum(), 1)
        rec = tp / max(m.sum(), 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    if not accs:
        return 0.0, 0.0
    return float(np.mean(accs)), float(np.mean(f1s))


@torch.no_grad()
def _evaluate(bank: ProjectorBank, head: Head, test_cache: Dict[str, tuple]):
    """Route each client's cached test (X,y) through its FM projector."""
    bank.eval(); head.eval()
    yt, yp = [], []
    for cid, (X, y) in test_cache.items():
        if cid not in bank.client_fm:
            continue
        pred = head(bank.get(cid)(X)).argmax(1)
        yt.extend(y.cpu().tolist()); yp.extend(pred.cpu().tolist())
    if not yt:
        return 0.0, 0.0
    return _macro_metrics(np.array(yt), np.array(yp))


class _MLPHead(nn.Module):
    """FedFM-WSI-style 2-layer MLP classifier on raw single-FM features."""

    def __init__(self, d_in: int, n_classes: int = C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, n_classes))

    def forward(self, x):
        return self.net(x)


def homogeneous_fedavg(splits: dict, global_test: list, d_in: int, *,
                       rounds=150, local_epochs=1, lr=3e-4, device=None,
                       seed=42, log_every=25):
    """PROPER homogeneous upper bound: standard FedAvg of an MLP head on a
    single FM's frozen features (no projector, no prototypes) over the SAME
    canonical partition/splits — apples-to-apples with train(). This is the
    FedFM-WSI-style ceiling, not 'our method restricted to one FM'."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = _MLPHead(d_in).to(device)
    client_ids = [c for c in splits if splits[c]["train"]]
    train_cache = _cache_by_client(
        {c: splits[c]["train"] for c in client_ids}, device)
    tbc: Dict[str, list] = {}
    for s in global_test:
        tbc.setdefault(s["client_id"], []).append(s)
    test_cache = _cache_by_client(tbc, device)
    history = []
    for rnd in range(1, rounds + 1):
        states, ws = [], []
        for cid in client_ids:
            cached = train_cache.get(cid)
            if cached is None:
                continue
            X, y = cached
            m = copy.deepcopy(g).to(device)
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            m.train()
            for _ in range(local_epochs):
                opt.zero_grad()
                loss = F.cross_entropy(m(X), y)
                loss.backward()
                opt.step()
            states.append({kk: v.detach().clone()
                           for kk, v in m.state_dict().items()})
            ws.append(X.size(0))
        if states:
            g.load_state_dict(_avg_state(states, ws))
        if rnd % log_every == 0 or rnd == rounds:
            g.eval()
            with torch.no_grad():
                yt, yp = [], []
                for cid, (X, y) in test_cache.items():
                    pred = g(X).argmax(1)
                    yt.extend(y.cpu().tolist()); yp.extend(pred.cpu().tolist())
            acc, f1 = _macro_metrics(np.array(yt), np.array(yp))
            history.append({"round": rnd, "macro_acc": acc, "macro_f1": f1})
            print(f"  [homog-fedavg r{rnd}] macro_acc={acc:.4f}")
    return {"macro_acc": acc, "macro_f1": f1, "history": history,
            "rounds": rounds, "mode": "homogeneous_fedavg"}
