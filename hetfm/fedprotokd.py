"""hetfm.fedprotokd — faithful-LIGHT FedProtoKD baseline (Week-5, locked #4).

FedProtoKD (IJCAI 2025) is model-heterogeneous FL via prototype knowledge
distillation. Original: small trainable encoders co-adapted, prototypes +
KD as the cross-client bridge, NO weight averaging of the heterogeneous
models.

Adaptation to THIS setting (documented for the paper's fairness note):
encoders are frozen pathology FMs, so the heterogeneous trainable "model"
per site is its (projector + head); prototypes live in the shared k-space.
Kept faithful to FedProtoKD's defining choices:
  * per-site projector AND head are LOCAL — never FedAvg-aggregated
    (the model-heterogeneous premise; this is the contrast vs our method,
     which DOES FedAvg per-FM-group projectors + a shared head).
  * the ONLY exchanged object is class prototypes (server: count-weighted
    global means, EMA).
  * cross-site knowledge transfers via KD: local head logits are distilled
    toward a prototype-similarity teacher + a prototype-alignment pull.

Eval: no single global model exists -> route each test sample through ITS
OWN site's projector+head (same as proto_anchor._evaluate's per-client
routing), which is the honest way to score an un-aggregated federation.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm.proto_anchor import (C, _avg_state, _cache_by_client,    # noqa: E402
                                _macro_metrics)
from hetfm.projector import ProjectorBank                            # noqa: E402


class _Heads(nn.Module):
    """One local k->C head per client (never aggregated)."""

    def __init__(self, client_fm: Dict[str, str], k: int):
        super().__init__()
        self.m = nn.ModuleDict()
        for cid in client_fm:
            self.m[self._s(cid)] = nn.Linear(k, C)

    @staticmethod
    def _s(cid):
        return cid.replace(".", "_").replace("-", "_")

    def get(self, cid):
        return self.m[self._s(cid)]


def train(splits: dict, global_test: list, client_fm: Dict[str, str],
          fm_dims: Dict[str, int], *, k=256, depth="linear", rounds=150,
          local_epochs=1, lr=3e-4, lam_proto=1.0, lam_kd=1.0, tau=2.0,
          rho=0.5, min_count=8, warmup=5, aggregate_projector=False,
          device=None, seed=42, log_every=50):
    """FedProtoKD-light.

    aggregate_projector=False -> VARIANT A (paper-default, FAITHFUL):
      per-site projector AND head, neither aggregated; prototypes only.
      Under class≈site coupling there is NO non-degenerate global
      classifier — a per-site-routed eval is trivially inflated. We
      EXPOSE this: return `degeneracy` = {head_selfeval, proto_persite}
      (both artifacts, NOT leaderboard numbers).
    aggregate_projector=True -> VARIANT B (best-effort tuned):
      relax ONLY the projector to per-FM-tied FedAvg (shared, sees all
      classes via other sites) -> nearest-global-prototype eval is
      non-degenerate. Heads stay per-site & prototype-KD kept -> still
      FedProtoKD-flavoured, not our method (which FedAvgs a SHARED head).
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tie = "fm" if aggregate_projector else "site"
    bank = ProjectorBank(fm_dims, client_fm, k=k, depth=depth,
                          tie=tie).to(device)
    heads = _Heads(client_fm, k).to(device)               # always per-site
    prototypes = torch.zeros(C, k, device=device)
    proto_seen = torch.zeros(C, device=device)
    client_ids = [c for c in splits if splits[c]["train"]]

    train_cache = _cache_by_client(
        {c: splits[c]["train"] for c in client_ids}, device)
    tbc: Dict[str, list] = {}
    for s in global_test:
        tbc.setdefault(s["client_id"], []).append(s)
    test_cache = _cache_by_client(tbc, device)
    history = []

    for rnd in range(1, rounds + 1):
        kd_on = (rnd > warmup) and (float(proto_seen.sum()) >= C)
        cls_sum = torch.zeros(C, k, device=device)
        cls_cnt = torch.zeros(C, device=device)
        proj_states: Dict[str, list] = {}
        proj_w: Dict[str, list] = {}

        for cid in client_ids:
            cached = train_cache.get(cid)
            if cached is None:
                continue
            X, y = cached
            proj = copy.deepcopy(bank.get(cid)).to(device)
            head = heads.get(cid)                          # local, in-place
            opt = torch.optim.Adam(list(proj.parameters())
                                   + list(head.parameters()), lr=lr)
            proj.train(); head.train()
            for _ in range(local_epochs):
                opt.zero_grad()
                z = proj(X)
                logits = head(z)
                loss = F.cross_entropy(logits, y)
                if kd_on:
                    zc = F.normalize(z, dim=1)
                    mc = F.normalize(prototypes, dim=1)
                    loss = loss + lam_proto * (
                        1.0 - (zc * mc[y]).sum(1)).mean()
                    teacher = (zc @ mc.t()) / tau            # [N,C]
                    kd = F.kl_div(
                        F.log_softmax(logits / tau, dim=1),
                        F.softmax(teacher.detach(), dim=1),
                        reduction="batchmean") * (tau * tau)
                    loss = loss + lam_kd * kd
                loss.backward()
                opt.step()

            key = bank.key_for(cid)
            proj_states.setdefault(key, []).append(
                {kk: v.detach().clone()
                 for kk, v in proj.state_dict().items()})
            proj_w.setdefault(key, []).append(X.size(0))
            with torch.no_grad():
                z = proj(X)
            for c in y.unique():
                m = (y == c)
                if int(m.sum()) >= min_count:
                    cls_sum[c] += z[m].sum(0)
                    cls_cnt[c] += int(m.sum())

        # projector: per-FM-group FedAvg (B) or just write back each site (A)
        for key, states in proj_states.items():
            bank.load_for(key, _avg_state(states, proj_w[key]))
        # server: prototypes (count-wtd, EMA) — the FedProtoKD bridge
        new_mu = torch.where(cls_cnt[:, None] > 0,
                             cls_sum / cls_cnt.clamp(min=1)[:, None],
                             prototypes)
        upd = cls_cnt > 0
        prototypes[upd] = rho * prototypes[upd] + (1 - rho) * new_mu[upd]
        proto_seen[upd] = 1.0

        if rnd % log_every == 0 or rnd == rounds:
            acc, f1 = _eval(bank, prototypes, test_cache)
            history.append({"round": rnd, "macro_acc": acc, "macro_f1": f1})
            print(f"  [fedprotokd/{tie} r{rnd}] macro_acc={acc:.4f}",
                  flush=True)

    acc, f1 = _eval(bank, prototypes, test_cache)
    out = {"macro_acc": acc, "macro_f1": f1, "history": history,
           "rounds": rounds,
           "mode": ("fedprotokd_B_tuned_aggProj" if aggregate_projector
                    else "fedprotokd_A_faithful")}
    if not aggregate_projector:
        # VARIANT A: expose the degeneracy explicitly. Neither number is a
        # leaderboard score — both are per-site-routed artifacts under the
        # 107/107 single-class-per-site partition.
        out["degeneracy"] = {
            "head_selfeval_per_site": round(
                _eval_head_selfeval(bank, heads, test_cache), 4),
            "proto_per_site_routed": round(acc, 4),
            "note": ("Under 107/107 single-class-per-site there is no valid "
                     "global classifier: per-site-routed nearest-prototype "
                     "is ~0.91-0.94 — near-constant across schemes/seeds and "
                     "0.3-0.6 ABOVE the same method's non-degenerate eval "
                     "(variant B) — a pure routing artifact; meanwhile the "
                     "per-site head itself is weak (~0.40, KD destabilises "
                     "it). Faithful FedProtoKD has NO usable global "
                     "classifier here. Qualitative finding; the fair number "
                     "is variant B (aggregate_projector=True)."),
        }
        out["macro_acc"] = None     # refuse to emit a leaderboard number
        out["macro_f1"] = None
    return out


@torch.no_grad()
def _eval_head_selfeval(bank: ProjectorBank, heads: _Heads, test_cache):
    """Each per-site head scoring ITS OWN single-class test -> ~1.0 by
    construction. Demonstrates the un-aggregated degeneracy."""
    bank.eval()
    yt, yp = [], []
    for cid, (X, y) in test_cache.items():
        if cid not in bank.client_fm:
            continue
        pred = heads.get(cid)(bank.get(cid)(X)).argmax(1)
        yt.extend(y.cpu().tolist()); yp.extend(pred.cpu().tolist())
    if not yt:
        return 0.0
    return _macro_metrics(np.array(yt), np.array(yp))[0]


@torch.no_grad()
def _eval(bank: ProjectorBank, prototypes, test_cache):
    """Global 9-class inference = NEAREST GLOBAL PROTOTYPE (cosine) on each
    test client's own local projector embedding. The per-site local head is
    single-class-overfit under class≈site partition (un-aggregated) and is
    meaningless for the global task — prototypes are FedProtoKD's actual
    transferable classifier, so scoring by them is the faithful choice."""
    bank.eval()
    mc = F.normalize(prototypes, dim=1)
    yt, yp = [], []
    for cid, (X, y) in test_cache.items():
        if cid not in bank.client_fm:
            continue
        zc = F.normalize(bank.get(cid)(X), dim=1)
        pred = (zc @ mc.t()).argmax(1)
        yt.extend(y.cpu().tolist()); yp.extend(pred.cpu().tolist())
    if not yt:
        return 0.0, 0.0
    return _macro_metrics(np.array(yt), np.array(yp))
