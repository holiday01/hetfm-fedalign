"""hetfm.r2_trainer — prototype-anchored trainer extended for the single-backend
batch (September 2026).  proto_anchor.py is left untouched.

With head_mode="shared" the optimisation path is identical to
proto_anchor.train (same op order), so the headline configuration reproduces
the submitted numbers bit-for-bit on the same backend.  Extensions:

  head_mode  'shared' : FedAvg-averaged classifier head (as submitted)
             'local'  : one head per site, never averaged; global inference by
                        nearest global prototype (a per-site-routed local-head
                        score is also returned, as a degeneracy artefact)
             'none'   : no head at all; anchors only; nearest-prototype classifier
  diagnostics computed once at the end on the global test set: class silhouette,
             cross-FM prototype cosine agreement (per class and mean), and an
             FM-identity linear probe (balanced accuracy, 5-fold CV)
  extra outputs: macro-F1, per-class recall, nearest-prototype accuracy for every
             mode, wall-clock seconds, peak GPU memory, trainable-parameter counts
"""
from __future__ import annotations

import copy
import time
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm.proto_anchor import (C, Head, _avg_state, _cache_by_client,   # noqa: E402
                                _macro_metrics)
from hetfm.projector import ProjectorBank                               # noqa: E402


class _LocalHeads(nn.Module):
    def __init__(self, client_ids: List[str], k: int):
        super().__init__()
        self.m = nn.ModuleDict({self._s(c): nn.Linear(k, C) for c in client_ids})

    @staticmethod
    def _s(c):
        return c.replace(".", "_").replace("-", "_")

    def get(self, c):
        return self.m[self._s(c)]


def _n_params(mod: nn.Module) -> int:
    return int(sum(p.numel() for p in mod.parameters()))


@torch.no_grad()
def _collect(bank: ProjectorBank, test_cache: Dict[str, tuple]):
    """Projected test embeddings, labels, FM/group key, per client."""
    bank.eval()
    Z, Y, FM, CID = [], [], [], []
    for cid, (X, y) in test_cache.items():
        if cid not in bank.client_fm:
            continue
        z = bank.get(cid)(X)
        Z.append(z); Y.append(y)
        FM += [bank.client_fm[cid]] * len(z); CID += [cid] * len(z)
    return torch.cat(Z), torch.cat(Y), np.array(FM), np.array(CID)


def _metrics(yt: np.ndarray, yp: np.ndarray):
    acc, f1 = _macro_metrics(yt, yp)
    rec = []
    for c in range(C):
        m = yt == c
        rec.append(float((yp[m] == c).mean()) if m.sum() else float("nan"))
    return acc, f1, rec


@torch.no_grad()
def _eval_head(bank, head, test_cache):
    Z, Y, _, _ = _collect(bank, test_cache)
    head.eval()
    return Y.cpu().numpy(), head(Z).argmax(1).cpu().numpy()


@torch.no_grad()
def _eval_proto(bank, prototypes, test_cache):
    Z, Y, _, _ = _collect(bank, test_cache)
    mc = F.normalize(prototypes, dim=1)
    zc = F.normalize(Z, dim=1)
    return Y.cpu().numpy(), (zc @ mc.t()).argmax(1).cpu().numpy()


@torch.no_grad()
def _eval_local_routed(bank, lheads, test_cache):
    bank.eval(); lheads.eval()
    yt, yp = [], []
    for cid, (X, y) in test_cache.items():
        if cid not in bank.client_fm:
            continue
        pred = lheads.get(cid)(bank.get(cid)(X)).argmax(1)
        yt.extend(y.cpu().tolist()); yp.extend(pred.cpu().tolist())
    return np.array(yt), np.array(yp)


@torch.no_grad()
def diagnostics(bank: ProjectorBank, test_cache: Dict[str, tuple],
                max_n: int = 3000) -> dict:
    Z, Y, FM, _ = _collect(bank, test_cache)
    Z = Z.cpu().numpy(); Y = Y.cpu().numpy()
    n = len(Z)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, max_n, replace=False) if n > max_n else np.arange(n)
    out = {"n_test": int(n), "n_fm_groups": int(len(set(FM)))}
    try:
        from sklearn.metrics import silhouette_score
        out["silhouette_class"] = float(silhouette_score(Z[idx], Y[idx]))
    except Exception as e:                                       # noqa: BLE001
        out["silhouette_class"] = float("nan"); out["silhouette_err"] = str(e)
    fms = sorted(set(FM))
    per_class = {}
    for c in range(C):
        means = []
        for f in fms:
            m = (Y == c) & (FM == f)
            if m.sum() >= 5:
                v = Z[m].mean(0)
                means.append(v / (np.linalg.norm(v) + 1e-9))
        if len(means) >= 2:
            per_class[c] = float(np.mean([a @ b for a, b in combinations(means, 2)]))
    out["xfm_prototype_cosine_per_class"] = per_class
    out["xfm_prototype_cosine"] = (float(np.mean(list(per_class.values())))
                                   if per_class else float("nan"))
    out["n_classes_with_xfm_cosine"] = len(per_class)
    # FM-identity linear probe on the shared latent space
    if len(fms) >= 2:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            from sklearn.metrics import balanced_accuracy_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            Xp = Z[idx]; fp = FM[idx]
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000,
                                                   class_weight="balanced"))
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            pred = cross_val_predict(clf, Xp, fp, cv=cv)
            out["fm_probe_balanced_acc"] = float(balanced_accuracy_score(fp, pred))
            out["fm_probe_chance"] = float(1.0 / len(fms))
        except Exception as e:                                   # noqa: BLE001
            out["fm_probe_balanced_acc"] = float("nan"); out["fm_probe_err"] = str(e)
    return out


def train_r2(splits: dict, global_test: list, client_fm: Dict[str, str],
             fm_dims: Dict[str, int], *, k=256, depth="linear", tie="fm",
             rounds=150, local_epochs=1, lr=3e-4, lam_proto=1.0, lam_con=1.0,
             tau=0.2, rho=0.5, min_count=8, warmup=5, head_mode="shared",
             device=None, seed=42, log_every=150, run_diagnostics=True):
    assert head_mode in ("shared", "local", "none")
    t0 = time.time()
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    bank = ProjectorBank(fm_dims, client_fm, k=k, depth=depth, tie=tie).to(device)
    client_ids = [c for c in splits if splits[c]["train"]]
    head = Head(k).to(device) if head_mode == "shared" else None
    lheads = _LocalHeads(client_ids, k).to(device) if head_mode == "local" else None
    prototypes = torch.zeros(C, k, device=device)
    proto_seen = torch.zeros(C, device=device)

    train_cache = _cache_by_client(
        {c: splits[c]["train"] for c in client_ids}, device)
    test_by_client: Dict[str, list] = {}
    for s in global_test:
        test_by_client.setdefault(s["client_id"], []).append(s)
    test_cache = _cache_by_client(test_by_client, device)
    t_load = time.time() - t0
    history = []

    def _primary():
        if head_mode == "shared":
            return _eval_head(bank, head, test_cache)
        return _eval_proto(bank, prototypes, test_cache)

    t_train0 = time.time()
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
            params = list(proj.parameters())
            if head_mode == "shared":
                h = copy.deepcopy(head).to(device)
                params += list(h.parameters()); h.train()
            elif head_mode == "local":
                h = lheads.get(cid)
                params += list(h.parameters()); h.train()
            else:
                h = None
            opt = torch.optim.Adam(params, lr=lr)
            proj.train()
            for _ in range(local_epochs):
                opt.zero_grad()
                z = proj(X)
                loss = None
                if h is not None:
                    loss = F.cross_entropy(h(z), y)
                if anchor_on:
                    zc = F.normalize(z, dim=1)
                    mc = F.normalize(prototypes, dim=1)
                    cos_y = (zc * mc[y]).sum(1)
                    t_p = lam_proto * (1.0 - cos_y).mean()
                    loss = t_p if loss is None else loss + t_p
                    sim = zc @ mc.t() / tau
                    loss = loss + lam_con * F.cross_entropy(sim, y)
                if loss is None:
                    break
                loss.backward()
                opt.step()

            n = X.size(0)
            key = bank.key_for(cid)
            proj_updates.setdefault(key, []).append(
                {kk: v.detach().clone() for kk, v in proj.state_dict().items()})
            proj_w.setdefault(key, []).append(n)
            if head_mode == "shared":
                head_states.append({kk: v.detach().clone()
                                    for kk, v in h.state_dict().items()})
                head_w.append(n)
            with torch.no_grad():
                z = proj(X)
            for c in y.unique():
                m = (y == c)
                if int(m.sum()) >= min_count:
                    cls_sum[c] += z[m].sum(0)
                    cls_cnt[c] += int(m.sum())

        if head_states:
            head.load_state_dict(_avg_state(head_states, head_w))
        for key, ups in proj_updates.items():
            bank.load_for(key, _avg_state(ups, proj_w[key]))
        new_mu = torch.where(cls_cnt[:, None] > 0,
                             cls_sum / cls_cnt.clamp(min=1)[:, None],
                             prototypes)
        upd = cls_cnt > 0
        prototypes[upd] = rho * prototypes[upd] + (1 - rho) * new_mu[upd]
        proto_seen[upd] = 1.0

        if rnd % log_every == 0 or rnd == rounds:
            yt, yp = _primary()
            acc, f1 = _macro_metrics(yt, yp)
            history.append({"round": rnd, "macro_acc": acc, "macro_f1": f1})
            print(f"  [r2/{head_mode} r{rnd}] macro_acc={acc:.4f} macro_f1={f1:.4f}",
                  flush=True)
    t_train = time.time() - t_train0

    yt, yp = _primary()
    acc, f1, rec = _metrics(yt, yp)
    out = {"macro_acc": acc, "macro_f1": f1, "per_class_recall": rec,
           "primary_eval": "head" if head_mode == "shared" else "prototype",
           "history": history, "rounds": rounds, "k": k, "depth": depth,
           "tie": tie, "head_mode": head_mode, "lam_proto": lam_proto,
           "lam_con": lam_con, "min_count": min_count, "seed": seed,
           "device": str(device)}
    ytp, ypp = _eval_proto(bank, prototypes, test_cache)
    out["acc_proto"], out["f1_proto"], _ = _metrics(ytp, ypp)
    if head_mode == "shared":
        out["acc_head"], out["f1_head"] = acc, f1
    if head_mode == "local":
        ytl, ypl = _eval_local_routed(bank, lheads, test_cache)
        out["acc_local_routed_artefact"], _, _ = _metrics(ytl, ypl)
    if run_diagnostics:
        t_d = time.time()
        out["diag"] = diagnostics(bank, test_cache)
        out["diag"]["seconds"] = round(time.time() - t_d, 2)
    out["n_params"] = {
        "projectors": {key: _n_params(bank._mods[key]) for key in bank._mods},
        "projectors_total": _n_params(bank),
        "head": (_n_params(head) if head is not None else
                 (_n_params(lheads) if lheads is not None else 0)),
        "prototypes_not_trainable": int(C * k)}
    out["timing"] = {"load_s": round(t_load, 2), "train_s": round(t_train, 2),
                     "per_round_s": round(t_train / rounds, 3),
                     "total_s": round(time.time() - t0, 2)}
    if str(device).startswith("cuda"):
        out["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return out
