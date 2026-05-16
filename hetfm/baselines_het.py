"""hetfm.baselines_het — RQ3 alignment-necessity controls (Week-4).

Question (RQ3, proposal §3): is explicit prototype anchoring necessary, or
do dimension-matching / classical-alignment + FedAvg suffice?

Three controls, all on the SAME routed heterogeneous splits as the method,
all FedAvg of a head on frozen mean-pooled features (apples-to-apples with
proto_anchor.homogeneous_fedavg / train):

  zero_pad_fedavg   : pad every FM embedding to max dim (20480) with zeros,
                      FedAvg one shared MLP head. Trivial dim-matching control.
  local_pca_fedavg  : each FM gets its OWN PCA->k fitted on that FM's train
                      features only (no cross-FM/cross-site PCA), then FedAvg
                      a shared k->C head. Unsupervised-alignment control —
                      PCA bases are arbitrary per FM, so a shared head over
                      inconsistently-rotated k-spaces should underperform.
  procrustes_fedavg : PCA->k per FM, then an Orthogonal Procrustes rotation
                      per non-reference FM fitted to match per-class-mean
                      ANCHORS (a small public anchor set) of the reference FM,
                      then FedAvg a shared head. Classical linear alignment.

All reuse proto_anchor helpers so metrics/optimiser/rounds are identical to
the method -> the only thing that varies is the alignment operator.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from hetfm.proto_anchor import (                               # noqa: E402
    C, _MLPHead, _avg_state, _cache_by_client, _macro_metrics)

MAX_DIM = 20480  # Virchow2 — the zero-pad target


def _client_ids(splits: dict) -> List[str]:
    return [c for c in splits if splits[c]["train"]]


def _fm_of(splits: dict, cid: str) -> str:
    return splits[cid]["train"][0]["fm"]


def _fedavg_head(train_cache: Dict[str, tuple], test_cache: Dict[str, tuple],
                 d_in: int, *, rounds, lr, local_epochs, device, seed,
                 log_every, tag, transform=None):
    """Standard FedAvg of an MLP head. `transform` (cid,X)->X' is applied to
    cached features once (frozen) before training/eval."""
    torch.manual_seed(seed)
    g = _MLPHead(d_in).to(device)

    def _xf(cid, X):
        return X if transform is None else transform(cid, X)

    tr = {c: (_xf(c, X), y) for c, (X, y) in train_cache.items()}
    te = {c: (_xf(c, X), y) for c, (X, y) in test_cache.items()}
    history = []
    for rnd in range(1, rounds + 1):
        states, ws = [], []
        for cid, (X, y) in tr.items():
            m = copy.deepcopy(g).to(device)
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            m.train()
            for _ in range(local_epochs):
                opt.zero_grad()
                loss = F.cross_entropy(m(X), y)
                loss.backward()
                opt.step()
            states.append({k: v.detach().clone()
                           for k, v in m.state_dict().items()})
            ws.append(X.size(0))
        if states:
            g.load_state_dict(_avg_state(states, ws))
        if rnd % log_every == 0 or rnd == rounds:
            g.eval()
            with torch.no_grad():
                yt, yp = [], []
                for cid, (X, y) in te.items():
                    yp.extend(g(X).argmax(1).cpu().tolist())
                    yt.extend(y.cpu().tolist())
            acc, f1 = _macro_metrics(np.array(yt), np.array(yp))
            history.append({"round": rnd, "macro_acc": acc, "macro_f1": f1})
            print(f"  [{tag} r{rnd}] macro_acc={acc:.4f}")
    return {"macro_acc": acc, "macro_f1": f1, "history": history,
            "rounds": rounds, "mode": tag}


def _caches(splits: dict, global_test: list, device):
    cids = _client_ids(splits)
    train_cache = _cache_by_client(
        {c: splits[c]["train"] for c in cids}, device)
    tbc: Dict[str, list] = {}
    for s in global_test:
        tbc.setdefault(s["client_id"], []).append(s)
    test_cache = _cache_by_client(tbc, device)
    fm_of = {c: _fm_of(splits, c) for c in cids}
    # test clients use their own FM (same routed partition)
    for c in test_cache:
        if c not in fm_of and c in splits and splits[c]["test"]:
            fm_of[c] = splits[c]["test"][0]["fm"]
    return train_cache, test_cache, fm_of


# --------------------------------------------------------------------------- #
def zero_pad_fedavg(splits, global_test, *, rounds=150, lr=3e-4,
                    local_epochs=1, device=None, seed=42, log_every=50):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr, te, _ = _caches(splits, global_test, device)

    def pad(_cid, X):
        if X.size(1) >= MAX_DIM:
            return X[:, :MAX_DIM]
        return F.pad(X, (0, MAX_DIM - X.size(1)))

    return _fedavg_head(tr, te, MAX_DIM, rounds=rounds, lr=lr,
                        local_epochs=local_epochs, device=device, seed=seed,
                        log_every=log_every, tag="zeropad", transform=pad)


def _fit_pca_per_fm(tr, fm_of, k, device):
    """Per-FM PCA basis from that FM's TRAIN features only (no leakage,
    no cross-FM mixing). Returns {fm: (mean[d], V[d,k])}."""
    by_fm: Dict[str, list] = {}
    for cid, (X, _y) in tr.items():
        by_fm.setdefault(fm_of[cid], []).append(X)
    bases = {}
    for fm, xs in by_fm.items():
        M = torch.cat(xs, 0).to(device)
        mu = M.mean(0, keepdim=True)
        Mc = M - mu
        # economy SVD: right singular vecs = principal directions
        _, _, Vh = torch.linalg.svd(Mc, full_matrices=False)
        kk = min(k, Vh.size(0))
        V = Vh[:kk].t().contiguous()                  # [d, k]
        bases[fm] = (mu, V)
    return bases


def local_pca_fedavg(splits, global_test, *, k=256, rounds=150, lr=3e-4,
                     local_epochs=1, device=None, seed=42, log_every=50):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr, te, fm_of = _caches(splits, global_test, device)
    bases = _fit_pca_per_fm(tr, fm_of, k, device)

    def proj(cid, X):
        mu, V = bases[fm_of[cid]]
        out = (X - mu) @ V
        if out.size(1) < k:                           # pad short FMs to k
            out = F.pad(out, (0, k - out.size(1)))
        return out

    return _fedavg_head(tr, te, k, rounds=rounds, lr=lr,
                        local_epochs=local_epochs, device=device, seed=seed,
                        log_every=log_every, tag="localpca", transform=proj)


def procrustes_fedavg(splits, global_test, *, k=256, ref_fm="UNI_v2",
                      rounds=150, lr=3e-4, local_epochs=1, device=None,
                      seed=42, log_every=50):
    """PCA->k per FM, then an Orthogonal Procrustes rotation R_fm aligning
    each FM's per-class-mean anchors to the reference FM's, then FedAvg.
    Anchors = per-class means over TRAIN features of each FM (the 'small
    public anchor set' of class-mean pairs in proposal §6.2)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr, te, fm_of = _caches(splits, global_test, device)
    bases = _fit_pca_per_fm(tr, fm_of, k, device)

    def _pca(cid_fm, X):
        mu, V = bases[cid_fm]
        out = (X - mu) @ V
        if out.size(1) < k:
            out = F.pad(out, (0, k - out.size(1)))
        return out

    # per-FM class-mean anchors in PCA-k space (train only)
    anch_sum: Dict[str, torch.Tensor] = {}
    anch_cnt: Dict[str, torch.Tensor] = {}
    for cid, (X, y) in tr.items():
        fm = fm_of[cid]
        Z = _pca(fm, X)
        if fm not in anch_sum:
            anch_sum[fm] = torch.zeros(C, k, device=device)
            anch_cnt[fm] = torch.zeros(C, device=device)
        for c in range(C):
            m = (y == c)
            if int(m.sum()):
                anch_sum[fm][c] += Z[m].sum(0)
                anch_cnt[fm][c] += int(m.sum())
    anchors = {fm: anch_sum[fm] / anch_cnt[fm].clamp(min=1)[:, None]
               for fm in anch_sum}
    if ref_fm not in anchors:
        ref_fm = next(iter(anchors))
    A_ref = anchors[ref_fm]
    R: Dict[str, torch.Tensor] = {}
    for fm, A_fm in anchors.items():
        if fm == ref_fm:
            R[fm] = torch.eye(k, device=device)
            continue
        common = (anch_cnt[fm] > 0) & (anch_cnt[ref_fm] > 0)
        M = A_fm[common].t() @ A_ref[common]          # [k,k]
        U, _, Vh = torch.linalg.svd(M)
        R[fm] = (U @ Vh)                              # nearest orthogonal map

    def proj(cid, X):
        fm = fm_of[cid]
        return _pca(fm, X) @ R[fm]

    return _fedavg_head(tr, te, k, rounds=rounds, lr=lr,
                        local_epochs=local_epochs, device=device, seed=seed,
                        log_every=log_every, tag="procrustes", transform=proj)


BASELINES = {
    "zeropad": zero_pad_fedavg,
    "localpca": local_pca_fedavg,
    "procrustes": procrustes_fedavg,
}
