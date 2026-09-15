"""hetfm.r2_localonly — local-only lower bound of the mixed federation at the
same seeds as every other arm.

Each site trains its own projector (d_fm -> k, same architecture as the
protocol's local step) and its own head on its own training slides only,
with no communication: full-batch Adam, `rounds` epochs (the protocol's
150 rounds x 1 local epoch of full-batch updates).  Every site's model is
then scored on the global test set, routed through that site's foundation
model, and the arm's macro accuracy is the mean over sites (the definition
of the Week-1 lower bound).
Also returned: the slide-weighted mean and the best single site.

  python -m hetfm.run_r2_batch --only localonly_balanced --seeds 62
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hetfm.proto_anchor import Head, _cache_by_client, _macro_metrics  # noqa: E402
from hetfm import assign                                             # noqa: E402


def train_local_only(splits: dict, global_test: list, client_fm: Dict[str, str],
                     fm_dims: Dict[str, int], *, k=256, rounds=150, lr=3e-4,
                     device=None, seed=42, feat_root=None, log_every=150):
    t0 = time.time()
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feat_root = feat_root or assign.FEAT_ROOT
    client_ids = [c for c in splits if splits[c]["train"]]
    train_cache = _cache_by_client({c: splits[c]["train"] for c in client_ids}, device)
    # the global test set routed through each foundation model once
    test_by_fm = {}
    for fm in sorted(set(client_fm[c] for c in client_ids)):
        samples = copy.deepcopy(global_test)
        for s in samples:
            s["fm"] = fm
            s["path"] = str(Path(feat_root) / fm / s["filename"])
        test_by_fm[fm] = _cache_by_client({"__test__": samples}, device)["__test__"]
    per_site, n_train = {}, {}
    for cid in client_ids:
        if cid not in train_cache:
            continue
        X, y = train_cache[cid]
        fm = client_fm[cid]
        model = nn.Sequential(nn.Linear(fm_dims[fm], k), Head(k)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(rounds):
            opt.zero_grad()
            F.cross_entropy(model(X), y).backward()
            opt.step()
        model.eval()
        Xt, yt = test_by_fm[fm]
        with torch.no_grad():
            yp = model(Xt).argmax(1).cpu().numpy()
        acc, f1 = _macro_metrics(yt.cpu().numpy(), yp)
        per_site[cid] = round(float(acc), 4)
        n_train[cid] = int(len(y))
    accs = np.array(list(per_site.values()))
    w = np.array([n_train[c] for c in per_site], dtype=float)
    return {"macro_acc": float(accs.mean()),
            "macro_acc_weighted": float((accs * w).sum() / w.sum()),
            "macro_acc_best_site": float(accs.max()),
            "n_sites": int(len(accs)), "per_site_macro_acc": per_site,
            "rounds": rounds, "k": k, "lr": lr, "seed": seed, "device": str(device),
            "mode": "local_only", "timing": {"total_s": round(time.time() - t0, 2)}}
