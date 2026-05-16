"""hetfm.projector — per-FM-type-tied projector bank (locked decision #6).

Default (tie='fm'): exactly one projector per FM architecture (3 total),
shared by all sites holding that FM and FedAvg-aggregated within the FM group.
Ablation (tie='site'): one projector per client, never aggregated (fully local).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class Projector(nn.Module):
    """d_in -> k.  depth: 'linear' | '1hidden' | '2hidden'."""

    def __init__(self, d_in: int, k: int, depth: str = "1hidden"):
        super().__init__()
        if depth == "linear":
            self.net = nn.Linear(d_in, k)
        elif depth == "1hidden":
            h = max(k, 512)
            self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(),
                                     nn.Linear(h, k))
        elif depth == "2hidden":
            h = max(k, 512)
            self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(),
                                     nn.Linear(h, h), nn.ReLU(),
                                     nn.Linear(h, k))
        else:
            raise ValueError(f"depth {depth!r}")

    def forward(self, x):
        return self.net(x)


class ProjectorBank(nn.Module):
    """Holds projectors keyed by FM (tie='fm') or by client_id (tie='site').

    fm_dims: {fm_name: d_in}.  client_fm: {client_id: fm_name}.
    """

    def __init__(self, fm_dims: Dict[str, int], client_fm: Dict[str, str],
                 k: int = 256, depth: str = "1hidden", tie: str = "fm"):
        super().__init__()
        self.tie = tie
        self.client_fm = dict(client_fm)
        self.k = k
        self._mods = nn.ModuleDict()
        if tie == "fm":
            for fm, d in fm_dims.items():
                self._mods[fm] = Projector(d, k, depth)
        elif tie == "site":
            for cid, fm in client_fm.items():
                self._mods[self._safe(cid)] = Projector(fm_dims[fm], k, depth)
        else:
            raise ValueError(f"tie {tie!r}")

    @staticmethod
    def _safe(cid: str) -> str:                     # ModuleDict key constraints
        return cid.replace(".", "_").replace("-", "_")

    def key_for(self, client_id: str) -> str:
        return self.client_fm[client_id] if self.tie == "fm" \
            else self._safe(client_id)

    def get(self, client_id: str) -> Projector:
        return self._mods[self.key_for(client_id)]

    def groups(self) -> Dict[str, list]:
        """key -> list of client_ids sharing that projector (for FedAvg)."""
        g: Dict[str, list] = {}
        for cid in self.client_fm:
            g.setdefault(self.key_for(cid), []).append(cid)
        return g

    def state_for(self, key: str) -> dict:
        return {k: v.detach().clone() for k, v in
                self._mods[key].state_dict().items()}

    def load_for(self, key: str, sd: dict):
        self._mods[key].load_state_dict(sd)
