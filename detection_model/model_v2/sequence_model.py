"""TCN hand encoder + attention-pool chunk aggregator (optional ensemble member).

The tabular LightGBM flattens each hand into aggregate features and loses the
*within-hand action order*. This model recovers exactly that signal and nothing
that depends on hand order:

    actions (ordered by action_id) ──▶ TCN ──▶ attention pool ──▶ hand embedding
    hands (a bag)                  ──▶ attention pool (no position) ──▶ chunk emb
    chunk embedding                ──▶ MLP ──▶ P(bot | chunk)

* **Within a hand**: a small dilated 1D-conv (TCN) captures local action n-grams
  (bet→raise→call motifs) with few parameters — a strong small-data fit.
* **Across hands**: a masked attention pool with **no positional encoding**, so
  the aggregator is permutation-invariant (shuffling hands cannot change the
  chunk embedding). Hands are an exchangeable bag — never a sequence.

sklearn-style ``fit(chunks, y)`` / ``predict_proba(chunks)`` so it stacks with the
LightGBM. Picklable via state_dict. CPU-only by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for the TCN sequence model.") from exc

from .features import ACTION_TYPES, STREET_ORDER, _amount_bucket, _f
from .schema import _sort_actions

_ATYPE = {a: i + 1 for i, a in enumerate(ACTION_TYPES)}   # 0 = pad/unknown
_STREET = {s: i + 1 for i, s in enumerate(STREET_ORDER)}  # 0 = pad/unknown
_N_ATYPE = len(_ATYPE) + 1
_N_STREET = len(_STREET) + 1
_N_BUCKET = 10           # _amount_bucket returns 0..8
_CONT_DIM = 2            # log1p(amount_bb), log1p(pot_after_bb)


# ------------------------------------------------------------------ encoding

def _encode_hand(hand: Dict[str, Any], max_actions: int) -> Dict[str, np.ndarray]:
    meta = hand.get("metadata") or {}
    hero = int(meta.get("hero_seat") or 0)
    actions = (hand.get("actions") or [])[:max_actions]
    cat = np.zeros((max_actions, 4), dtype=np.int64)
    cont = np.zeros((max_actions, _CONT_DIM), dtype=np.float32)
    mask = np.zeros(max_actions, dtype=np.bool_)
    for i, a in enumerate(actions):
        if not isinstance(a, dict):
            continue
        at = str(a.get("action_type") or "").lower().strip()
        st = str(a.get("street") or "").lower().strip()
        actor = int(a.get("actor_seat") or -1)
        amt = max(0.0, _f(a.get("normalized_amount_bb"), 0.0))
        cat[i, 0] = _ATYPE.get(at, 0)
        cat[i, 1] = _STREET.get(st, 0)
        cat[i, 2] = 1 if (hero and actor == hero) else 0
        cat[i, 3] = min(_amount_bucket(amt), _N_BUCKET - 1)
        cont[i, 0] = math.log1p(amt)
        cont[i, 1] = math.log1p(max(0.0, _f(a.get("pot_after"), 0.0)))
        mask[i] = True
    return {"cat": cat, "cont": cont, "mask": mask}


def _subsample(n: int, cap: int) -> List[int]:
    if n <= cap:
        return list(range(n))
    # even span over the bag (order-invariant: which hands, not their order)
    return sorted({int(round(i * (n - 1) / (cap - 1))) for i in range(cap)})[:cap]


def encode_chunk(hands: List[Dict[str, Any]], max_hands: int, max_actions: int) -> Dict[str, np.ndarray]:
    hands = [_sort_actions(h) for h in hands if isinstance(h, dict)]
    idx = _subsample(len(hands), max_hands)
    cat = np.zeros((max_hands, max_actions, 4), dtype=np.int64)
    cont = np.zeros((max_hands, max_actions, _CONT_DIM), dtype=np.float32)
    amask = np.zeros((max_hands, max_actions), dtype=np.bool_)
    hmask = np.zeros(max_hands, dtype=np.bool_)
    for j, si in enumerate(idx):
        enc = _encode_hand(hands[si], max_actions)
        cat[j], cont[j], amask[j] = enc["cat"], enc["cont"], enc["mask"]
        hmask[j] = bool(enc["mask"].any())
    return {"cat": cat, "cont": cont, "amask": amask, "hmask": hmask}


# --------------------------------------------------------------------- model

class _AttnPool(nn.Module):
    """Masked, permutation-invariant attention pool (single learned query)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x [B, T, d], mask [B, T] (True = valid)
        scores = (self.proj(x) @ self.q.transpose(1, 2)).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (w * x).sum(dim=1)


class _TCNBlock(nn.Module):
    def __init__(self, d: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(d, d, kernel, padding=pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, amask: torch.Tensor) -> torch.Tensor:
        # x [B, d, T], amask [B, T]
        h = self.drop(self.act(self.norm(self.conv(x))))
        h = h * amask.unsqueeze(1)          # zero padded positions
        return x + h                         # residual


@dataclass
class SeqConfig:
    d_model: int = 48
    kernel: int = 3
    dilations: tuple = (1, 2)
    dropout: float = 0.1
    max_hands: int = 40
    max_actions: int = 14

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["dilations"] = list(self.dilations)
        return d


class _TCNChunkNet(nn.Module):
    def __init__(self, cfg: SeqConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.atype = nn.Embedding(_N_ATYPE, d, padding_idx=0)
        self.street = nn.Embedding(_N_STREET, d, padding_idx=0)
        self.hero = nn.Embedding(2, d)
        self.bucket = nn.Embedding(_N_BUCKET, d, padding_idx=0)
        self.cont = nn.Linear(_CONT_DIM, d)
        self.in_norm = nn.LayerNorm(d)
        self.tcn = nn.ModuleList([_TCNBlock(d, cfg.kernel, dil, cfg.dropout) for dil in cfg.dilations])
        self.hand_pool = _AttnPool(d)
        self.chunk_pool = _AttnPool(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, 1))

    def forward(self, cat, cont, amask, hmask) -> torch.Tensor:
        b, h, a, _ = cat.shape
        x = (
            self.atype(cat[..., 0]) + self.street(cat[..., 1])
            + self.hero(cat[..., 2]) + self.bucket(cat[..., 3]) + self.cont(cont)
        )
        x = self.in_norm(x).reshape(b * h, a, -1).transpose(1, 2)   # [B*H, d, A]
        m = amask.reshape(b * h, a)
        for blk in self.tcn:
            x = blk(x, m)
        x = x.transpose(1, 2)                                       # [B*H, A, d]
        safe = m.clone(); safe[~m.any(dim=1), 0] = True             # avoid all-masked softmax
        hand_emb = self.hand_pool(x, safe).reshape(b, h, -1)       # [B, H, d]
        hmask_safe = hmask.clone(); hmask_safe[~hmask.any(dim=1), 0] = True
        chunk_emb = self.chunk_pool(hand_emb, hmask_safe)          # [B, d]
        return self.head(chunk_emb).squeeze(-1)


class _DS(Dataset):
    def __init__(self, encoded, y=None):
        self.enc = encoded
        self.y = y

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        e = self.enc[i]
        label = float(self.y[i]) if self.y is not None else 0.0
        return e, label


def _collate(batch):
    keys = ("cat", "cont", "amask", "hmask")
    out = {}
    for k in keys:
        arr = np.stack([b[0][k] for b in batch])
        out[k] = torch.from_numpy(arr)
    out["y"] = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return out


@dataclass
class TCNSequenceModel:
    """sklearn-style wrapper: fit(chunks, y) / predict_proba(chunks)."""

    config: SeqConfig = field(default_factory=SeqConfig)
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.15
    patience: int = 4
    seed: int = 44
    device: str = "cpu"
    verbose: bool = False
    _state: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def _encode(self, chunks):
        return [encode_chunk(c, self.config.max_hands, self.config.max_actions) for c in chunks]

    def fit(self, chunks: Sequence[Sequence[Dict[str, Any]]], y: Sequence[int]) -> "TCNSequenceModel":
        torch.manual_seed(self.seed)
        y = np.asarray(y, dtype=np.float32)
        enc = self._encode(chunks)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(enc))
        nval = max(1, int(round(self.val_fraction * len(enc))))
        vi, ti = order[:nval], order[nval:]
        tl = DataLoader(_DS([enc[i] for i in ti], y[ti]), batch_size=self.batch_size, shuffle=True, collate_fn=_collate)
        vl = DataLoader(_DS([enc[i] for i in vi], y[vi]), batch_size=self.batch_size, shuffle=False, collate_fn=_collate)

        pos = float(y[ti].sum()); neg = float((y[ti] == 0).sum())
        pw = torch.tensor([neg / max(pos, 1.0)], device=self.device)
        model = _TCNChunkNet(self.config).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        best, best_state, bad = float("inf"), None, 0
        for ep in range(self.epochs):
            model.train()
            for bt in tl:
                logit = model(bt["cat"].to(self.device), bt["cont"].to(self.device),
                              bt["amask"].to(self.device), bt["hmask"].to(self.device))
                loss = loss_fn(logit, bt["y"].to(self.device))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            vloss = self._val_loss(model, vl, loss_fn)
            if self.verbose:
                print(f"    tcn epoch {ep+1}/{self.epochs} val_loss={vloss:.4f}")
            if vloss + 1e-5 < best:
                best, bad = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._state = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "config": self.config.to_dict()}
        return self

    def _val_loss(self, model, loader, loss_fn) -> float:
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for bt in loader:
                logit = model(bt["cat"].to(self.device), bt["cont"].to(self.device),
                              bt["amask"].to(self.device), bt["hmask"].to(self.device))
                tot += float(loss_fn(logit, bt["y"].to(self.device))) * len(bt["y"]); n += len(bt["y"])
        return tot / max(n, 1)

    def _build(self) -> _TCNChunkNet:
        cfg = SeqConfig(**{**self._state["config"], "dilations": tuple(self._state["config"]["dilations"])})
        m = _TCNChunkNet(cfg).to(self.device)
        m.load_state_dict(self._state["state_dict"]); m.eval()
        return m

    def predict_proba(self, chunks: Sequence[Sequence[Dict[str, Any]]]) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("predict_proba called before fit.")
        enc = self._encode(chunks)
        model = self._build()
        loader = DataLoader(_DS(enc), batch_size=self.batch_size, shuffle=False, collate_fn=_collate)
        logits: List[float] = []
        with torch.no_grad():
            for bt in loader:
                logit = model(bt["cat"].to(self.device), bt["cont"].to(self.device),
                              bt["amask"].to(self.device), bt["hmask"].to(self.device))
                logits.extend(logit.cpu().tolist())
        p = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits), -40, 40)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict_chunk_scores(self, chunks) -> List[float]:
        return self.predict_proba(chunks)[:, 1].tolist()
