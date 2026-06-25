"""Hierarchical chunk classifier (schema v2).

Three tiers, mirroring how a chunk is structured:

    actions ──▶ (Transformer + attention pool) ──▶ hand embedding
    hands   ──▶ (chunk encoder + attention pool) ──▶ chunk embedding
    chunk   ──▶ MLP head ──▶ logit  (auxiliary probe used to train the encoder)

What v2 adds over v1:

* **Richer action tokens** — 6 categorical channels (street, action_type, seat,
  amount_bucket, pot_flow, first_in_street) instead of 3, plus the numeric block.
* **Per-hand meta fusion** — each hand embedding is augmented with a learned
  projection of behavioral context (stack depth, actor count, per-street action
  counts, hero engagement) and a deepest-street-reached embedding.
* **Pluggable chunk encoder** — ``chunk_encoder="transformer"`` (default,
  permutation-invariant set encoder, the best inductive fit for a homogeneous
  bag of hands) or ``"gru"`` (the original ordered bidirectional GRU). An optional
  soft hand-position channel keeps a light ordering signal either way.

The production score still comes from the XGBoost head on
``concat(extract_chunk_embedding(...), engineered_features)`` plus the
ScoreCalibrator; the MLP head here is the training probe that shapes the encoder.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SCHEMA_VERSION = 3
CHUNK_ENCODERS = ("transformer", "gru")


class _AttentionPool(nn.Module):
    """Single learned-query attention pool (Set Transformer PMA, k=1)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch = x.size(0)
        query = self.query.expand(batch, 1, -1)
        all_padded = key_padding_mask.all(dim=1) if key_padding_mask is not None else None
        safe_mask = key_padding_mask
        if safe_mask is not None and all_padded is not None and all_padded.any():
            # Avoid a fully-masked softmax (NaNs): let an empty row attend to slot 0.
            safe_mask = safe_mask.clone()
            safe_mask[all_padded, 0] = False
        pooled, _ = self.attn(query=query, key=x, value=x, key_padding_mask=safe_mask, need_weights=False)
        pooled = self.norm(pooled.squeeze(1))
        if all_padded is not None and all_padded.any():
            pooled = pooled.masked_fill(all_padded.unsqueeze(-1), 0.0)
        return pooled


class HierarchicalChunkClassifier(nn.Module):
    def __init__(
        self,
        street_vocab_size: int,
        action_type_vocab_size: int,
        seat_vocab_size: int,
        numeric_dim: int,
        feature_dim: int,
        *,
        amount_bucket_vocab_size: int,
        pot_flow_vocab_size: int,
        first_in_street_vocab_size: int,
        actor_role_vocab_size: int,
        street_position_vocab_size: int,
        hand_end_vocab_size: int,
        hand_meta_dim: int,
        max_actions_per_hand: int = 64,
        max_hands: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        chunk_layers: int = 1,
        chunk_encoder: str = "transformer",
        use_hand_position: bool = True,
        bidirectional_gru: bool = True,
        dropout: float = 0.30,
        pad_id: int = 0,
        **_: object,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads. Got {d_model} / {n_heads}")
        chunk_encoder = str(chunk_encoder).lower()
        if chunk_encoder not in CHUNK_ENCODERS:
            raise ValueError(f"chunk_encoder must be one of {CHUNK_ENCODERS}, got {chunk_encoder!r}")

        self.pad_id = int(pad_id)
        self.numeric_dim = int(numeric_dim)
        self.feature_dim = int(feature_dim)
        self.max_actions_per_hand = int(max_actions_per_hand)
        self.max_hands = int(max_hands)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.chunk_layers = int(chunk_layers)
        self.chunk_encoder = chunk_encoder
        self.use_hand_position = bool(use_hand_position)
        self.bidirectional_gru = bool(bidirectional_gru)
        self.dropout_p = float(dropout)

        self.config: Dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "street_vocab_size": int(street_vocab_size),
            "action_type_vocab_size": int(action_type_vocab_size),
            "seat_vocab_size": int(seat_vocab_size),
            "numeric_dim": self.numeric_dim,
            "feature_dim": self.feature_dim,
            "amount_bucket_vocab_size": int(amount_bucket_vocab_size),
            "pot_flow_vocab_size": int(pot_flow_vocab_size),
            "first_in_street_vocab_size": int(first_in_street_vocab_size),
            "actor_role_vocab_size": int(actor_role_vocab_size),
            "street_position_vocab_size": int(street_position_vocab_size),
            "hand_end_vocab_size": int(hand_end_vocab_size),
            "hand_meta_dim": int(hand_meta_dim),
            "max_actions_per_hand": self.max_actions_per_hand,
            "max_hands": self.max_hands,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "chunk_layers": self.chunk_layers,
            "chunk_encoder": self.chunk_encoder,
            "use_hand_position": self.use_hand_position,
            "bidirectional_gru": self.bidirectional_gru,
            "dropout": self.dropout_p,
            "pad_id": self.pad_id,
        }

        # --- action-level categorical + numeric embeddings ---
        self.street_embedding = nn.Embedding(street_vocab_size, d_model, padding_idx=pad_id)
        self.action_type_embedding = nn.Embedding(action_type_vocab_size, d_model, padding_idx=pad_id)
        self.seat_embedding = nn.Embedding(seat_vocab_size, d_model, padding_idx=pad_id)
        self.amount_bucket_embedding = nn.Embedding(amount_bucket_vocab_size, d_model, padding_idx=pad_id)
        self.pot_flow_embedding = nn.Embedding(pot_flow_vocab_size, d_model, padding_idx=pad_id)
        self.first_in_street_embedding = nn.Embedding(first_in_street_vocab_size, d_model, padding_idx=pad_id)
        self.actor_role_embedding = nn.Embedding(actor_role_vocab_size, d_model, padding_idx=pad_id)
        self.street_position_embedding = nn.Embedding(street_position_vocab_size, d_model, padding_idx=pad_id)

        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.action_position_embedding = nn.Embedding(max_actions_per_hand, d_model)
        self.action_input_norm = nn.LayerNorm(d_model)
        self.action_input_dropout = nn.Dropout(dropout)

        hand_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.hand_encoder = nn.TransformerEncoder(hand_layer, num_layers=n_layers)
        self.hand_pool = _AttentionPool(d_model, n_heads, dropout)
        self.hand_norm = nn.LayerNorm(d_model)

        # --- per-hand meta fusion ---
        self.hand_end_embedding = nn.Embedding(hand_end_vocab_size, d_model, padding_idx=pad_id)
        self.hand_meta_projection = nn.Sequential(
            nn.Linear(hand_meta_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.hand_meta_norm = nn.LayerNorm(d_model)

        # --- optional soft ordering signal over hands ---
        if self.use_hand_position:
            self.position_projection = nn.Sequential(
                nn.Linear(1, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
            )

        # --- chunk-level encoder (pluggable) ---
        if self.chunk_encoder == "transformer":
            chunk_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.chunk_transformer = nn.TransformerEncoder(chunk_layer, num_layers=self.chunk_layers)
            self.gru_projection = nn.Identity()
        else:
            gru_hidden = max(1, d_model // 2) if self.bidirectional_gru else d_model
            self.hand_gru = nn.GRU(
                input_size=d_model, hidden_size=gru_hidden, num_layers=self.chunk_layers,
                dropout=dropout if self.chunk_layers > 1 else 0.0,
                batch_first=True, bidirectional=self.bidirectional_gru,
            )
            gru_out = gru_hidden * (2 if self.bidirectional_gru else 1)
            self.gru_projection = nn.Linear(gru_out, d_model) if gru_out != d_model else nn.Identity()
        self.chunk_pool = _AttentionPool(d_model, n_heads, dropout)
        self.chunk_norm = nn.LayerNorm(d_model)

        # --- MLP probe head (trains the encoder; production uses XGBoost+calibrator) ---
        if feature_dim > 0:
            self.feature_projection = nn.Sequential(
                nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
            )
            head_in = d_model * 2
        else:
            self.feature_projection = None
            head_in = d_model
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    # ---- action -> hand ----------------------------------------------------

    def _embed_actions(self, action_cat: torch.Tensor, action_num: torch.Tensor) -> torch.Tensor:
        # action_cat: [N, A, 8]
        street = action_cat[:, :, 0].clamp(0, self.street_embedding.num_embeddings - 1)
        atype = action_cat[:, :, 1].clamp(0, self.action_type_embedding.num_embeddings - 1)
        seat = action_cat[:, :, 2].clamp(0, self.seat_embedding.num_embeddings - 1)
        amount = action_cat[:, :, 3].clamp(0, self.amount_bucket_embedding.num_embeddings - 1)
        pot_flow = action_cat[:, :, 4].clamp(0, self.pot_flow_embedding.num_embeddings - 1)
        first = action_cat[:, :, 5].clamp(0, self.first_in_street_embedding.num_embeddings - 1)
        actor_role = action_cat[:, :, 6].clamp(0, self.actor_role_embedding.num_embeddings - 1)
        street_pos = action_cat[:, :, 7].clamp(0, self.street_position_embedding.num_embeddings - 1)

        positions = torch.arange(action_cat.shape[1], device=action_cat.device).unsqueeze(0)
        positions = positions.clamp(max=self.action_position_embedding.num_embeddings - 1)

        x = (
            self.street_embedding(street)
            + self.action_type_embedding(atype)
            + self.seat_embedding(seat)
            + self.amount_bucket_embedding(amount)
            + self.pot_flow_embedding(pot_flow)
            + self.first_in_street_embedding(first)
            + self.actor_role_embedding(actor_role)
            + self.street_position_embedding(street_pos)
            + self.numeric_projection(action_num)
            + self.action_position_embedding(positions)
        )
        return self.action_input_dropout(self.action_input_norm(x))

    def encode_hands(
        self, action_cat: torch.Tensor, action_num: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        b, h, a, cat_dim = action_cat.shape
        flat_cat = action_cat.reshape(b * h, a, cat_dim)
        flat_num = action_num.reshape(b * h, a, self.numeric_dim)
        flat_mask = action_mask.reshape(b * h, a)

        empty = ~flat_mask.any(dim=1)
        if empty.any():
            flat_mask = flat_mask.clone()
            flat_mask[empty, 0] = True  # keep the pooler stable for empty hands

        x = self._embed_actions(flat_cat, flat_num)
        encoded = self.hand_encoder(x, src_key_padding_mask=~flat_mask)
        pooled = self.hand_norm(self.hand_pool(encoded, key_padding_mask=~flat_mask))
        return pooled.reshape(b, h, self.d_model)

    def _fuse_hand_meta(
        self, hand_emb: torch.Tensor, hand_meta: torch.Tensor, hand_end: torch.Tensor, hand_mask: torch.Tensor
    ) -> torch.Tensor:
        hand_end = hand_end.clamp(0, self.hand_end_embedding.num_embeddings - 1)
        meta = self.hand_meta_projection(hand_meta) + self.hand_end_embedding(hand_end)
        fused = self.hand_meta_norm(hand_emb + meta)
        return fused.masked_fill(~hand_mask.unsqueeze(-1), 0.0)

    def _add_hand_position(self, hand_emb: torch.Tensor, hand_mask: torch.Tensor) -> torch.Tensor:
        if not self.use_hand_position:
            return hand_emb
        b, h, _ = hand_emb.shape
        idx = torch.arange(h, device=hand_emb.device).float()
        valid = hand_mask.sum(dim=1).float().clamp(min=1.0)
        rel = (idx.unsqueeze(0) / (valid.unsqueeze(1) - 1.0).clamp(min=1.0)) * hand_mask.float()
        return hand_emb + self.position_projection(rel.unsqueeze(-1))

    # ---- hand -> chunk -----------------------------------------------------

    def encode_chunk(self, hand_emb: torch.Tensor, hand_mask: torch.Tensor) -> torch.Tensor:
        b, h, _ = hand_emb.shape
        empty = ~hand_mask.any(dim=1)
        if empty.any():
            hand_mask = hand_mask.clone()
            hand_mask[empty, 0] = True

        if self.chunk_encoder == "transformer":
            contextual = self.chunk_transformer(hand_emb, src_key_padding_mask=~hand_mask)
        else:
            lengths = hand_mask.long().sum(dim=1).clamp(min=1)
            packed = pack_padded_sequence(
                hand_emb, lengths=lengths.detach().cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.hand_gru(packed)
            contextual, _ = pad_packed_sequence(out, batch_first=True, total_length=h)
            contextual = self.gru_projection(contextual)

        contextual = self.chunk_norm(contextual)
        return self.chunk_pool(contextual, key_padding_mask=~hand_mask)

    def extract_chunk_embedding(
        self,
        action_cat: torch.Tensor,
        action_num: torch.Tensor,
        action_mask: torch.Tensor,
        hand_mask: torch.Tensor,
        hand_meta: torch.Tensor,
        hand_end: torch.Tensor,
    ) -> torch.Tensor:
        hand_emb = self.encode_hands(action_cat, action_num, action_mask)
        hand_emb = self._fuse_hand_meta(hand_emb, hand_meta, hand_end, hand_mask)
        hand_emb = self._add_hand_position(hand_emb, hand_mask)
        return self.encode_chunk(hand_emb, hand_mask)

    def forward(
        self,
        action_cat: torch.Tensor,
        action_num: torch.Tensor,
        action_mask: torch.Tensor,
        hand_mask: torch.Tensor,
        features: torch.Tensor,
        hand_meta: torch.Tensor,
        hand_end: torch.Tensor,
    ) -> torch.Tensor:
        chunk_emb = self.extract_chunk_embedding(
            action_cat, action_num, action_mask, hand_mask, hand_meta, hand_end
        )
        if self.feature_projection is not None:
            probe_in = torch.cat([chunk_emb, self.feature_projection(features)], dim=-1)
        else:
            probe_in = chunk_emb
        return self.head(probe_in).squeeze(-1)
