"""Poker44 chunk bot-detection — v2 feature-based pipeline.

A from-scratch, order-invariant, scale-robust design (see
``poker_chunk_model_design_approach.md``):

    raw JSON -> schema normalizer -> hand/chunk feature extractor
             -> LightGBM chunk classifier -> isotonic calibration
             -> per-chunk bot probability (+ drift report, top features)

The model is intentionally tabular-first: on a modest labeled set the
gradient-boosted tree on aggregate behavioral features is the most robust core,
and every feature is permutation-invariant across hands (hand order is never
trusted) and scale-robust (ratios/buckets/entropy, not raw stakes).
"""

from .schema import Chunk, load_chunks
from .features import chunk_feature_vector, feature_names_for

__all__ = ["Chunk", "load_chunks", "chunk_feature_vector", "feature_names_for"]
