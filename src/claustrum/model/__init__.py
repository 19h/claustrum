"""Model architecture module for CLAUSTRUM.

Implements the hierarchical transformer architecture:
- 12-layer BERT-style encoder for instruction sequences
- 3-layer Graph Attention Network for CFG aggregation
- Attention pooling for function-level embeddings
- Support for pretraining tasks (MIM, CWP, DUP)
"""

from claustrum.model.config import ClaustrumConfig
from claustrum.model.encoder import ClaustrumEncoder
from claustrum.model.gnn import CFGAttentionNetwork
from claustrum.model.pooling import AttentionPooling
from claustrum.model.pretraining import (
    MaskedInstructionModel,
    ContextWindowPredictor,
    DefUsePredictor,
    PretrainingModel,
)

__all__ = [
    "ClaustrumConfig",
    "ClaustrumEncoder",
    "CFGAttentionNetwork",
    "AttentionPooling",
    "MaskedInstructionModel",
    "ContextWindowPredictor",
    "DefUsePredictor",
    "PretrainingModel",
]
