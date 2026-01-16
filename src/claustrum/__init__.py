"""
CLAUSTRUM: Cross-ISA Semantic Code Embedding System

A comprehensive binary analysis platform that creates architecture-neutral embeddings
for binary functions, enabling cross-architecture similarity search, semantic clustering,
and vulnerability detection.

Key Components:
    - lifting: Multi-backend IR lifting (VEX, ESIL, P-Code)
    - normalization: Architecture-neutral IR canonicalization
    - tokenization: IR tokenization with shared vocabulary
    - model: Hierarchical transformer + GNN architecture
    - training: Pretraining and contrastive fine-tuning
    - evaluation: Retrieval and clustering metrics
    - inference: Production embedding generation
    - data: Dataset loading and preprocessing
    - tracing: Execution trace collection and prediction

Example:
    >>> from claustrum import Claustrum
    >>> embedder = Claustrum.from_pretrained("claustrum-base")
    >>> embedding = embedder.embed_function(binary_path, function_addr)
    >>> similar = embedder.search(embedding, k=10)
"""

__version__ = "0.1.0"
__author__ = "CLAUSTRUM Team"

from claustrum.lifting import IRLifter, VEXLifter, ESILLifter
from claustrum.normalization import IRNormalizer, NormalizedIR
from claustrum.tokenization import IRTokenizer
from claustrum.model import ClaustrumEncoder, ClaustrumConfig
from claustrum.inference import ClaustrumEmbedder


# Data loading (lazy import to avoid circular dependencies)
def _import_data():
    from claustrum.data import (
        BinaryFunctionDataset,
        CrossISADataset,
        PretrainingDataset,
        ContrastiveCollator,
        PretrainingCollator,
        create_train_dataloader,
        create_eval_dataloader,
    )

    return {
        "BinaryFunctionDataset": BinaryFunctionDataset,
        "CrossISADataset": CrossISADataset,
        "PretrainingDataset": PretrainingDataset,
        "ContrastiveCollator": ContrastiveCollator,
        "PretrainingCollator": PretrainingCollator,
        "create_train_dataloader": create_train_dataloader,
        "create_eval_dataloader": create_eval_dataloader,
    }


# Tracing (lazy import due to optional dependencies)
def _import_tracing():
    from claustrum.tracing import (
        MicroTraceCollector,
        ExecutionTrace,
        TracePredictor,
        TraceTokenizer,
    )

    return {
        "MicroTraceCollector": MicroTraceCollector,
        "ExecutionTrace": ExecutionTrace,
        "TracePredictor": TracePredictor,
        "TraceTokenizer": TraceTokenizer,
    }


__all__ = [
    # Lifting
    "IRLifter",
    "VEXLifter",
    "ESILLifter",
    # Normalization
    "IRNormalizer",
    "NormalizedIR",
    # Tokenization
    "IRTokenizer",
    # Model
    "ClaustrumEncoder",
    "ClaustrumConfig",
    # Inference
    "ClaustrumEmbedder",
    # Data (available via import)
    "BinaryFunctionDataset",
    "CrossISADataset",
    "PretrainingDataset",
    "ContrastiveCollator",
    "PretrainingCollator",
    "create_train_dataloader",
    "create_eval_dataloader",
    # Tracing (available via import)
    "MicroTraceCollector",
    "ExecutionTrace",
    "TracePredictor",
    "TraceTokenizer",
]


def __getattr__(name):
    """Lazy import for data and tracing modules."""
    if name in (
        "BinaryFunctionDataset",
        "CrossISADataset",
        "PretrainingDataset",
        "ContrastiveCollator",
        "PretrainingCollator",
        "create_train_dataloader",
        "create_eval_dataloader",
    ):
        return _import_data()[name]
    elif name in (
        "MicroTraceCollector",
        "ExecutionTrace",
        "TracePredictor",
        "TraceTokenizer",
    ):
        return _import_tracing()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
