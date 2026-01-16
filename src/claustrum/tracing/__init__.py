"""Execution trace collection and prediction for CLAUSTRUM.

This module provides Trex-style micro-trace collection and execution semantics
learning, which provides the strongest semantic grounding for function embeddings.

Key Components:
    - MicroTraceCollector: Collects execution traces via emulation
    - TracePredictor: Model head for trace prediction task
    - TraceMaskingStrategy: Masking strategy for trace tokens

Micro-traces capture dynamic execution state including:
    - Register values after each instruction
    - Memory read/write values
    - Flags/status register state

Example:
    >>> from claustrum.tracing import MicroTraceCollector, TracePredictor
    >>> collector = MicroTraceCollector(isa="x86_64")
    >>> traces = collector.collect_traces(function_bytes, num_traces=10)
    >>> # Train with trace prediction objective
    >>> predictor = TracePredictor(config)
    >>> loss = predictor(hidden_states, trace_labels)
"""

from claustrum.tracing.collector import (
    MicroTraceCollector,
    ExecutionTrace,
    TracePoint,
    RegisterState,
    MemoryAccess,
)
from claustrum.tracing.predictor import (
    TracePredictor,
    TracePredictionHead,
    TraceTokenizer,
)
from claustrum.tracing.masking import (
    TraceMaskingStrategy,
    create_trace_masks,
)

__all__ = [
    # Collector
    "MicroTraceCollector",
    "ExecutionTrace",
    "TracePoint",
    "RegisterState",
    "MemoryAccess",
    # Predictor
    "TracePredictor",
    "TracePredictionHead",
    "TraceTokenizer",
    # Masking
    "TraceMaskingStrategy",
    "create_trace_masks",
]
