"""Inference module for CLAUSTRUM.

Provides production inference capabilities:
- ClaustrumEmbedder: End-to-end embedding generation
- FAISS index for similarity search
- ONNX export for optimized inference
- Batch embedding generation
"""

from claustrum.inference.embedder import ClaustrumEmbedder
from claustrum.inference.index import (
    FAISSIndex,
    build_ivf_pq_index,
    build_hnsw_index,
)
from claustrum.inference.export import (
    export_to_onnx,
    quantize_model,
    ONNXInferenceSession,
)

__all__ = [
    "ClaustrumEmbedder",
    "FAISSIndex",
    "build_ivf_pq_index",
    "build_hnsw_index",
    "export_to_onnx",
    "quantize_model",
    "ONNXInferenceSession",
]
