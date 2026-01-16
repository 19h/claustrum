"""Evaluation module for CLAUSTRUM.

Provides metrics for:
- Retrieval: MRR, Recall@K, mAP
- Clustering: ARI, NMI, Silhouette
"""

from claustrum.evaluation.metrics import (
    compute_mrr,
    compute_recall_at_k,
    compute_map,
    compute_ari,
    compute_nmi,
    compute_silhouette,
    EvaluationMetrics,
    evaluate_retrieval,
    evaluate_clustering,
)

__all__ = [
    "compute_mrr",
    "compute_recall_at_k",
    "compute_map",
    "compute_ari",
    "compute_nmi",
    "compute_silhouette",
    "EvaluationMetrics",
    "evaluate_retrieval",
    "evaluate_clustering",
]
