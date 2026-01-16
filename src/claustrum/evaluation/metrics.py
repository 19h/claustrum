"""Evaluation metrics for binary similarity and clustering.

Implements metrics from the plan:
- Retrieval: MRR, Recall@K, mAP
- Clustering: ARI, NMI, Silhouette

Target metrics for production:
- Cross-ISA Recall@1: >60% on 100K pool
- Cross-ISA MRR: >0.7 on 10K pool
- Semantic clustering ARI: >0.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    # Retrieval metrics
    mrr: float = 0.0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_50: float = 0.0
    map_score: float = 0.0

    # Clustering metrics
    ari: float = 0.0
    nmi: float = 0.0
    silhouette: float = 0.0

    # Cross-architecture specific
    cross_arch_recall_at_1: float = 0.0
    cross_arch_mrr: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mrr": self.mrr,
            "recall@1": self.recall_at_1,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "recall@50": self.recall_at_50,
            "mAP": self.map_score,
            "ari": self.ari,
            "nmi": self.nmi,
            "silhouette": self.silhouette,
            "cross_arch_recall@1": self.cross_arch_recall_at_1,
            "cross_arch_mrr": self.cross_arch_mrr,
        }


def compute_similarity_matrix(
    embeddings: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings: (N, D) embeddings
        normalize: Whether to L2 normalize

    Returns:
        (N, N) similarity matrix
    """
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)

    return np.dot(embeddings, embeddings.T)


def compute_mrr(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
    exclude_self: bool = True,
) -> float:
    """Compute Mean Reciprocal Rank.

    MRR measures the average reciprocal position of the first
    correct match in ranked results.

    Args:
        similarity_matrix: (N, N) pairwise similarities
        labels: (N,) ground truth labels
        exclude_self: Exclude self-similarity (diagonal)

    Returns:
        MRR score (0-1, higher is better)
    """
    n = len(labels)
    reciprocal_ranks = []

    for i in range(n):
        sims = similarity_matrix[i].copy()

        if exclude_self:
            sims[i] = -np.inf

        # Get sorted indices (descending similarity)
        sorted_indices = np.argsort(sims)[::-1]

        # Find rank of first correct match
        for rank, idx in enumerate(sorted_indices, 1):
            if labels[idx] == labels[i] and idx != i:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # No correct match found
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks))


def compute_recall_at_k(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
    k: int,
    exclude_self: bool = True,
) -> float:
    """Compute Recall@K.

    Measures the fraction of queries where at least one correct
    match appears in the top-K results.

    Args:
        similarity_matrix: (N, N) pairwise similarities
        labels: (N,) ground truth labels
        k: Number of results to consider
        exclude_self: Exclude self-similarity

    Returns:
        Recall@K (0-1, higher is better)
    """
    n = len(labels)
    hits = 0

    for i in range(n):
        sims = similarity_matrix[i].copy()

        if exclude_self:
            sims[i] = -np.inf

        # Get top-K indices
        top_k = np.argsort(sims)[::-1][:k]

        # Check if any match has same label
        for idx in top_k:
            if labels[idx] == labels[i]:
                hits += 1
                break

    return hits / n


def compute_map(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
    exclude_self: bool = True,
) -> float:
    """Compute Mean Average Precision.

    mAP considers the ranking quality of all relevant results,
    not just the first one.

    Args:
        similarity_matrix: (N, N) pairwise similarities
        labels: (N,) ground truth labels
        exclude_self: Exclude self-similarity

    Returns:
        mAP score (0-1, higher is better)
    """
    n = len(labels)
    average_precisions = []

    for i in range(n):
        sims = similarity_matrix[i].copy()

        if exclude_self:
            sims[i] = -np.inf

        # Get sorted indices
        sorted_indices = np.argsort(sims)[::-1]

        # Count total positives (same label, excluding self)
        num_positives = np.sum(labels == labels[i]) - (1 if exclude_self else 0)

        if num_positives == 0:
            continue

        # Compute precision at each relevant position
        precisions = []
        num_hits = 0

        for rank, idx in enumerate(sorted_indices, 1):
            if labels[idx] == labels[i] and (not exclude_self or idx != i):
                num_hits += 1
                precisions.append(num_hits / rank)

        if precisions:
            average_precisions.append(np.mean(precisions))

    return float(np.mean(average_precisions)) if average_precisions else 0.0


def compute_ari(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Compute Adjusted Rand Index.

    ARI measures clustering agreement with ground truth,
    corrected for chance. Range [-1, 1], 1.0 is perfect.

    Args:
        predicted_labels: Predicted cluster assignments
        true_labels: Ground truth cluster assignments

    Returns:
        ARI score (-1 to 1, higher is better)
    """
    return float(adjusted_rand_score(true_labels, predicted_labels))


def compute_nmi(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Compute Normalized Mutual Information.

    NMI measures shared information between predicted and
    ground truth clusters. Range [0, 1], 1.0 is perfect.

    Args:
        predicted_labels: Predicted cluster assignments
        true_labels: Ground truth cluster assignments

    Returns:
        NMI score (0-1, higher is better)
    """
    return float(normalized_mutual_info_score(true_labels, predicted_labels))


def compute_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Silhouette Score.

    Silhouette measures cluster cohesion vs separation without
    ground truth. Range [-1, 1], higher is better.

    Args:
        embeddings: (N, D) embeddings
        labels: (N,) cluster assignments

    Returns:
        Silhouette score (-1 to 1, higher is better)
    """
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    return float(silhouette_score(embeddings, labels))


def evaluate_retrieval(
    embeddings: np.ndarray,
    labels: np.ndarray,
    isa_labels: Optional[np.ndarray] = None,
    ks: tuple[int, ...] = (1, 5, 10, 50),
) -> EvaluationMetrics:
    """Run full retrieval evaluation.

    Args:
        embeddings: (N, D) embeddings
        labels: (N,) source function labels
        isa_labels: (N,) ISA labels for cross-arch metrics
        ks: K values for Recall@K

    Returns:
        EvaluationMetrics with all retrieval metrics
    """
    sim_matrix = compute_similarity_matrix(embeddings)

    metrics = EvaluationMetrics()

    # Standard retrieval metrics
    metrics.mrr = compute_mrr(sim_matrix, labels)
    metrics.map_score = compute_map(sim_matrix, labels)

    for k in ks:
        recall = compute_recall_at_k(sim_matrix, labels, k)
        if k == 1:
            metrics.recall_at_1 = recall
        elif k == 5:
            metrics.recall_at_5 = recall
        elif k == 10:
            metrics.recall_at_10 = recall
        elif k == 50:
            metrics.recall_at_50 = recall

    # Cross-architecture metrics
    if isa_labels is not None:
        # Compute metrics only for cross-architecture pairs
        # (query and target have different ISAs)
        n = len(labels)
        cross_arch_rr = []
        cross_arch_hits = 0
        cross_arch_total = 0

        for i in range(n):
            sims = sim_matrix[i].copy()
            sims[i] = -np.inf

            sorted_indices = np.argsort(sims)[::-1]

            # Find first cross-arch match
            for rank, idx in enumerate(sorted_indices, 1):
                if labels[idx] == labels[i] and isa_labels[idx] != isa_labels[i]:
                    cross_arch_rr.append(1.0 / rank)
                    if rank == 1:
                        cross_arch_hits += 1
                    cross_arch_total += 1
                    break

        if cross_arch_total > 0:
            metrics.cross_arch_mrr = float(np.mean(cross_arch_rr))
            metrics.cross_arch_recall_at_1 = cross_arch_hits / cross_arch_total

    return metrics


def evaluate_clustering(
    embeddings: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
) -> EvaluationMetrics:
    """Run full clustering evaluation.

    Args:
        embeddings: (N, D) embeddings
        predicted_labels: Predicted cluster assignments
        true_labels: Ground truth cluster assignments

    Returns:
        EvaluationMetrics with all clustering metrics
    """
    metrics = EvaluationMetrics()

    metrics.ari = compute_ari(predicted_labels, true_labels)
    metrics.nmi = compute_nmi(predicted_labels, true_labels)
    metrics.silhouette = compute_silhouette(embeddings, predicted_labels)

    return metrics


def benchmark_retrieval(
    embeddings: np.ndarray,
    labels: np.ndarray,
    pool_sizes: tuple[int, ...] = (1000, 10000, 100000),
) -> dict[int, EvaluationMetrics]:
    """Benchmark retrieval at multiple pool sizes.

    Performance drops significantly with pool size, so this
    tests across different scales.

    Args:
        embeddings: (N, D) embeddings
        labels: (N,) labels
        pool_sizes: Pool sizes to test

    Returns:
        Dictionary mapping pool size to metrics
    """
    results = {}
    n = len(embeddings)

    for pool_size in pool_sizes:
        if pool_size > n:
            continue

        # Random sample for this pool size
        indices = np.random.permutation(n)[:pool_size]
        pool_embeddings = embeddings[indices]
        pool_labels = labels[indices]

        metrics = evaluate_retrieval(pool_embeddings, pool_labels)
        results[pool_size] = metrics

    return results


# Convenience functions for training scripts

def compute_retrieval_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ks: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """Compute retrieval metrics and return as dictionary.
    
    Convenience function for use in training loops.
    
    Args:
        embeddings: (N, D) embeddings
        labels: (N,) labels
        ks: K values for Recall@K
        
    Returns:
        Dictionary with metric names and values
    """
    metrics = evaluate_retrieval(embeddings, labels, ks=tuple(ks))
    
    result = {
        "mrr": metrics.mrr,
        "map": metrics.map_score,
    }
    
    for k in ks:
        if k == 1:
            result["recall@1"] = metrics.recall_at_1
        elif k == 5:
            result["recall@5"] = metrics.recall_at_5
        elif k == 10:
            result["recall@10"] = metrics.recall_at_10
        elif k == 50:
            result["recall@50"] = metrics.recall_at_50
            
    return result


def compute_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: Optional[int] = None,
) -> dict[str, float]:
    """Compute clustering metrics using KMeans.
    
    Convenience function for use in training loops.
    
    Args:
        embeddings: (N, D) embeddings
        labels: (N,) true labels
        n_clusters: Number of clusters (auto-detected from labels if None)
        
    Returns:
        Dictionary with metric names and values
    """
    from sklearn.cluster import KMeans
    
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
        
    # Run KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted = kmeans.fit_predict(embeddings)
    
    metrics = evaluate_clustering(embeddings, predicted, labels)
    
    return {
        "ari": metrics.ari,
        "nmi": metrics.nmi,
        "silhouette": metrics.silhouette,
    }
