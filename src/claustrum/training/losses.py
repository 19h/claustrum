"""Loss functions for contrastive learning.

Implements Multi-Positive InfoNCE loss as specified in the plan:

L = -sum_i (1/|P(i)|) * sum_{p in P(i)} log[exp(z_i * z_p / tau) / sum_{a in A(i)} exp(z_i * z_a / tau)]

Where P(i) is the set of all ISA variants from the same source function,
A(i) is all samples in the batch, and tau is temperature (0.07-0.1).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_similarity_matrix(
    embeddings: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings: (batch, embedding_dim)
        normalize: Whether to L2 normalize embeddings

    Returns:
        Similarity matrix (batch, batch)
    """
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    return torch.mm(embeddings, embeddings.t())


class MultiPositiveInfoNCE(nn.Module):
    """Multi-Positive InfoNCE loss for cross-ISA contrastive learning.

    Extends standard contrastive learning to handle multiple positives
    per anchor (all ISA variants of the same source function).

    Args:
        temperature: Temperature for softmax (0.07-0.1 recommended)
        reduction: How to reduce the loss ('mean' or 'sum')
    """

    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Multi-Positive InfoNCE loss.

        Args:
            embeddings: (batch, embedding_dim) normalized embeddings
            labels: (batch,) source function labels (same label = same source)
            weights: Optional (batch,) sample weights

        Returns:
            Scalar loss
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Compute similarity matrix
        sim_matrix = compute_similarity_matrix(embeddings, normalize=True)
        sim_matrix = sim_matrix / self.temperature

        # Create positive mask: 1 if samples have same label
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.t()).float()

        # Remove self-similarity from positives
        eye_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask * (1 - eye_mask)

        # Count positives per anchor
        num_positives = positive_mask.sum(dim=1)

        # Compute log-sum-exp for denominator (all samples except self)
        # Using the log-sum-exp trick for numerical stability
        all_mask = 1 - eye_mask
        sim_masked = sim_matrix - eye_mask * 1e9  # Mask self
        log_sum_exp = torch.logsumexp(sim_masked, dim=1)

        # Compute loss for each anchor
        # For each positive, compute log(exp(sim) / sum_exp) = sim - log_sum_exp
        log_prob = sim_matrix - log_sum_exp.unsqueeze(1)

        # Mask to only positive pairs
        log_prob_positive = log_prob * positive_mask

        # Average over positives for each anchor
        # Avoid division by zero for anchors with no positives
        num_positives_safe = num_positives.clamp(min=1)
        loss_per_anchor = -log_prob_positive.sum(dim=1) / num_positives_safe

        # Mask out anchors with no positives
        valid_anchors = num_positives > 0

        if weights is not None:
            loss_per_anchor = loss_per_anchor * weights

        if self.reduction == "mean":
            if valid_anchors.sum() > 0:
                loss = loss_per_anchor[valid_anchors].mean()
            else:
                loss = loss_per_anchor.mean()
        elif self.reduction == "sum":
            loss = loss_per_anchor.sum()
        else:
            loss = loss_per_anchor

        return loss


class ContrastiveLoss(nn.Module):
    """Standard contrastive loss with single positive per anchor.

    Useful for fine-tuning when you have explicit positive pairs.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            anchor: (batch, dim) anchor embeddings
            positive: (batch, dim) positive embeddings
            negatives: Optional (batch, num_neg, dim) negative embeddings
                       If None, uses in-batch negatives

        Returns:
            Scalar loss
        """
        device = anchor.device
        batch_size = anchor.size(0)

        # Normalize
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)

        # Positive similarity
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature  # (batch,)

        if negatives is not None:
            # Explicit negatives
            negatives = F.normalize(negatives, p=2, dim=-1)
            neg_sim = (
                torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature
            )  # (batch, num_neg)

            # Concatenate positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            # In-batch negatives
            # All other positives serve as negatives
            all_sim = torch.mm(anchor, positive.t()) / self.temperature

            # Diagonal is positive, off-diagonal is negative
            labels = torch.arange(batch_size, device=device)
            logits = all_sim

        loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Extension of contrastive learning that supports multiple positives
    based on class labels.

    Reference: Khosla et al., "Supervised Contrastive Learning"
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        contrast_mode: str = "all",
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            features: (batch, n_views, dim) or (batch, dim)
            labels: (batch,) class labels
            mask: Optional (batch, batch) mask for valid pairs

        Returns:
            Scalar loss
        """
        device = features.device

        if features.dim() == 2:
            features = features.unsqueeze(1)

        batch_size = features.size(0)
        n_views = features.size(1)

        # Flatten views
        features = features.reshape(-1, features.size(-1))  # (batch * n_views, dim)
        features = F.normalize(features, p=2, dim=-1)

        # Expand labels for views
        labels = labels.repeat(n_views)

        # Compute similarity
        sim_matrix = torch.mm(features, features.t()) / self.temperature

        # Create masks
        label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # Remove self-similarity
        logits_mask = 1 - torch.eye(features.size(0), device=device)
        label_mask = label_mask * logits_mask

        # Compute log probabilities
        sim_matrix = sim_matrix - (1 - logits_mask) * 1e9
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)

        # Compute mean log probability over positives
        mean_log_prob = (label_mask * log_prob).sum(dim=1) / label_mask.sum(dim=1).clamp(min=1)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        loss = loss.mean()

        return loss


class TripletLoss(nn.Module):
    """Triplet margin loss.

    Alternative to InfoNCE that uses margin instead of softmax.
    """

    def __init__(
        self,
        margin: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet loss.

        Args:
            anchor: (batch, dim)
            positive: (batch, dim)
            negative: (batch, dim)

        Returns:
            Scalar loss
        """
        # Normalize
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)

        # Distances
        pos_dist = (anchor - positive).pow(2).sum(dim=-1)
        neg_dist = (anchor - negative).pow(2).sum(dim=-1)

        # Margin loss
        loss = F.relu(pos_dist - neg_dist + self.margin)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
