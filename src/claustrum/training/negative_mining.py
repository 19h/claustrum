"""Hard negative mining for contrastive learning.

Implements progressive hard negative mining from the plan:
- Start with random negatives (beta=0)
- Progress to harder negatives (beta=2.0)
- Sample negatives proportional to similarity scores
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MiningConfig:
    """Configuration for hard negative mining."""

    # Temperature for sampling (beta in the plan)
    initial_temperature: float = 0.0
    final_temperature: float = 2.0

    # Warmup period
    warmup_epochs: int = 10

    # Number of negatives to mine per sample
    num_negatives: int = 32

    # Minimum similarity to consider as hard negative
    min_similarity: float = -1.0

    # Maximum similarity (exclude too-similar negatives that might be false negatives)
    max_similarity: float = 0.9

    # Use semi-hard negatives (harder than positive but with margin)
    semi_hard: bool = False
    semi_hard_margin: float = 0.1


class HardNegativeMiner:
    """Mines hard negatives based on embedding similarity.

    Hard negatives are samples that are similar to the anchor
    but belong to different classes (source functions).
    """

    def __init__(self, config: Optional[MiningConfig] = None):
        self.config = config or MiningConfig()
        self.temperature = self.config.initial_temperature

    def update_temperature(self, epoch: int) -> float:
        """Update mining temperature based on epoch.

        Args:
            epoch: Current training epoch

        Returns:
            New temperature value
        """
        if epoch >= self.config.warmup_epochs:
            progress = min(1.0, (epoch - self.config.warmup_epochs) / self.config.warmup_epochs)
            self.temperature = self.config.initial_temperature + progress * (
                self.config.final_temperature - self.config.initial_temperature
            )
        else:
            self.temperature = self.config.initial_temperature

        return self.temperature

    def mine(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_negatives: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mine hard negatives for each sample.

        Args:
            embeddings: (batch, dim) normalized embeddings
            labels: (batch,) source function labels
            num_negatives: Number of negatives to mine per sample

        Returns:
            (negative_indices, negative_weights)
            - negative_indices: (batch, num_neg) indices of mined negatives
            - negative_weights: (batch, num_neg) sampling weights
        """
        device = embeddings.device
        batch_size = embeddings.size(0)
        num_neg = num_negatives or self.config.num_negatives

        # Compute similarity matrix
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.mm(embeddings, embeddings.t())

        # Create negative mask (different labels = potential negatives)
        labels_expanded = labels.unsqueeze(1)
        negative_mask = (labels_expanded != labels_expanded.t()).float()

        # Apply similarity bounds
        valid_mask = negative_mask.clone()
        valid_mask *= (sim_matrix >= self.config.min_similarity).float()
        valid_mask *= (sim_matrix <= self.config.max_similarity).float()

        # Compute sampling probabilities
        if self.temperature > 0:
            # Sample proportional to similarity^temperature
            sampling_logits = sim_matrix * self.temperature
            sampling_logits = sampling_logits - (1 - valid_mask) * 1e9
            sampling_probs = F.softmax(sampling_logits, dim=-1) * valid_mask
        else:
            # Uniform random sampling
            sampling_probs = valid_mask / valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)

        # Sample negatives
        # Use multinomial sampling
        neg_indices = torch.zeros(batch_size, num_neg, dtype=torch.long, device=device)
        neg_weights = torch.zeros(batch_size, num_neg, device=device)

        for i in range(batch_size):
            probs = sampling_probs[i]
            valid_count = (probs > 0).sum().item()

            if valid_count > 0:
                # Sample without replacement if possible
                k = int(min(num_neg, valid_count))
                try:
                    sampled = torch.multinomial(probs, k, replacement=False)
                except RuntimeError:
                    # Fallback to replacement if distribution is degenerate
                    sampled = torch.multinomial(probs.clamp(min=1e-10), k, replacement=True)

                neg_indices[i, :k] = sampled
                neg_weights[i, :k] = probs[sampled]

                # Pad with random valid indices if needed
                if k < num_neg:
                    valid_indices = (probs > 0).nonzero().squeeze(-1)
                    pad_indices = valid_indices[torch.randint(len(valid_indices), (num_neg - k,))]
                    neg_indices[i, k:] = pad_indices
                    neg_weights[i, k:] = probs[pad_indices]
            else:
                # No valid negatives - use all others
                others = (labels != labels[i]).nonzero().squeeze(-1)
                if len(others) > 0:
                    neg_indices[i] = others[torch.randint(len(others), (num_neg,))]
                    neg_weights[i] = 1.0 / num_neg

        # Normalize weights
        neg_weights = neg_weights / neg_weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        return neg_indices, neg_weights

    def get_hard_negative_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_negatives: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings of mined hard negatives.

        Args:
            embeddings: (batch, dim) embeddings
            labels: (batch,) labels
            num_negatives: Number of negatives

        Returns:
            (negative_embeddings, weights)
            - negative_embeddings: (batch, num_neg, dim)
            - weights: (batch, num_neg)
        """
        neg_indices, neg_weights = self.mine(embeddings, labels, num_negatives)

        # Gather negative embeddings
        batch_size, num_neg = neg_indices.shape
        dim = embeddings.size(-1)

        neg_embeddings = embeddings[neg_indices.view(-1)].view(batch_size, num_neg, dim)

        return neg_embeddings, neg_weights


class ProgressiveHardNegativeMiner(HardNegativeMiner):
    """Hard negative miner with additional progressive features.

    Includes:
    - Memory bank for cross-batch negatives
    - Curriculum-aware mining
    - False negative detection
    """

    def __init__(
        self,
        config: Optional[MiningConfig] = None,
        memory_size: int = 65536,
        embedding_dim: int = 256,
    ):
        super().__init__(config)

        self.memory_size = memory_size
        self.embedding_dim = embedding_dim

        # Memory bank for past embeddings
        self._memory_embeddings: Optional[torch.Tensor] = None
        self._memory_labels: Optional[torch.Tensor] = None
        self._memory_ptr = 0
        self._memory_initialized = False

    def init_memory(self, device: torch.device) -> None:
        """Initialize memory bank."""
        self._memory_embeddings = torch.zeros(self.memory_size, self.embedding_dim, device=device)
        self._memory_labels = torch.full((self.memory_size,), -1, dtype=torch.long, device=device)
        self._memory_initialized = True

    def update_memory(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Update memory bank with new embeddings.

        Args:
            embeddings: (batch, dim) new embeddings
            labels: (batch,) labels
        """
        if not self._memory_initialized:
            self.init_memory(embeddings.device)

        # Type narrowing: after init_memory, these are guaranteed non-None
        memory_embeddings = self._memory_embeddings
        memory_labels = self._memory_labels
        assert memory_embeddings is not None and memory_labels is not None

        batch_size = embeddings.size(0)

        # Circular buffer update
        ptr = self._memory_ptr
        if ptr + batch_size <= self.memory_size:
            memory_embeddings[ptr : ptr + batch_size] = embeddings.detach()
            memory_labels[ptr : ptr + batch_size] = labels.detach()
        else:
            # Wrap around
            first_part = self.memory_size - ptr
            memory_embeddings[ptr:] = embeddings[:first_part].detach()
            memory_labels[ptr:] = labels[:first_part].detach()

            second_part = batch_size - first_part
            memory_embeddings[:second_part] = embeddings[first_part:].detach()
            memory_labels[:second_part] = labels[first_part:].detach()

        self._memory_ptr = (ptr + batch_size) % self.memory_size

    def mine_from_memory(
        self,
        query_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
        num_negatives: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mine hard negatives from memory bank.

        Args:
            query_embeddings: (batch, dim) query embeddings
            query_labels: (batch,) query labels
            num_negatives: Number of negatives to mine

        Returns:
            (negative_embeddings, weights)
        """
        if not self._memory_initialized:
            return self.get_hard_negative_embeddings(query_embeddings, query_labels, num_negatives)

        # Type narrowing: after init, these are guaranteed non-None
        memory_embeddings = self._memory_embeddings
        memory_labels = self._memory_labels
        if memory_embeddings is None or memory_labels is None:
            return self.get_hard_negative_embeddings(query_embeddings, query_labels, num_negatives)

        device = query_embeddings.device
        batch_size = query_embeddings.size(0)
        num_neg = num_negatives or self.config.num_negatives

        # Valid memory entries
        valid_mask = memory_labels >= 0
        valid_embeddings = memory_embeddings[valid_mask]
        valid_labels = memory_labels[valid_mask]

        if len(valid_embeddings) == 0:
            return self.get_hard_negative_embeddings(query_embeddings, query_labels, num_negatives)

        # Compute similarity to memory
        query_norm = F.normalize(query_embeddings, p=2, dim=-1)
        memory_norm = F.normalize(valid_embeddings, p=2, dim=-1)
        sim_matrix = torch.mm(query_norm, memory_norm.t())  # (batch, memory_valid)

        # Mask same-label entries
        negative_mask = (query_labels.unsqueeze(1) != valid_labels.unsqueeze(0)).float()

        # Apply bounds
        valid_neg_mask = negative_mask.clone()
        valid_neg_mask *= (sim_matrix >= self.config.min_similarity).float()
        valid_neg_mask *= (sim_matrix <= self.config.max_similarity).float()

        # Compute sampling probabilities
        if self.temperature > 0:
            sampling_logits = sim_matrix * self.temperature
            sampling_logits = sampling_logits - (1 - valid_neg_mask) * 1e9
            sampling_probs = F.softmax(sampling_logits, dim=-1) * valid_neg_mask
        else:
            sampling_probs = valid_neg_mask / valid_neg_mask.sum(dim=-1, keepdim=True).clamp(min=1)

        # Sample negatives
        neg_embeddings = torch.zeros(batch_size, num_neg, self.embedding_dim, device=device)
        neg_weights = torch.zeros(batch_size, num_neg, device=device)

        for i in range(batch_size):
            probs = sampling_probs[i]
            valid_count = (probs > 0).sum().item()

            if valid_count > 0:
                k = int(min(num_neg, valid_count))
                try:
                    sampled = torch.multinomial(probs, k, replacement=False)
                except RuntimeError:
                    sampled = torch.multinomial(probs.clamp(min=1e-10), k, replacement=True)

                neg_embeddings[i, :k] = valid_embeddings[sampled]
                neg_weights[i, :k] = probs[sampled]

                if k < num_neg:
                    # Pad with random
                    valid_indices = (probs > 0).nonzero().squeeze(-1)
                    pad_indices = valid_indices[torch.randint(len(valid_indices), (num_neg - k,))]
                    neg_embeddings[i, k:] = valid_embeddings[pad_indices]
                    neg_weights[i, k:] = probs[pad_indices]

        neg_weights = neg_weights / neg_weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        return neg_embeddings, neg_weights

    def sample_hard_negatives(
        self,
        query_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
        num_negatives: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sample hard negatives from memory bank.

        Convenience method that returns None if memory is not ready.

        Args:
            query_embeddings: (batch, dim) query embeddings
            query_labels: (batch,) query labels
            num_negatives: Number of negatives to sample

        Returns:
            Tuple of (embeddings, labels) or (None, None) if not ready
        """
        if not self._memory_initialized:
            return None, None

        memory_labels = self._memory_labels
        if memory_labels is None:
            return None, None

        num_neg = num_negatives if num_negatives is not None else self.config.num_negatives
        valid_count = (memory_labels >= 0).sum().item()
        if valid_count < num_neg * 2:
            return None, None

        neg_embeddings, neg_weights = self.mine_from_memory(
            query_embeddings, query_labels, num_negatives
        )

        # Create pseudo-labels for negatives (different from all queries)
        max_label = query_labels.max().item()
        neg_labels = torch.arange(
            max_label + 1,
            max_label + 1 + neg_embeddings.size(0),
            device=neg_embeddings.device,
        )

        # Flatten to (batch * num_neg, dim)
        neg_embeddings_flat = neg_embeddings.view(-1, neg_embeddings.size(-1))
        neg_labels_flat = neg_labels.repeat_interleave(neg_embeddings.size(1))

        return neg_embeddings_flat, neg_labels_flat

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            "memory_embeddings": self._memory_embeddings,
            "memory_labels": self._memory_labels,
            "memory_ptr": self._memory_ptr,
            "memory_initialized": self._memory_initialized,
            "temperature": self.temperature,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict."""
        self._memory_embeddings = state_dict.get("memory_embeddings")
        self._memory_labels = state_dict.get("memory_labels")
        self._memory_ptr = state_dict.get("memory_ptr", 0)
        self._memory_initialized = state_dict.get("memory_initialized", False)
        self.temperature = state_dict.get("temperature", self.config.initial_temperature)
