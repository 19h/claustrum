"""Pooling mechanisms for aggregating token/block embeddings.

Implements attention pooling as recommended in the plan for
function-level embedding from basic block representations.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from claustrum.model.config import ClaustrumConfig


class AttentionPooling(nn.Module):
    """Attention-based pooling weighted by block centrality.

    Computes attention weights over basic blocks based on their
    importance in the control flow graph, then produces a
    weighted average for the function embedding.
    """

    def __init__(self, config: ClaustrumConfig):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.Tanh(),
            nn.Linear(config.hidden_size // 4, 1),
        )

        self.output_proj = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        block_boundaries: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Compute attention-pooled embedding.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) with 1 for real tokens
            block_boundaries: Optional list of block start indices

        Returns:
            Pooled embedding (batch, embedding_size)
        """
        # Compute attention scores
        scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)

        if attention_mask is not None:
            # Mask padding
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Weighted sum
        pooled = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            hidden_states,  # (batch, seq_len, hidden_size)
        ).squeeze(1)  # (batch, hidden_size)

        # Project to output dimension
        output = self.output_proj(pooled)

        # L2 normalize
        output = F.normalize(output, p=2, dim=-1)

        return output


class MeanPooling(nn.Module):
    """Mean pooling over valid tokens."""

    def __init__(self, config: ClaustrumConfig):
        super().__init__()
        self.output_proj = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mean-pooled embedding.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)

        Returns:
            Pooled embedding (batch, embedding_size)
        """
        if attention_mask is not None:
            # Mask padding
            mask = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states * mask
            pooled = hidden_states.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        output = self.output_proj(pooled)
        output = F.normalize(output, p=2, dim=-1)

        return output


class CLSPooling(nn.Module):
    """CLS token pooling (first token)."""

    def __init__(self, config: ClaustrumConfig):
        super().__init__()
        self.output_proj = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get CLS token embedding.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            CLS embedding (batch, embedding_size)
        """
        cls_output = hidden_states[:, 0, :]
        output = self.output_proj(cls_output)
        output = F.normalize(output, p=2, dim=-1)
        return output


class HierarchicalPooling(nn.Module):
    """Two-stage hierarchical pooling.

    First pools tokens within each basic block, then pools
    block embeddings to function embedding.
    """

    def __init__(self, config: ClaustrumConfig):
        super().__init__()

        # Token-to-block pooling
        self.token_attention = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.Tanh(),
            nn.Linear(config.hidden_size // 4, 1),
        )

        # Block-to-function pooling
        self.block_attention = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.Tanh(),
            nn.Linear(config.hidden_size // 4, 1),
        )

        self.output_proj = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        block_boundaries: Optional[list[list[int]]] = None,
    ) -> torch.Tensor:
        """Hierarchical pooling.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
            block_boundaries: List of block start indices per batch item

        Returns:
            Function embedding (batch, embedding_size)
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        if block_boundaries is None:
            # Fall back to attention pooling
            scores = self.token_attention(hidden_states).squeeze(-1)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        else:
            # Hierarchical pooling
            function_embeddings = []

            for b in range(batch_size):
                boundaries = block_boundaries[b]
                blocks = []

                # Pool tokens within each block
                for i, start in enumerate(boundaries):
                    end = boundaries[i + 1] if i + 1 < len(boundaries) else hidden_states.size(1)

                    block_hidden = hidden_states[b, start:end]

                    if attention_mask is not None:
                        block_mask = attention_mask[b, start:end]
                        block_hidden = block_hidden[block_mask == 1]

                    if block_hidden.size(0) > 0:
                        # Attention pool within block
                        scores = self.token_attention(block_hidden).squeeze(-1)
                        weights = F.softmax(scores, dim=-1)
                        block_embed = (weights.unsqueeze(-1) * block_hidden).sum(dim=0)
                        blocks.append(block_embed)

                if blocks:
                    # Stack blocks and pool to function
                    block_tensor = torch.stack(blocks)  # (num_blocks, hidden)
                    scores = self.block_attention(block_tensor).squeeze(-1)
                    weights = F.softmax(scores, dim=-1)
                    func_embed = (weights.unsqueeze(-1) * block_tensor).sum(dim=0)
                else:
                    # Fallback to mean
                    func_embed = hidden_states[b].mean(dim=0)

                function_embeddings.append(func_embed)

            pooled = torch.stack(function_embeddings)

        output = self.output_proj(pooled)
        output = F.normalize(output, p=2, dim=-1)

        return output


def get_pooling(config: ClaustrumConfig) -> nn.Module:
    """Get pooling module based on config.

    Args:
        config: Model configuration

    Returns:
        Pooling module
    """
    pooling_types = {
        "attention": AttentionPooling,
        "mean": MeanPooling,
        "cls": CLSPooling,
        "hierarchical": HierarchicalPooling,
    }

    pooling_cls = pooling_types.get(config.pooling_type, AttentionPooling)
    return pooling_cls(config)
