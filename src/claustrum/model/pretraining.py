"""Pretraining tasks for CLAUSTRUM.

Implements the self-supervised pretraining objectives from the plan:
- Masked Instruction Modeling (MIM): BERT-style masked LM
- Context Window Prediction (CWP): Predict if instructions are in same block
- Def-Use Prediction (DUP): Predict def-use relationships between instructions
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from claustrum.model.config import ClaustrumConfig
from claustrum.model.encoder import ClaustrumEncoder


class MaskedInstructionModel(nn.Module):
    """Masked Instruction Modeling head.

    Predicts masked tokens in the input sequence, teaching the model
    to understand instruction semantics through context.

    Masking strategy (from PalmTree):
    - 15% of tokens masked
    - 80% replaced with [MASK]
    - 10% replaced with random token
    - 10% unchanged
    """

    def __init__(self, config: ClaustrumConfig):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Output projection to vocabulary
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        masked_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict masked tokens.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            masked_positions: Optional (batch, num_masked) indices of masked positions

        Returns:
            Logits over vocabulary (batch, seq_len, vocab_size) or
            (batch, num_masked, vocab_size) if masked_positions provided
        """
        if masked_positions is not None:
            # Gather only masked positions
            batch_size, seq_len, hidden_size = hidden_states.shape
            num_masked = masked_positions.size(1)

            # Expand masked_positions for gather
            masked_positions_expanded = masked_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
            hidden_states = torch.gather(hidden_states, 1, masked_positions_expanded)

        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)

        return logits


class ContextWindowPredictor(nn.Module):
    """Context Window Prediction head.

    Binary classification: given two instructions, predict whether
    they appear in the same basic block.
    """

    def __init__(self, config: ClaustrumConfig):
        super().__init__()

        # Concatenate two instruction embeddings and classify
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 2),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Predict if instruction pairs are in same block.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            pairs: (batch, num_pairs, 2) indices of instruction pairs

        Returns:
            Logits (batch, num_pairs, 2)
        """
        batch_size, num_pairs, _ = pairs.shape
        hidden_size = hidden_states.size(-1)

        # Gather instruction embeddings for each pair
        idx1 = pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, hidden_size)
        idx2 = pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, hidden_size)

        emb1 = torch.gather(hidden_states, 1, idx1)
        emb2 = torch.gather(hidden_states, 1, idx2)

        # Concatenate and classify
        pair_emb = torch.cat([emb1, emb2], dim=-1)
        logits = self.classifier(pair_emb)

        return logits


class DefUsePredictor(nn.Module):
    """Def-Use Prediction head.

    Binary classification: given two instructions, predict whether
    the first defines a value that the second uses.
    """

    def __init__(self, config: ClaustrumConfig):
        super().__init__()

        # Use asymmetric architecture since def-use is directional
        self.def_transform = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.use_transform = nn.Linear(config.hidden_size, config.hidden_size // 2)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, 2),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Predict def-use relationships.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            pairs: (batch, num_pairs, 2) where pair[0] is def, pair[1] is use

        Returns:
            Logits (batch, num_pairs, 2)
        """
        batch_size, num_pairs, _ = pairs.shape
        hidden_size = hidden_states.size(-1)

        # Gather instruction embeddings
        idx_def = pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, hidden_size)
        idx_use = pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, hidden_size)

        emb_def = torch.gather(hidden_states, 1, idx_def)
        emb_use = torch.gather(hidden_states, 1, idx_use)

        # Transform with role-specific projections
        def_proj = self.def_transform(emb_def)
        use_proj = self.use_transform(emb_use)

        # Concatenate and classify
        pair_emb = torch.cat([def_proj, use_proj], dim=-1)
        logits = self.classifier(pair_emb)

        return logits


class PretrainingModel(nn.Module):
    """Combined pretraining model with all objectives.

    Wraps the encoder and all pretraining heads, computing
    joint loss during pretraining.
    """

    def __init__(self, config: ClaustrumConfig):
        super().__init__()

        self.config = config
        self.encoder = ClaustrumEncoder(config)

        # Pretraining heads
        self.mim_head = MaskedInstructionModel(config)
        self.cwp_head = ContextWindowPredictor(config)
        self.dup_head = DefUsePredictor(config)

        # Loss weights
        self.mim_weight = 1.0
        self.cwp_weight = 0.5
        self.dup_weight = 0.5

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        masked_lm_labels: Optional[torch.Tensor] = None,
        masked_positions: Optional[torch.Tensor] = None,
        cwp_pairs: Optional[torch.Tensor] = None,
        cwp_labels: Optional[torch.Tensor] = None,
        dup_pairs: Optional[torch.Tensor] = None,
        dup_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass with optional loss computation.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            masked_lm_labels: Target labels for masked positions (batch, num_masked)
            masked_positions: Indices of masked positions (batch, num_masked)
            cwp_pairs: Context window pairs (batch, num_pairs, 2)
            cwp_labels: CWP labels (batch, num_pairs)
            dup_pairs: Def-use pairs (batch, num_pairs, 2)
            dup_labels: DUP labels (batch, num_pairs)

        Returns:
            Dictionary with losses and outputs
        """
        # Encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = encoder_outputs["last_hidden_state"]

        outputs = {
            "hidden_states": hidden_states,
            "pooler_output": encoder_outputs["pooler_output"],
        }

        total_loss = 0.0

        # MIM loss
        if masked_lm_labels is not None:
            mim_logits = self.mim_head(hidden_states, masked_positions)
            mim_loss = F.cross_entropy(
                mim_logits.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
                ignore_index=-100,
            )
            outputs["mim_loss"] = mim_loss
            outputs["mim_logits"] = mim_logits
            total_loss += self.mim_weight * mim_loss

        # CWP loss
        if cwp_labels is not None and cwp_pairs is not None:
            cwp_logits = self.cwp_head(hidden_states, cwp_pairs)
            cwp_loss = F.cross_entropy(
                cwp_logits.view(-1, 2),
                cwp_labels.view(-1),
            )
            outputs["cwp_loss"] = cwp_loss
            outputs["cwp_logits"] = cwp_logits
            total_loss += self.cwp_weight * cwp_loss

        # DUP loss
        if dup_labels is not None and dup_pairs is not None:
            dup_logits = self.dup_head(hidden_states, dup_pairs)
            dup_loss = F.cross_entropy(
                dup_logits.view(-1, 2),
                dup_labels.view(-1),
            )
            outputs["dup_loss"] = dup_loss
            outputs["dup_logits"] = dup_logits
            total_loss += self.dup_weight * dup_loss

        if total_loss > 0:
            outputs["loss"] = total_loss

        return outputs

    def get_encoder(self) -> ClaustrumEncoder:
        """Get the underlying encoder."""
        return self.encoder


def create_mlm_masks(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    special_token_ids: set[int],
    mlm_probability: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create masked LM inputs and labels.

    Args:
        input_ids: Original token IDs (batch, seq_len)
        vocab_size: Vocabulary size
        mask_token_id: ID of [MASK] token
        special_token_ids: Set of special token IDs to not mask
        mlm_probability: Probability of masking each token

    Returns:
        (masked_input_ids, labels, mask_positions)
    """
    device = input_ids.device
    batch_size, seq_len = input_ids.shape

    # Create probability mask (don't mask special tokens)
    probability_matrix = torch.full((batch_size, seq_len), mlm_probability, device=device)

    for special_id in special_token_ids:
        probability_matrix.masked_fill_(input_ids == special_id, 0.0)

    # Sample which tokens to mask
    masked = torch.bernoulli(probability_matrix).bool()

    # Create labels (-100 for unmasked)
    labels = input_ids.clone()
    labels[~masked] = -100

    # Get mask positions for each sample
    # (Fixed size by taking max number of masked across batch)
    num_masked = int(masked.sum(dim=1).max().item())
    mask_positions = torch.zeros(batch_size, num_masked, dtype=torch.long, device=device)

    for i in range(batch_size):
        positions = masked[i].nonzero().squeeze(-1)
        if len(positions) > 0:
            # Pad if needed
            if len(positions) < num_masked:
                padding = torch.zeros(num_masked - len(positions), dtype=torch.long, device=device)
                positions = torch.cat([positions, padding])
            mask_positions[i] = positions[:num_masked]

    # Modify input_ids
    masked_input_ids = input_ids.clone()

    # 80% MASK
    indices_mask = torch.bernoulli(torch.full_like(probability_matrix, 0.8)).bool() & masked
    masked_input_ids[indices_mask] = mask_token_id

    # 10% random
    indices_random = (
        torch.bernoulli(torch.full_like(probability_matrix, 0.5)).bool() & masked & ~indices_mask
    )
    random_tokens = torch.randint(vocab_size, (batch_size, seq_len), device=device)
    masked_input_ids[indices_random] = random_tokens[indices_random]

    # 10% unchanged (already handled)

    return masked_input_ids, labels, mask_positions


def sample_cwp_pairs(
    seq_len: int,
    block_boundaries: list[int],
    num_positive: int = 16,
    num_negative: int = 16,
) -> Tuple[list[tuple[int, int]], list[int]]:
    """Sample pairs for Context Window Prediction.

    Args:
        seq_len: Sequence length
        block_boundaries: List of block start indices
        num_positive: Number of positive pairs (same block)
        num_negative: Number of negative pairs (different blocks)

    Returns:
        (pairs, labels) where pairs is list of (idx1, idx2) and labels is 0/1
    """
    pairs = []
    labels = []

    # Build block membership
    block_membership = {}
    for i, start in enumerate(block_boundaries):
        end = block_boundaries[i + 1] if i + 1 < len(block_boundaries) else seq_len
        for j in range(start, end):
            block_membership[j] = i

    # Sample positive pairs (same block)
    for _ in range(num_positive):
        if len(block_boundaries) > 0:
            block_idx = random.randint(0, len(block_boundaries) - 1)
            start = block_boundaries[block_idx]
            end = (
                block_boundaries[block_idx + 1]
                if block_idx + 1 < len(block_boundaries)
                else seq_len
            )

            if end - start >= 2:
                idx1, idx2 = random.sample(range(start, end), 2)
                pairs.append((idx1, idx2))
                labels.append(1)

    # Sample negative pairs (different blocks)
    for _ in range(num_negative):
        if len(block_boundaries) >= 2:
            b1, b2 = random.sample(range(len(block_boundaries)), 2)

            start1 = block_boundaries[b1]
            end1 = block_boundaries[b1 + 1] if b1 + 1 < len(block_boundaries) else seq_len

            start2 = block_boundaries[b2]
            end2 = block_boundaries[b2 + 1] if b2 + 1 < len(block_boundaries) else seq_len

            idx1 = random.randint(start1, end1 - 1)
            idx2 = random.randint(start2, end2 - 1)

            pairs.append((idx1, idx2))
            labels.append(0)

    return pairs, labels
