"""Masking strategies for trace prediction pretraining.

Provides specialized masking for execution trace values,
enabling the model to learn execution semantics through prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import numpy as np

if TYPE_CHECKING:
    from claustrum.tracing.predictor import TraceTokenizer


class MaskingStrategy(Enum):
    """Trace masking strategies."""

    RANDOM = "random"  # Random positions
    CONTIGUOUS = "contiguous"  # Contiguous blocks
    REGISTER_WISE = "register"  # Entire register sequences
    VALUE_BASED = "value"  # Based on value patterns


@dataclass
class TraceMaskingConfig:
    """Configuration for trace masking."""

    strategy: MaskingStrategy = MaskingStrategy.RANDOM
    mask_probability: float = 0.15
    contiguous_span_length: int = 3

    # For value-based masking
    mask_zero_values: bool = True
    mask_repeated_values: bool = True


class TraceMaskingStrategy:
    """Implements masking strategies for trace prediction.

    Args:
        config: Masking configuration
    """

    def __init__(self, config: Optional[TraceMaskingConfig] = None):
        self.config = config or TraceMaskingConfig()

    def create_mask(
        self,
        trace_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create mask for trace values.

        Args:
            trace_values: Trace token IDs (batch, seq_len, num_registers) or (batch, seq_len)
            attention_mask: Optional attention mask to respect padding

        Returns:
            Tuple of (masked_values, labels) where labels has -100 for non-masked positions
        """
        strategy = self.config.strategy

        if strategy == MaskingStrategy.RANDOM:
            return self._random_masking(trace_values, attention_mask)
        elif strategy == MaskingStrategy.CONTIGUOUS:
            return self._contiguous_masking(trace_values, attention_mask)
        elif strategy == MaskingStrategy.REGISTER_WISE:
            return self._register_wise_masking(trace_values, attention_mask)
        elif strategy == MaskingStrategy.VALUE_BASED:
            return self._value_based_masking(trace_values, attention_mask)
        else:
            return self._random_masking(trace_values, attention_mask)

    def _random_masking(
        self,
        trace_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard random masking."""
        device = trace_values.device
        shape = trace_values.shape

        # Create probability matrix
        prob_matrix = torch.full(shape, self.config.mask_probability, device=device)

        # Respect attention mask (don't mask padding)
        if attention_mask is not None:
            mask = attention_mask
            if mask.dim() < prob_matrix.dim():
                # Expand attention mask to match trace dimensions
                for _ in range(prob_matrix.dim() - mask.dim()):
                    mask = mask.unsqueeze(-1)
                mask = mask.expand_as(prob_matrix)
            prob_matrix = prob_matrix * mask.float()

        # Sample mask positions
        masked_indices = torch.bernoulli(prob_matrix).bool()

        # Create labels (-100 for non-masked)
        labels = trace_values.clone()
        labels[~masked_indices] = -100

        # Create masked values (replace with mask token)
        masked_values = trace_values.clone()
        # Mask token ID is typically 1
        masked_values[masked_indices] = 1

        return masked_values, labels

    def _contiguous_masking(
        self,
        trace_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask contiguous spans of trace values.

        Useful for learning temporal dependencies in execution.
        """
        device = trace_values.device
        batch_size = trace_values.shape[0]
        seq_len = trace_values.shape[1]

        # Initialize mask
        masked_indices = torch.zeros_like(trace_values, dtype=torch.bool)

        # Number of spans to mask
        num_spans = int(seq_len * self.config.mask_probability / self.config.contiguous_span_length)
        num_spans = max(1, num_spans)

        for b in range(batch_size):
            # Get valid length from attention mask
            if attention_mask is not None:
                valid_len = attention_mask[b].sum().item()
            else:
                valid_len = seq_len

            # Sample span start positions
            for _ in range(num_spans):
                if valid_len <= self.config.contiguous_span_length:
                    continue

                start = np.random.randint(0, valid_len - self.config.contiguous_span_length)
                end = start + self.config.contiguous_span_length

                if trace_values.dim() == 3:
                    masked_indices[b, start:end, :] = True
                else:
                    masked_indices[b, start:end] = True

        # Create labels and masked values
        labels = trace_values.clone()
        labels[~masked_indices] = -100

        masked_values = trace_values.clone()
        masked_values[masked_indices] = 1  # Mask token

        return masked_values, labels

    def _register_wise_masking(
        self,
        trace_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask entire register sequences.

        Forces the model to predict one register's values from others.
        Only applicable when trace_values has register dimension.
        """
        if trace_values.dim() != 3:
            return self._random_masking(trace_values, attention_mask)

        device = trace_values.device
        batch_size, seq_len, num_registers = trace_values.shape

        # Initialize mask
        masked_indices = torch.zeros_like(trace_values, dtype=torch.bool)

        # Randomly select registers to mask for each batch
        num_to_mask = max(1, int(num_registers * self.config.mask_probability * 2))

        for b in range(batch_size):
            regs_to_mask = np.random.choice(
                num_registers,
                size=min(num_to_mask, num_registers),
                replace=False,
            )

            for r in regs_to_mask:
                masked_indices[b, :, r] = True

        # Respect attention mask
        if attention_mask is not None:
            attn_expanded = attention_mask.unsqueeze(-1).expand_as(masked_indices)
            masked_indices = masked_indices & attn_expanded.bool()

        # Create labels and masked values
        labels = trace_values.clone()
        labels[~masked_indices] = -100

        masked_values = trace_values.clone()
        masked_values[masked_indices] = 1

        return masked_values, labels

    def _value_based_masking(
        self,
        trace_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask based on value patterns.

        Preferentially masks interesting values like:
        - Values after changes
        - Non-zero values
        - Repeated patterns
        """
        device = trace_values.device

        # Start with random masking probability
        prob_matrix = torch.full_like(trace_values, self.config.mask_probability, dtype=torch.float)

        # Increase probability for values after changes
        if trace_values.dim() == 3:
            # Compute changes along sequence dimension
            changes = (trace_values[:, 1:, :] != trace_values[:, :-1, :]).float()
            # Pad to original length
            changes = torch.cat([torch.zeros_like(changes[:, :1, :]), changes], dim=1)
            # Increase probability where changes occurred
            prob_matrix = prob_matrix + changes * 0.1
        elif trace_values.dim() == 2:
            changes = (trace_values[:, 1:] != trace_values[:, :-1]).float()
            changes = torch.cat([torch.zeros_like(changes[:, :1]), changes], dim=1)
            prob_matrix = prob_matrix + changes * 0.1

        # Optionally mask zero values less frequently
        if not self.config.mask_zero_values:
            zero_mask = (trace_values == 0).float()
            prob_matrix = prob_matrix * (1 - 0.5 * zero_mask)

        # Clamp probabilities
        prob_matrix = prob_matrix.clamp(0, 0.5)

        # Respect attention mask
        if attention_mask is not None:
            mask = attention_mask
            if mask.dim() < prob_matrix.dim():
                mask = mask.unsqueeze(-1).expand_as(prob_matrix)
            prob_matrix = prob_matrix * mask.float()

        # Sample mask
        masked_indices = torch.bernoulli(prob_matrix).bool()

        # Create labels and masked values
        labels = trace_values.clone()
        labels[~masked_indices] = -100

        masked_values = trace_values.clone()
        masked_values[masked_indices] = 1

        return masked_values, labels


def create_trace_masks(
    trace_values: torch.Tensor,
    mask_probability: float = 0.15,
    strategy: str = "random",
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience function to create trace masks.

    Args:
        trace_values: Trace token IDs
        mask_probability: Probability of masking each position
        strategy: Masking strategy ("random", "contiguous", "register", "value")
        attention_mask: Optional attention mask

    Returns:
        Tuple of (masked_values, labels, mask_positions)
    """
    config = TraceMaskingConfig(
        strategy=MaskingStrategy(strategy),
        mask_probability=mask_probability,
    )

    masker = TraceMaskingStrategy(config)
    masked_values, labels = masker.create_mask(trace_values, attention_mask)

    # Get mask positions
    mask_positions = labels != -100

    return masked_values, labels, mask_positions


def prepare_trace_batch(
    ir_token_ids: torch.Tensor,
    trace_values: torch.Tensor,
    trace_tokenizer: "TraceTokenizer",
    mask_probability: float = 0.15,
    attention_mask: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Prepare a batch for trace prediction training.

    Args:
        ir_token_ids: IR token IDs (batch, seq_len)
        trace_values: Raw trace values (batch, seq_len, num_registers)
        trace_tokenizer: Tokenizer for trace values
        mask_probability: Masking probability
        attention_mask: Attention mask

    Returns:
        Dictionary with all tensors needed for training
    """
    # Tokenize trace values if not already done
    if trace_values.max() > trace_tokenizer.vocab_size:
        # Values are raw integers, need to tokenize
        batch_size, seq_len, num_regs = trace_values.shape
        tokenized = torch.zeros_like(trace_values)

        for b in range(batch_size):
            for s in range(seq_len):
                for r in range(num_regs):
                    value = trace_values[b, s, r].item()
                    bucket = trace_tokenizer.value_to_bucket(value)
                    tokenized[b, s, r] = bucket + 3  # Offset for special tokens

        trace_values = tokenized

    # Create masks
    masked_values, labels, mask_positions = create_trace_masks(
        trace_values,
        mask_probability=mask_probability,
        strategy="random",
        attention_mask=attention_mask,
    )

    return {
        "input_ids": ir_token_ids,
        "attention_mask": attention_mask
        if attention_mask is not None
        else torch.ones_like(ir_token_ids),
        "trace_labels": labels,
        "trace_mask": mask_positions.any(dim=-1) if mask_positions.dim() == 3 else mask_positions,
    }
