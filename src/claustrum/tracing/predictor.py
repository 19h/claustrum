"""Execution trace prediction model components.

Provides model heads for predicting execution trace values,
enabling the model to learn execution semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from claustrum.model.config import ClaustrumConfig
from claustrum.tracing.collector import ExecutionTrace, TracePoint


@dataclass
class TraceTokenizerConfig:
    """Configuration for trace value tokenization."""

    # Value tokenization
    num_value_buckets: int = 256  # Buckets for register values
    value_embedding_dim: int = 64

    # Special tokens
    pad_token: str = "[PAD]"
    mask_token: str = "[MASK]"
    unk_token: str = "[UNK]"

    # Register roles to track
    tracked_registers: tuple = (
        "ARG0",
        "ARG1",
        "ARG2",
        "ARG3",
        "RET",
        "SP",
        "FP",
    )


class TraceTokenizer:
    """Tokenizes execution trace values for model input.

    Converts register values to discrete tokens using bucketing,
    enabling the model to predict execution state.

    Args:
        config: Tokenizer configuration
    """

    def __init__(self, config: Optional[TraceTokenizerConfig] = None):
        self.config = config or TraceTokenizerConfig()

        # Build vocabulary
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build token vocabulary."""
        idx = 0

        # Special tokens
        for token in [self.config.pad_token, self.config.mask_token, self.config.unk_token]:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Value bucket tokens
        for bucket in range(self.config.num_value_buckets):
            token = f"VAL_{bucket}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Register tokens (combined with value buckets)
        for reg in self.config.tracked_registers:
            for bucket in range(self.config.num_value_buckets):
                token = f"{reg}_{bucket}"
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1

        self.vocab_size = idx

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.config.pad_token]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id[self.config.mask_token]

    def value_to_bucket(self, value: int) -> int:
        """Convert integer value to bucket index.

        Uses logarithmic bucketing for large value ranges.
        """
        if value == 0:
            return 0

        # Handle negative values
        if value < 0:
            value = abs(value)
            offset = self.config.num_value_buckets // 2
        else:
            offset = 0

        # Logarithmic bucketing
        import math

        bucket = min(int(math.log2(value + 1)), self.config.num_value_buckets // 2 - 1)

        return bucket + offset

    def tokenize_trace(
        self,
        trace: ExecutionTrace,
        max_length: int = 128,
    ) -> dict[str, list[int]]:
        """Tokenize an execution trace.

        Args:
            trace: Execution trace to tokenize
            max_length: Maximum sequence length

        Returns:
            Dictionary with tokenized sequences per register
        """
        result = {}

        for reg in self.config.tracked_registers:
            tokens = []

            for tp in trace.trace_points[:max_length]:
                value = tp.register_state.general.get(reg)

                if value is None:
                    value = tp.register_state.special.get(reg, 0)

                bucket = self.value_to_bucket(value)
                token = f"{reg}_{bucket}"
                token_id = self.token_to_id.get(token, self.token_to_id[self.config.unk_token])
                tokens.append(token_id)

            # Pad to max_length
            while len(tokens) < max_length:
                tokens.append(self.pad_token_id)

            result[reg] = tokens

        return result

    def tokenize_values(
        self,
        values: list[int],
        max_length: int = 128,
    ) -> list[int]:
        """Tokenize a sequence of values (register-agnostic).

        Args:
            values: List of integer values
            max_length: Maximum sequence length

        Returns:
            List of token IDs
        """
        tokens = []

        for value in values[:max_length]:
            bucket = self.value_to_bucket(value)
            token = f"VAL_{bucket}"
            token_id = self.token_to_id.get(token, self.token_to_id[self.config.unk_token])
            tokens.append(token_id)

        # Pad
        while len(tokens) < max_length:
            tokens.append(self.pad_token_id)

        return tokens


class TracePredictionHead(nn.Module):
    """Model head for predicting trace values.

    Given instruction hidden states, predicts the register values
    after executing each instruction.

    Args:
        config: Model configuration
        trace_vocab_size: Size of trace value vocabulary
        num_registers: Number of registers to predict
    """

    def __init__(
        self,
        config: ClaustrumConfig,
        trace_vocab_size: int = 256,
        num_registers: int = 7,
    ):
        super().__init__()

        self.config = config
        self.trace_vocab_size = trace_vocab_size
        self.num_registers = num_registers

        # Project hidden states to trace prediction space
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Per-register prediction heads
        self.register_predictors = nn.ModuleList(
            [nn.Linear(config.hidden_size, trace_vocab_size) for _ in range(num_registers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        register_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict trace values.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            register_idx: If specified, only predict for this register

        Returns:
            Logits (batch, seq_len, num_registers, vocab_size) or
            (batch, seq_len, vocab_size) if register_idx specified
        """
        # Project
        x = self.projection(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)

        if register_idx is not None:
            # Single register prediction
            return self.register_predictors[register_idx](x)
        else:
            # All registers
            outputs = []
            for predictor in self.register_predictors:
                outputs.append(predictor(x))
            # Stack: (batch, seq_len, num_registers, vocab_size)
            return torch.stack(outputs, dim=2)


class TracePredictor(nn.Module):
    """Complete trace prediction model.

    Combines encoder with trace prediction heads for the
    execution trace prediction pretraining objective.

    Args:
        config: Model configuration
        encoder: Optional pre-existing encoder
        trace_config: Trace tokenizer configuration
    """

    def __init__(
        self,
        config: ClaustrumConfig,
        encoder: Optional[nn.Module] = None,
        trace_config: Optional[TraceTokenizerConfig] = None,
    ):
        super().__init__()

        self.config = config
        self.trace_config = trace_config or TraceTokenizerConfig()

        # Encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            from claustrum.model.encoder import ClaustrumEncoder

            self.encoder = ClaustrumEncoder(config)

        # Trace tokenizer
        self.trace_tokenizer = TraceTokenizer(self.trace_config)

        # Prediction head
        self.trace_head = TracePredictionHead(
            config=config,
            trace_vocab_size=self.trace_tokenizer.vocab_size,
            num_registers=len(self.trace_config.tracked_registers),
        )

        # Loss weight
        self.trace_weight = 1.0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        trace_labels: Optional[torch.Tensor] = None,
        trace_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Forward pass with optional trace prediction loss.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            trace_labels: Trace value labels (batch, seq_len, num_registers)
            trace_mask: Mask for trace loss (batch, seq_len)

        Returns:
            Dictionary with outputs and optional loss
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

        # Trace prediction
        trace_logits = self.trace_head(hidden_states)
        outputs["trace_logits"] = trace_logits

        # Compute loss if labels provided
        if trace_labels is not None:
            batch_size, seq_len, num_regs, vocab_size = trace_logits.shape

            # Flatten for cross entropy
            logits_flat = trace_logits.view(-1, vocab_size)
            labels_flat = trace_labels.view(-1)

            # Compute loss
            trace_loss = F.cross_entropy(
                logits_flat,
                labels_flat,
                ignore_index=-100,
                reduction="none",
            )

            # Apply mask if provided
            if trace_mask is not None:
                mask_flat = trace_mask.unsqueeze(-1).expand(-1, -1, num_regs).reshape(-1)
                trace_loss = trace_loss * mask_flat
                trace_loss = trace_loss.sum() / mask_flat.sum().clamp(min=1)
            else:
                trace_loss = trace_loss.mean()

            outputs["trace_loss"] = trace_loss
            outputs["loss"] = self.trace_weight * trace_loss

        return outputs

    def predict_traces(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict trace values for input sequence.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask

        Returns:
            Predicted trace values (batch, seq_len, num_registers)
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            trace_logits = outputs["trace_logits"]
            # Argmax to get predicted values
            return trace_logits.argmax(dim=-1)


class TraceAugmentedPretraining(nn.Module):
    """Pretraining model with trace prediction objective.

    Extends standard pretraining (MIM, CWP, DUP) with trace prediction
    for stronger semantic grounding.
    """

    def __init__(
        self,
        config: ClaustrumConfig,
        trace_config: Optional[TraceTokenizerConfig] = None,
    ):
        super().__init__()

        self.config = config

        # Import pretraining components
        from claustrum.model.pretraining import (
            MaskedInstructionModel,
            ContextWindowPredictor,
            DefUsePredictor,
        )
        from claustrum.model.encoder import ClaustrumEncoder

        # Encoder
        self.encoder = ClaustrumEncoder(config)

        # Standard pretraining heads
        self.mim_head = MaskedInstructionModel(config)
        self.cwp_head = ContextWindowPredictor(config)
        self.dup_head = DefUsePredictor(config)

        # Trace prediction
        self.trace_config = trace_config or TraceTokenizerConfig()
        self.trace_tokenizer = TraceTokenizer(self.trace_config)
        self.trace_head = TracePredictionHead(
            config=config,
            trace_vocab_size=self.trace_tokenizer.vocab_size,
            num_registers=len(self.trace_config.tracked_registers),
        )

        # Loss weights
        self.mim_weight = 1.0
        self.cwp_weight = 0.5
        self.dup_weight = 0.5
        self.trace_weight = 0.5

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # MIM
        masked_lm_labels: Optional[torch.Tensor] = None,
        masked_positions: Optional[torch.Tensor] = None,
        # CWP
        cwp_pairs: Optional[torch.Tensor] = None,
        cwp_labels: Optional[torch.Tensor] = None,
        # DUP
        dup_pairs: Optional[torch.Tensor] = None,
        dup_labels: Optional[torch.Tensor] = None,
        # Trace
        trace_labels: Optional[torch.Tensor] = None,
        trace_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Forward pass with all pretraining objectives.

        Returns:
            Dictionary with all losses and outputs
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
            total_loss += self.mim_weight * mim_loss

        # CWP loss
        if cwp_labels is not None and cwp_pairs is not None:
            cwp_logits = self.cwp_head(hidden_states, cwp_pairs)
            cwp_loss = F.cross_entropy(cwp_logits.view(-1, 2), cwp_labels.view(-1))
            outputs["cwp_loss"] = cwp_loss
            total_loss += self.cwp_weight * cwp_loss

        # DUP loss
        if dup_labels is not None and dup_pairs is not None:
            dup_logits = self.dup_head(hidden_states, dup_pairs)
            dup_loss = F.cross_entropy(dup_logits.view(-1, 2), dup_labels.view(-1))
            outputs["dup_loss"] = dup_loss
            total_loss += self.dup_weight * dup_loss

        # Trace prediction loss
        if trace_labels is not None:
            trace_logits = self.trace_head(hidden_states)
            batch_size, seq_len, num_regs, vocab_size = trace_logits.shape

            trace_loss = F.cross_entropy(
                trace_logits.view(-1, vocab_size),
                trace_labels.view(-1),
                ignore_index=-100,
            )

            outputs["trace_loss"] = trace_loss
            outputs["trace_logits"] = trace_logits
            total_loss += self.trace_weight * trace_loss

        if total_loss > 0:
            outputs["loss"] = total_loss

        return outputs
