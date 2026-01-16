"""CLAUSTRUM Encoder - 12-layer BERT-style transformer for IR sequences.

Implements the core transformer encoder following the plan's architecture:
- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- GELU activation
- Pre-layer normalization
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from claustrum.model.config import ClaustrumConfig


class ClaustrumEmbeddings(nn.Module):
    """Embedding layer combining token, position, and type embeddings."""

    def __init__(self, config: ClaustrumConfig):
        super().__init__()

        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Position IDs buffer
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = token_embeds + position_embeds + type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, config: ClaustrumConfig):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention."""
        batch_size, seq_length, _ = x.shape
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # attention_mask: (batch, 1, 1, seq_len) with 0 for real tokens, large negative for padding
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        batch_size, seq_length = context_layer.shape[:2]
        context_layer = context_layer.view(batch_size, seq_length, self.all_head_size)

        outputs = (context_layer,)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class TransformerBlock(nn.Module):
    """Single transformer block with pre-layer normalization."""

    def __init__(self, config: ClaustrumConfig):
        super().__init__()

        self.attention = MultiHeadSelfAttention(config)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Activation
        if config.hidden_act == "gelu":
            self.activation = F.gelu
        elif config.hidden_act == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        # Self-attention with pre-norm
        attention_output = self.attention(
            self.attention_layer_norm(hidden_states),
            attention_mask,
            output_attentions,
        )

        attention_hidden = attention_output[0]
        attention_hidden = self.attention_output(attention_hidden)
        attention_hidden = self.attention_dropout(attention_hidden)
        hidden_states = hidden_states + attention_hidden  # Residual

        # FFN with pre-norm
        ffn_output = self.intermediate(self.output_layer_norm(hidden_states))
        ffn_output = self.activation(ffn_output)
        ffn_output = self.output(ffn_output)
        ffn_output = self.output_dropout(ffn_output)
        hidden_states = hidden_states + ffn_output  # Residual

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_output[1],)

        return outputs


class ClaustrumEncoder(nn.Module):
    """CLAUSTRUM Encoder - hierarchical transformer for binary code.

    Architecture:
    - Embedding layer (token + position + type)
    - 12 transformer blocks
    - Projection head for final embeddings

    The encoder produces contextualized token representations that can be:
    - Pooled for function-level embeddings
    - Passed to GNN for CFG-aware aggregation
    - Used for pretraining tasks (MLM, CWP, DUP)
    """

    def __init__(self, config: ClaustrumConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = ClaustrumEmbeddings(config)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Projection head for final embeddings
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.embedding_size),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert attention mask to additive mask for softmax.

        Input: (batch, seq_len) with 1 for real tokens, 0 for padding
        Output: (batch, 1, 1, seq_len) with 0 for real tokens, -10000 for padding
        """
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = extended_mask.to(dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0
        return extended_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> dict:
        """Forward pass through the encoder.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            token_type_ids: Token type IDs (batch, seq_len)
            position_ids: Position IDs (batch, seq_len)
            output_hidden_states: Return all hidden states
            output_attentions: Return attention weights
            return_dict: Return dictionary (always True)

        Returns:
            Dictionary with:
                - last_hidden_state: (batch, seq_len, hidden_size)
                - pooler_output: (batch, embedding_size)
                - hidden_states: Optional tuple of layer outputs
                - attentions: Optional tuple of attention weights
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Convert attention mask
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, self.embeddings.token_embeddings.weight.dtype
        )

        # Embeddings
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids)

        all_hidden_states: tuple[torch.Tensor, ...] = ()
        all_attentions: tuple[torch.Tensor, ...] = ()

        # Transformer layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                extended_attention_mask,
                output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Pooling: use [CLS] token (first token)
        cls_output = hidden_states[:, 0, :]
        pooler_output = self.projection(cls_output)

        # L2 normalize embeddings for similarity search
        pooler_output = F.normalize(pooler_output, p=2, dim=-1)

        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooler_output,
            "hidden_states": all_hidden_states if output_hidden_states else None,
            "attentions": all_attentions if output_attentions else None,
        }

    def get_input_embeddings(self) -> nn.Embedding:
        """Return token embeddings."""
        return self.embeddings.token_embeddings

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        """Set token embeddings."""
        self.embeddings.token_embeddings = embeddings

    def resize_token_embeddings(self, new_vocab_size: int) -> nn.Embedding:
        """Resize token embeddings to new vocabulary size."""
        old_embeddings = self.embeddings.token_embeddings
        new_embeddings = nn.Embedding(
            new_vocab_size,
            self.config.hidden_size,
            padding_idx=self.config.pad_token_id,
        )

        # Copy old weights
        num_to_copy = min(old_embeddings.num_embeddings, new_vocab_size)
        new_embeddings.weight.data[:num_to_copy] = old_embeddings.weight.data[:num_to_copy]

        self.embeddings.token_embeddings = new_embeddings
        self.config.vocab_size = new_vocab_size

        return new_embeddings

    @classmethod
    def from_pretrained(cls, model_path: str) -> "ClaustrumEncoder":
        """Load pretrained model.

        Args:
            model_path: Path to model directory

        Returns:
            Loaded ClaustrumEncoder
        """
        from pathlib import Path

        path = Path(model_path)
        config = ClaustrumConfig.from_pretrained(model_path)
        model = cls(config)

        # Load weights
        weights_path = path / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)

        return model

    def save_pretrained(self, save_path: str) -> None:
        """Save model to directory.

        Args:
            save_path: Path to save directory
        """
        from pathlib import Path

        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(str(path / "config.json"))

        # Save weights
        torch.save(self.state_dict(), path / "pytorch_model.bin")
