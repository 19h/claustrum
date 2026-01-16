"""Configuration for CLAUSTRUM model architecture.

Defines model hyperparameters following the plan:
- 12-layer BERT encoder
- 768 hidden dimensions
- 128-256 final embedding dimensions
- 3-layer GAT for CFG
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClaustrumConfig:
    """Configuration for the CLAUSTRUM encoder model.

    Architecture follows recommendations from the plan:
    - 12-layer BERT-style transformer for instruction sequences
    - 768 hidden dimensions (BERT-base size)
    - 128-256 dimensional final embeddings
    - 3-layer Graph Attention Network for CFG structure

    Attributes:
        vocab_size: Size of token vocabulary (~50K)
        hidden_size: Hidden dimension (768 recommended)
        num_hidden_layers: Number of transformer layers (12 recommended)
        num_attention_heads: Number of attention heads (12)
        intermediate_size: FFN intermediate size (4x hidden)
        hidden_dropout_prob: Dropout probability
        attention_probs_dropout_prob: Attention dropout
        max_position_embeddings: Maximum sequence length
        embedding_size: Final embedding dimension (128-256)

        # GNN configuration
        gnn_hidden_size: GNN hidden dimension
        gnn_num_layers: Number of GNN layers (3 recommended)
        gnn_num_heads: Number of attention heads in GAT
        gnn_dropout: GNN dropout probability

        # Pooling configuration
        pooling_type: How to aggregate to function embedding
    """

    # Vocabulary
    vocab_size: int = 50000

    # Transformer encoder
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072  # 4 * hidden_size
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512

    # Final embedding
    embedding_size: int = 256  # Output embedding dimension

    # GNN for CFG
    gnn_hidden_size: int = 256
    gnn_num_layers: int = 3
    gnn_num_heads: int = 4
    gnn_dropout: float = 0.1
    use_cfg_gnn: bool = True

    # Pooling
    pooling_type: str = "attention"  # "attention", "mean", "cls"

    # Layer normalization
    layer_norm_eps: float = 1e-12

    # Initialization
    initializer_range: float = 0.02

    # Activation
    hidden_act: str = "gelu"

    # Type embeddings (for multi-modal if needed)
    type_vocab_size: int = 2

    # Special token IDs
    pad_token_id: int = 0
    mask_token_id: int = 4

    # Pretraining
    mlm_probability: float = 0.15  # Masked instruction modeling

    # For contrastive learning
    temperature: float = 0.07

    def __post_init__(self):
        """Validate configuration."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "ClaustrumConfig":
        """Load configuration from pretrained model.

        Args:
            model_name_or_path: Model identifier or path

        Returns:
            ClaustrumConfig instance
        """
        import json
        from pathlib import Path

        config_path = Path(model_name_or_path) / "config.json"

        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            return cls(**config_dict)

        # Default configurations for known model names
        known_configs = {
            "claustrum-base": cls(),
            "claustrum-small": cls(
                hidden_size=512,
                num_hidden_layers=6,
                num_attention_heads=8,
                intermediate_size=2048,
                embedding_size=128,
            ),
            "claustrum-large": cls(
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=16,
                intermediate_size=4096,
                embedding_size=256,
            ),
        }

        if model_name_or_path in known_configs:
            return known_configs[model_name_or_path]

        raise ValueError(f"Unknown model: {model_name_or_path}")

    def save(self, path: str) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save configuration
        """
        import json
        from pathlib import Path

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        with open(save_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# Predefined configurations
CLAUSTRUM_BASE_CONFIG = ClaustrumConfig()

CLAUSTRUM_SMALL_CONFIG = ClaustrumConfig(
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
    embedding_size=128,
    gnn_hidden_size=128,
    gnn_num_layers=2,
)

CLAUSTRUM_LARGE_CONFIG = ClaustrumConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    embedding_size=256,
    gnn_hidden_size=256,
    gnn_num_layers=4,
)
