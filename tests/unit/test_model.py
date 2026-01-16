"""Unit tests for model architecture."""

import pytest
import torch

from claustrum.model import ClaustrumConfig, ClaustrumEncoder


class TestClaustrumConfig:
    """Test model configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ClaustrumConfig()

        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.embedding_size == 256

    def test_config_validation(self):
        """Test configuration validation."""
        # Should raise error for invalid head configuration
        with pytest.raises(ValueError):
            ClaustrumConfig(
                hidden_size=768,
                num_attention_heads=7,  # 768 not divisible by 7
            )

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = ClaustrumConfig()
        config_dict = config.to_dict()

        assert "hidden_size" in config_dict
        assert "num_hidden_layers" in config_dict
        assert config_dict["hidden_size"] == 768


class TestClaustrumEncoder:
    """Test encoder model."""

    @pytest.fixture
    def small_config(self):
        """Small config for fast testing."""
        return ClaustrumConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=64,
            embedding_size=64,
        )

    @pytest.fixture
    def model(self, small_config):
        """Create small model for testing."""
        return ClaustrumEncoder(small_config)

    def test_model_forward(self, model, small_config):
        """Test forward pass produces correct output shapes."""
        batch_size = 4
        seq_length = 32

        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        outputs = model(input_ids, attention_mask)

        assert "last_hidden_state" in outputs
        assert "pooler_output" in outputs

        assert outputs["last_hidden_state"].shape == (
            batch_size,
            seq_length,
            small_config.hidden_size,
        )
        assert outputs["pooler_output"].shape == (batch_size, small_config.embedding_size)

    def test_model_embeddings_normalized(self, model, small_config):
        """Test output embeddings are L2 normalized."""
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        attention_mask = torch.ones(2, 16)

        outputs = model(input_ids, attention_mask)
        embeddings = outputs["pooler_output"]

        # Check L2 normalization
        norms = torch.norm(embeddings, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_model_attention_mask(self, model, small_config):
        """Test attention mask is respected."""
        batch_size = 2
        seq_length = 32

        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_length))

        # Full attention
        mask_full = torch.ones(batch_size, seq_length)
        out_full = model(input_ids, mask_full)

        # Partial attention (mask half)
        mask_partial = torch.ones(batch_size, seq_length)
        mask_partial[:, seq_length // 2 :] = 0
        out_partial = model(input_ids, mask_partial)

        # Outputs should differ
        assert not torch.allclose(out_full["pooler_output"], out_partial["pooler_output"])

    def test_model_output_hidden_states(self, model, small_config):
        """Test returning all hidden states."""
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        attention_mask = torch.ones(2, 16)

        outputs = model(input_ids, attention_mask, output_hidden_states=True)

        assert outputs["hidden_states"] is not None
        # Should have num_layers + 1 states (including input embeddings and final)
        assert len(outputs["hidden_states"]) == small_config.num_hidden_layers + 1

    def test_model_gradient_flow(self, model, small_config):
        """Test gradients flow properly."""
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        attention_mask = torch.ones(2, 16)

        outputs = model(input_ids, attention_mask)

        # Compute loss and backward
        loss = outputs["pooler_output"].sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestModelSaveLoad:
    """Test model serialization."""

    @pytest.fixture
    def small_model(self):
        config = ClaustrumConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=32,
            embedding_size=32,
        )
        return ClaustrumEncoder(config)

    def test_save_and_load(self, small_model, tmp_path):
        """Test saving and loading model."""
        # Generate output
        input_ids = torch.randint(0, 100, (1, 16))
        attention_mask = torch.ones(1, 16)

        original_output = small_model(input_ids, attention_mask)["pooler_output"]

        # Save
        save_path = tmp_path / "model"
        small_model.save_pretrained(str(save_path))

        # Load
        loaded_model = ClaustrumEncoder.from_pretrained(str(save_path))

        # Compare outputs
        loaded_output = loaded_model(input_ids, attention_mask)["pooler_output"]

        assert torch.allclose(original_output, loaded_output, atol=1e-5)
