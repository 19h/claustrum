"""Unit tests for IR tokenization."""

import pytest
from claustrum.tokenization import IRTokenizer, Vocabulary
from claustrum.normalization.normalized_ir import (
    NormalizedFunction,
    NormalizedBlock,
    NormalizedInstruction,
    NormalizedOperand,
)


class TestVocabulary:
    """Test vocabulary management."""

    def test_vocabulary_creation(self):
        """Test vocabulary is created with expected tokens."""
        vocab = Vocabulary()

        # Check special tokens
        assert vocab.encode("[PAD]") == 0
        assert vocab.encode("[UNK]") == 1
        assert vocab.encode("[CLS]") == 2
        assert vocab.encode("[MASK]") == 4

        # Check opcodes
        assert "ADD" in vocab
        assert "LOAD" in vocab
        assert "CALL" in vocab

        # Check register roles
        assert "$ARG0" in vocab
        assert "$SP" in vocab
        assert "$RET" in vocab

    def test_vocabulary_size(self):
        """Test vocabulary has reasonable size."""
        vocab = Vocabulary()

        # Should be around 50K as specified in plan
        assert len(vocab) > 1000
        assert len(vocab) < 100000

    def test_encode_decode(self):
        """Test encode/decode roundtrip."""
        vocab = Vocabulary()

        tokens = ["ADD", "$ARG0", "$ARG1"]
        ids = vocab.encode_batch(tokens)
        decoded = vocab.decode_batch(ids)

        assert decoded == tokens

    def test_unknown_token(self):
        """Test unknown tokens map to UNK."""
        vocab = Vocabulary()

        unknown_id = vocab.encode("UNKNOWN_TOKEN_12345")
        assert unknown_id == vocab.unk_token_id


class TestTokenizer:
    """Test IR tokenization."""

    @pytest.fixture
    def sample_function(self):
        """Create a sample normalized function."""
        block = NormalizedBlock(
            block_id=0,
            instructions=[
                NormalizedInstruction(
                    opcode="ADD",
                    dest=NormalizedOperand.register("RET"),
                    src=[
                        NormalizedOperand.register("ARG0"),
                        NormalizedOperand.register("ARG1"),
                    ],
                ),
                NormalizedInstruction(
                    opcode="RET",
                ),
            ],
            is_entry=True,
            is_exit=True,
        )

        return NormalizedFunction(
            blocks=[block],
            cfg_edges=[],
            original_address=0x1000,
        )

    def test_tokenize_function(self, sample_function):
        """Test function tokenization."""
        tokenizer = IRTokenizer()
        output = tokenizer.tokenize(sample_function)

        assert len(output.input_ids) > 0
        assert len(output.attention_mask) == len(output.input_ids)

    def test_tokenize_with_padding(self, sample_function):
        """Test tokenization produces padded output."""
        from claustrum.tokenization.tokenizer import TokenizerConfig

        config = TokenizerConfig(max_length=64, padding="max_length")
        tokenizer = IRTokenizer(config=config)

        output = tokenizer.tokenize(sample_function)

        assert len(output.input_ids) == 64
        assert len(output.attention_mask) == 64

    def test_tokenize_preserves_structure(self, sample_function):
        """Test tokenization includes block markers."""
        tokenizer = IRTokenizer()
        output = tokenizer.tokenize(sample_function)

        # Should have block boundaries
        assert output.block_boundaries is not None
        assert len(output.block_boundaries) > 0

    def test_batch_tokenize(self, sample_function):
        """Test batch tokenization."""
        tokenizer = IRTokenizer()

        functions = [sample_function, sample_function]
        batch = tokenizer.batch_tokenize(functions)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert len(batch["input_ids"]) == 2

    def test_decode(self, sample_function):
        """Test decoding tokens back to strings."""
        tokenizer = IRTokenizer()

        output = tokenizer.tokenize(sample_function)
        decoded = tokenizer.decode(output.input_ids)

        # Should contain opcode
        assert "ADD" in decoded or "RET" in decoded


class TestTokenizerConfig:
    """Test tokenizer configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from claustrum.tokenization.tokenizer import TokenizerConfig

        config = TokenizerConfig()

        assert config.max_length == 512
        assert config.add_special_tokens is True
        assert config.include_block_markers is True

    def test_config_truncation(self):
        """Test truncation configuration."""
        from claustrum.tokenization.tokenizer import TokenizerConfig

        config = TokenizerConfig(max_length=32, truncation=True)
        tokenizer = IRTokenizer(config=config)

        # Create a long function
        instructions = [NormalizedInstruction(opcode="ADD") for _ in range(100)]
        block = NormalizedBlock(block_id=0, instructions=instructions)
        func = NormalizedFunction(blocks=[block])

        output = tokenizer.tokenize(func)

        assert len(output.input_ids) == 32
