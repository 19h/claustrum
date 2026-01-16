"""Unit tests for execution trace collection and prediction."""

import pytest
import torch

from claustrum.tracing import (
    ExecutionTrace,
    TracePoint,
    RegisterState,
    MemoryAccess,
    AccessType,
)
from claustrum.tracing.predictor import (
    TraceTokenizer,
    TraceTokenizerConfig,
    TracePredictionHead,
)
from claustrum.tracing.masking import (
    TraceMaskingStrategy,
    TraceMaskingConfig,
    MaskingStrategy,
    create_trace_masks,
)


class TestRegisterState:
    """Test RegisterState data structure."""

    def test_register_state_creation(self):
        """Test creating register state."""
        state = RegisterState(
            general={"ARG0": 100, "ARG1": 200, "RET": 0},
            flags={"ZF": True, "CF": False},
            special={"SP": 0x7FFF0000, "FP": 0x7FFF0100},
        )

        assert state.general["ARG0"] == 100
        assert state.flags["ZF"] is True
        assert state.special["SP"] == 0x7FFF0000

    def test_register_state_hash(self):
        """Test register state hashing."""
        state1 = RegisterState(general={"ARG0": 100})
        state2 = RegisterState(general={"ARG0": 100})
        state3 = RegisterState(general={"ARG0": 200})

        # Same state should have same hash
        assert state1.hash() == state2.hash()
        # Different state should have different hash
        assert state1.hash() != state3.hash()

    def test_register_state_to_dict(self):
        """Test converting to dictionary."""
        state = RegisterState(
            general={"ARG0": 100},
            flags={"ZF": True},
            special={"SP": 0x7FFF0000},
        )

        d = state.to_dict()

        assert "general" in d
        assert "flags" in d
        assert "special" in d


class TestTracePoint:
    """Test TracePoint data structure."""

    def test_trace_point_creation(self):
        """Test creating a trace point."""
        tp = TracePoint(
            instruction_index=0,
            instruction_address=0x1000,
            instruction_bytes=b"\x48\x89\xe5",
            register_state=RegisterState(general={"RET": 42}),
            memory_accesses=[],
            opcode="mov",
            operands=["rbp", "rsp"],
        )

        assert tp.instruction_index == 0
        assert tp.opcode == "mov"
        assert tp.register_state.general["RET"] == 42


class TestExecutionTrace:
    """Test ExecutionTrace data structure."""

    @pytest.fixture
    def sample_trace(self):
        """Create a sample execution trace."""
        trace_points = [
            TracePoint(
                instruction_index=i,
                instruction_address=0x1000 + i * 4,
                instruction_bytes=b"\x00" * 4,
                register_state=RegisterState(general={"ARG0": i * 10, "RET": i * 5}),
            )
            for i in range(5)
        ]

        return ExecutionTrace(
            function_id="test_func",
            isa="x86_64",
            trace_points=trace_points,
            input_state={"ARG0": 0, "ARG1": 0},
        )

    def test_trace_creation(self, sample_trace):
        """Test trace creation."""
        assert sample_trace.function_id == "test_func"
        assert sample_trace.num_instructions == 5

    def test_get_register_sequence(self, sample_trace):
        """Test extracting register value sequence."""
        arg0_values = sample_trace.get_register_sequence("ARG0")

        assert len(arg0_values) == 5
        assert arg0_values == [0, 10, 20, 30, 40]

    def test_coverage(self, sample_trace):
        """Test coverage calculation."""
        # Coverage is set in post_init
        assert sample_trace.coverage > 0


class TestTraceTokenizer:
    """Test trace value tokenization."""

    def test_tokenizer_creation(self):
        """Test tokenizer creation with vocabulary."""
        tokenizer = TraceTokenizer()

        assert tokenizer.vocab_size > 0
        assert tokenizer.pad_token_id >= 0
        assert tokenizer.mask_token_id >= 0

    def test_value_to_bucket(self):
        """Test value bucketing."""
        tokenizer = TraceTokenizer()

        # Zero should always be bucket 0
        assert tokenizer.value_to_bucket(0) == 0

        # Small values should be small buckets
        assert tokenizer.value_to_bucket(1) < tokenizer.value_to_bucket(1000)

    def test_tokenize_values(self):
        """Test tokenizing a sequence of values."""
        tokenizer = TraceTokenizer()

        values = [0, 10, 100, 1000, 10000]
        tokens = tokenizer.tokenize_values(values, max_length=10)

        assert len(tokens) == 10
        # First 5 should be value tokens, rest padding
        assert tokens[5:] == [tokenizer.pad_token_id] * 5

    def test_tokenize_trace(self):
        """Test tokenizing a full trace."""
        tokenizer = TraceTokenizer()

        trace = ExecutionTrace(
            function_id="test",
            isa="x86_64",
            trace_points=[
                TracePoint(
                    instruction_index=i,
                    instruction_address=0x1000 + i,
                    instruction_bytes=b"\x00",
                    register_state=RegisterState(
                        general={"ARG0": i * 10, "ARG1": i * 20, "RET": 0},
                        special={"SP": 0x7FFF0000, "FP": 0x7FFF0100},
                    ),
                )
                for i in range(3)
            ],
            input_state={},
        )

        tokenized = tokenizer.tokenize_trace(trace, max_length=8)

        assert "ARG0" in tokenized
        assert len(tokenized["ARG0"]) == 8


class TestTracePredictionHead:
    """Test trace prediction model head."""

    @pytest.fixture
    def config(self):
        """Create model config."""
        from claustrum.model.config import ClaustrumConfig

        return ClaustrumConfig(
            hidden_size=128,
            vocab_size=1000,
            layer_norm_eps=1e-12,
        )

    def test_prediction_head_creation(self, config):
        """Test creating prediction head."""
        head = TracePredictionHead(
            config=config,
            trace_vocab_size=256,
            num_registers=7,
        )

        assert head.trace_vocab_size == 256
        assert head.num_registers == 7

    def test_prediction_forward(self, config):
        """Test forward pass."""
        head = TracePredictionHead(
            config=config,
            trace_vocab_size=256,
            num_registers=7,
        )

        hidden_states = torch.randn(2, 32, 128)  # batch=2, seq=32, hidden=128

        logits = head(hidden_states)

        assert logits.shape == (2, 32, 7, 256)  # (batch, seq, num_regs, vocab)

    def test_single_register_prediction(self, config):
        """Test predicting single register."""
        head = TracePredictionHead(
            config=config,
            trace_vocab_size=256,
            num_registers=7,
        )

        hidden_states = torch.randn(2, 32, 128)

        logits = head(hidden_states, register_idx=0)

        assert logits.shape == (2, 32, 256)


class TestTraceMasking:
    """Test trace masking strategies."""

    def test_random_masking(self):
        """Test random masking strategy."""
        config = TraceMaskingConfig(
            strategy=MaskingStrategy.RANDOM,
            mask_probability=0.15,
        )
        masker = TraceMaskingStrategy(config)

        trace_values = torch.randint(0, 100, (2, 32, 7))  # batch=2, seq=32, regs=7

        masked_values, labels = masker.create_mask(trace_values)

        assert masked_values.shape == trace_values.shape
        assert labels.shape == trace_values.shape

        # Some positions should be masked
        assert (labels != -100).any()
        # Most positions should be unmasked
        assert (labels == -100).sum() > (labels != -100).sum()

    def test_contiguous_masking(self):
        """Test contiguous span masking."""
        config = TraceMaskingConfig(
            strategy=MaskingStrategy.CONTIGUOUS,
            mask_probability=0.15,
            contiguous_span_length=3,
        )
        masker = TraceMaskingStrategy(config)

        trace_values = torch.randint(0, 100, (2, 32, 7))

        masked_values, labels = masker.create_mask(trace_values)

        # Should have masked positions
        assert (labels != -100).any()

    def test_register_wise_masking(self):
        """Test register-wise masking."""
        config = TraceMaskingConfig(
            strategy=MaskingStrategy.REGISTER_WISE,
            mask_probability=0.15,
        )
        masker = TraceMaskingStrategy(config)

        trace_values = torch.randint(0, 100, (2, 32, 7))

        masked_values, labels = masker.create_mask(trace_values)

        # Should mask entire registers
        assert (labels != -100).any()

    def test_create_trace_masks_convenience(self):
        """Test convenience function."""
        trace_values = torch.randint(0, 100, (2, 32, 7))

        masked_values, labels, mask_positions = create_trace_masks(
            trace_values,
            mask_probability=0.15,
            strategy="random",
        )

        assert masked_values.shape == trace_values.shape
        assert labels.shape == trace_values.shape
        assert mask_positions.shape == trace_values.shape

    def test_masking_respects_attention_mask(self):
        """Test that masking respects attention mask."""
        config = TraceMaskingConfig(
            strategy=MaskingStrategy.RANDOM,
            mask_probability=0.5,  # High probability to ensure we'd see masking
        )
        masker = TraceMaskingStrategy(config)

        trace_values = torch.randint(0, 100, (2, 32))
        attention_mask = torch.zeros(2, 32)
        attention_mask[:, :16] = 1  # Only first 16 positions are valid

        masked_values, labels = masker.create_mask(trace_values, attention_mask)

        # Padded positions should not be in labels
        assert (labels[:, 16:] == -100).all()


class TestTracePredictor:
    """Test complete trace prediction model."""

    @pytest.fixture
    def config(self):
        """Create model config."""
        from claustrum.model.config import ClaustrumConfig

        return ClaustrumConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            vocab_size=1000,
        )

    def test_trace_predictor_creation(self, config):
        """Test creating trace predictor."""
        from claustrum.tracing.predictor import TracePredictor

        predictor = TracePredictor(config)

        assert predictor.encoder is not None
        assert predictor.trace_head is not None

    def test_trace_predictor_forward(self, config):
        """Test forward pass without labels."""
        from claustrum.tracing.predictor import TracePredictor

        predictor = TracePredictor(config)

        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)

        outputs = predictor(input_ids, attention_mask)

        assert "hidden_states" in outputs
        assert "trace_logits" in outputs
        assert "loss" not in outputs  # No labels provided

    def test_trace_predictor_with_labels(self, config):
        """Test forward pass with labels."""
        from claustrum.tracing.predictor import TracePredictor

        predictor = TracePredictor(config)

        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)
        trace_labels = torch.randint(0, predictor.trace_tokenizer.vocab_size, (2, 32, 7))

        outputs = predictor(input_ids, attention_mask, trace_labels=trace_labels)

        assert "loss" in outputs
        assert outputs["loss"].requires_grad
