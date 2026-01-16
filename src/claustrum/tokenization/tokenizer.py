"""IR Tokenizer for converting normalized IR to token sequences.

Implements full-instruction tokenization (as per NDSS BAR 2025 findings)
rather than subword methods, producing sequences suitable for transformer input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import numpy as np

from claustrum.normalization.normalized_ir import (
    NormalizedFunction,
    NormalizedBlock,
    NormalizedInstruction,
    NormalizedOperand,
)
from claustrum.tokenization.vocabulary import Vocabulary, get_default_vocabulary


@dataclass
class TokenizerConfig:
    """Configuration for IR tokenization."""

    # Maximum sequence length
    max_length: int = 512

    # Padding strategy
    padding: str = "max_length"  # "max_length", "longest", "do_not_pad"
    truncation: bool = True

    # Special token handling
    add_special_tokens: bool = True

    # Structure tokens
    include_block_markers: bool = True
    include_func_markers: bool = True

    # Instruction formatting
    separate_operands: bool = True  # Emit operands as separate tokens
    include_opcode: bool = True

    # Attention mask
    return_attention_mask: bool = True


@dataclass
class TokenizedOutput:
    """Output of tokenization."""

    input_ids: Any  # list[int] or torch.Tensor or np.ndarray depending on return_tensors
    attention_mask: Any  # list[int] or torch.Tensor or np.ndarray depending on return_tensors

    # Original token strings (for debugging)
    tokens: Optional[list[str]] = None

    # Block boundaries for hierarchical models
    block_boundaries: Optional[list[int]] = None  # Indices where blocks start

    # Length info
    original_length: int = 0


class IRTokenizer:
    """Tokenizer for normalized IR sequences.

    Converts normalized functions to token ID sequences suitable
    for transformer models. Uses full-instruction tokenization
    with a fixed vocabulary (~50K tokens).
    """

    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        config: Optional[TokenizerConfig] = None,
    ):
        self.vocab = vocab or get_default_vocabulary()
        self.config = config or TokenizerConfig()

    def tokenize(
        self,
        function: NormalizedFunction,
        return_tensors: Optional[str] = None,
    ) -> TokenizedOutput:
        """Tokenize a normalized function.

        Args:
            function: Normalized function to tokenize
            return_tensors: Optional tensor type ("pt" for PyTorch, "np" for NumPy)

        Returns:
            TokenizedOutput with token IDs and attention mask
        """
        tokens = []
        block_boundaries = []

        # Function start marker
        if self.config.include_func_markers and self.config.add_special_tokens:
            tokens.append("[CLS]")

        # Process each block
        for block in function.blocks:
            block_boundaries.append(len(tokens))
            block_tokens = self._tokenize_block(block)
            tokens.extend(block_tokens)

        # Function end marker
        if self.config.include_func_markers and self.config.add_special_tokens:
            tokens.append("[SEP]")

        original_length = len(tokens)

        # Truncation
        if self.config.truncation and len(tokens) > self.config.max_length:
            tokens = tokens[: self.config.max_length]
            # Ensure we end with SEP if we had it
            if self.config.add_special_tokens:
                tokens[-1] = "[SEP]"

        # Convert to IDs
        input_ids = [self.vocab.encode(t) for t in tokens]

        # Attention mask (1 for real tokens)
        attention_mask = [1] * len(input_ids)

        # Padding
        if self.config.padding == "max_length":
            pad_length = self.config.max_length - len(input_ids)
            if pad_length > 0:
                input_ids.extend([self.vocab.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)

        result = TokenizedOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokens=tokens if original_length <= 1000 else None,  # Don't store huge token lists
            block_boundaries=block_boundaries,
            original_length=original_length,
        )

        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch

            result.input_ids = torch.tensor([result.input_ids])
            result.attention_mask = torch.tensor([result.attention_mask])
        elif return_tensors == "np":
            import numpy as np

            result.input_ids = np.array([result.input_ids])
            result.attention_mask = np.array([result.attention_mask])

        return result

    def _tokenize_block(self, block: NormalizedBlock) -> list[str]:
        """Tokenize a single basic block."""
        tokens = []

        # Block start marker
        if self.config.include_block_markers:
            tokens.append("[BLOCK]")

            # Add block properties
            if block.is_entry:
                tokens.append("[ENTRY]")
            if block.is_exit:
                tokens.append("[EXIT]")
            if block.is_loop_header:
                tokens.append("[LOOP]")

        # Process instructions
        for instr in block.instructions:
            instr_tokens = self._tokenize_instruction(instr)
            tokens.extend(instr_tokens)

        # Block end marker
        if self.config.include_block_markers:
            tokens.append("[/BLOCK]")

        return tokens

    def _tokenize_instruction(self, instr: NormalizedInstruction) -> list[str]:
        """Tokenize a single instruction."""
        tokens = []

        # Opcode
        if self.config.include_opcode:
            tokens.append(instr.opcode)

        # Destination
        if instr.dest:
            tokens.append(self._tokenize_operand(instr.dest))

        # Sources
        for src in instr.src:
            tokens.append(self._tokenize_operand(src))

        return tokens

    def _tokenize_operand(self, operand: NormalizedOperand) -> str:
        """Convert operand to token string."""

        if operand.kind == NormalizedOperand.Kind.REGISTER:
            # Register role token
            return f"${operand.value}"

        elif operand.kind == NormalizedOperand.Kind.TEMP:
            # Temporary
            return f"t{operand.value}"

        elif operand.kind == NormalizedOperand.Kind.IMMEDIATE:
            # Immediate value or abstraction
            return operand.value

        elif operand.kind == NormalizedOperand.Kind.MEMORY:
            # Memory access pattern
            size = operand.size
            parts = []
            if operand.base:
                parts.append(f"${operand.base}")
            if operand.offset:
                parts.append(operand.offset)

            if parts:
                return f"MEM{size}[{'+'.join(parts)}]"
            else:
                return f"MEM{size}[0]"

        elif operand.kind == NormalizedOperand.Kind.ADDRESS:
            return "ADDR"

        elif operand.kind == NormalizedOperand.Kind.FUNCTION:
            return "FUNC"

        else:
            return str(operand.value)

    def batch_tokenize(
        self,
        functions: list[NormalizedFunction],
        return_tensors: Optional[str] = None,
    ) -> dict[str, Union[list, "torch.Tensor", "np.ndarray"]]:
        """Tokenize a batch of functions.

        Args:
            functions: List of normalized functions
            return_tensors: Optional tensor type

        Returns:
            Dictionary with batched input_ids and attention_mask
        """
        results = [self.tokenize(f) for f in functions]

        input_ids = [r.input_ids for r in results]
        attention_mask = [r.attention_mask for r in results]

        if return_tensors == "pt":
            import torch

            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            }
        elif return_tensors == "np":
            import numpy as np

            return {
                "input_ids": np.array(input_ids),
                "attention_mask": np.array(attention_mask),
            }

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to string representation.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded string
        """
        tokens = []
        special = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}

        for tid in token_ids:
            token = self.vocab.decode(tid)
            if skip_special_tokens and token in special:
                continue
            if token == "[PAD]":
                break  # Stop at padding
            tokens.append(token)

        return " ".join(tokens)

    def prepare_for_model(
        self,
        function: NormalizedFunction,
        return_tensors: str = "pt",
    ) -> dict:
        """Prepare tokenized output for model input.

        Args:
            function: Normalized function
            return_tensors: Tensor type ("pt" or "np")

        Returns:
            Dictionary with input_ids, attention_mask, ready for model
        """
        output = self.tokenize(function, return_tensors=return_tensors)

        return {
            "input_ids": output.input_ids,
            "attention_mask": output.attention_mask,
        }

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        """Return padding token ID."""
        return self.vocab.pad_token_id

    @property
    def mask_token_id(self) -> int:
        """Return mask token ID."""
        return self.vocab.mask_token_id

    @property
    def cls_token_id(self) -> int:
        """Return CLS token ID."""
        return self.vocab.cls_token_id

    @property
    def sep_token_id(self) -> int:
        """Return SEP token ID."""
        return self.vocab.sep_token_id

    def encode(
        self,
        function: NormalizedFunction,
        max_length: Optional[int] = None,
    ) -> list[int]:
        """Encode a normalized function to token IDs.

        Convenience method that wraps tokenize() and returns just the token IDs.

        Args:
            function: Normalized function to encode
            max_length: Optional maximum length override

        Returns:
            List of token IDs
        """
        # Temporarily override max_length if provided
        original_max_length = self.config.max_length
        if max_length is not None:
            self.config.max_length = max_length

        try:
            result = self.tokenize(function)
            return result.input_ids
        finally:
            self.config.max_length = original_max_length


def create_tokenizer(
    vocab_path: Optional[str] = None,
    max_length: int = 512,
) -> IRTokenizer:
    """Create a configured tokenizer.

    Args:
        vocab_path: Optional path to saved vocabulary
        max_length: Maximum sequence length

    Returns:
        Configured IRTokenizer
    """
    from pathlib import Path

    if vocab_path:
        vocab = Vocabulary.load(Path(vocab_path))
    else:
        vocab = get_default_vocabulary()

    config = TokenizerConfig(max_length=max_length)
    return IRTokenizer(vocab, config)
