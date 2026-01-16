"""IR Tokenization module for CLAUSTRUM.

Provides tokenization of normalized IR for embedding models:
- Fixed vocabulary of ~50K tokens covering opcodes, operands, and structure
- Special tokens for [MASK], [CLS], [SEP], [PAD], etc.
- Efficient BPE-free tokenization (full instruction tokens as per NDSS BAR 2025)
"""

from claustrum.tokenization.tokenizer import IRTokenizer, TokenizerConfig
from claustrum.tokenization.vocabulary import (
    Vocabulary,
    build_vocabulary,
    SPECIAL_TOKENS,
    OPCODE_TOKENS,
    REGISTER_ROLE_TOKENS,
    IMMEDIATE_TOKENS,
)

__all__ = [
    "IRTokenizer",
    "TokenizerConfig",
    "Vocabulary",
    "build_vocabulary",
    "SPECIAL_TOKENS",
    "OPCODE_TOKENS",
    "REGISTER_ROLE_TOKENS",
    "IMMEDIATE_TOKENS",
]
