"""IR Normalization module for CLAUSTRUM.

This module implements architecture-neutral normalization of lifted IR,
applying transformations that enable cross-ISA comparison:

- Constant propagation and folding
- Register role abstraction (ARG0, RET, SP, etc.)
- Memory operand canonicalization
- Address and immediate normalization
- Dead code elimination
"""

from claustrum.normalization.normalizer import IRNormalizer, NormalizationConfig
from claustrum.normalization.normalized_ir import (
    NormalizedIR,
    NormalizedInstruction,
    NormalizedBlock,
    NormalizedFunction,
)
from claustrum.normalization.register_mapping import (
    RegisterMapper,
    CALLING_CONVENTIONS,
)

__all__ = [
    "IRNormalizer",
    "NormalizationConfig",
    "NormalizedIR",
    "NormalizedInstruction",
    "NormalizedBlock",
    "NormalizedFunction",
    "RegisterMapper",
    "CALLING_CONVENTIONS",
]
