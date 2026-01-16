"""Vocabulary definitions for IR tokenization.

Defines the ~50K shared vocabulary covering:
- Special tokens ([PAD], [MASK], [CLS], [SEP], [UNK])
- Structure tokens ([FUNC], [BLOCK], etc.)
- Unified opcodes
- Register roles
- Normalized immediates
- Memory access patterns
- Per-ISA extension tokens (~1000 per ISA)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Special tokens
SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
    "[FUNC]": 5,
    "[/FUNC]": 6,
    "[BLOCK]": 7,
    "[/BLOCK]": 8,
    "[ENTRY]": 9,
    "[EXIT]": 10,
    "[LOOP]": 11,
}

# Unified opcodes (from IROpcode enum)
OPCODE_TOKENS = [
    # Memory
    "LOAD",
    "STORE",
    # Arithmetic
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "SDIV",
    "MOD",
    "SMOD",
    "NEG",
    # Bitwise
    "AND",
    "OR",
    "XOR",
    "NOT",
    "SHL",
    "SHR",
    "SAR",
    "ROL",
    "ROR",
    # Comparison
    "CMP_EQ",
    "CMP_NE",
    "CMP_LT",
    "CMP_LE",
    "CMP_ULT",
    "CMP_ULE",
    # Control flow
    "BRANCH",
    "JUMP",
    "CALL",
    "RET",
    # Data movement
    "MOV",
    "ZEXT",
    "SEXT",
    "TRUNC",
    # Floating point
    "FADD",
    "FSUB",
    "FMUL",
    "FDIV",
    "FCMP",
    "FCVT",
    # Special
    "NOP",
    "UNKNOWN",
    "PHI",
    "CPUID",
    "SYSCALL",
    "INT",
    "HLT",
    "CMPXCHG",
    "XCHG",
    # SIMD
    "VEC_ADD",
    "VEC_MUL",
    "VEC_LOAD",
    "VEC_STORE",
]

# Register role tokens
REGISTER_ROLE_TOKENS = [
    # Arguments
    "$ARG0",
    "$ARG1",
    "$ARG2",
    "$ARG3",
    "$ARG4",
    "$ARG5",
    "$ARG6",
    "$ARG7",
    # Return
    "$RET",
    "$RET_HI",
    # Stack/frame
    "$SP",
    "$FP",
    "$LR",
    "$PC",
    # FP arguments/return
    "$FP_ARG0",
    "$FP_ARG1",
    "$FP_ARG2",
    "$FP_ARG3",
    "$FP_RET",
    # Flags and general
    "$FLAGS",
    "$GENERAL",
    # Temporaries (up to 100)
    *[f"t{i}" for i in range(100)],
]

# Immediate tokens
IMMEDIATE_TOKENS = [
    # Small immediates (0-255 preserved)
    *[str(i) for i in range(256)],
    # Small negative
    *[str(i) for i in range(-128, 0)],
    # Size-based abstractions
    "IMM16",
    "IMM32",
    "IMM64",
    # Special values
    "ADDR",
    "FUNC",
]

# Memory access tokens
MEMORY_TOKENS = [
    # Size prefixes
    "MEM8",
    "MEM16",
    "MEM32",
    "MEM64",
    "MEM128",
    "MEM256",
    "MEM512",
    # Common patterns
    *[f"MEM{s}[${r}]" for s in [8, 16, 32, 64] for r in ["SP", "FP", "ARG0", "ARG1"]],
    *[
        f"MEM{s}[${r}+{o}]"
        for s in [8, 16, 32, 64]
        for r in ["SP", "FP"]
        for o in ["IMM16", "IMM32"]
    ],
]

# Block reference tokens
BLOCK_TOKENS = [
    *[f"BLOCK_{i}" for i in range(1000)],  # Up to 1000 blocks per function
]

# ISA-specific extension tokens (for opcodes that don't map cleanly)
ISA_EXTENSION_TOKENS: dict[str, list[str]] = {
    "x86": [
        "REP",
        "REPNE",
        "LOCK",
        "PUSH",
        "POP",
        "PUSHF",
        "POPF",
        "LEA",
        "XCHG",
        "BSWAP",
        "CMOV",
        "SETCC",
        "BT",
        "BTS",
        "BTR",
        "BTC",
        "BSF",
        "BSR",
        "POPCNT",
        "LZCNT",
        "TZCNT",
        "MOVS",
        "STOS",
        "LODS",
        "SCAS",
        "CMPS",
        "IN",
        "OUT",
        "INS",
        "OUTS",
    ],
    "arm": [
        "LDM",
        "STM",
        "PUSH",
        "POP",
        "BX",
        "BLX",
        "CBZ",
        "CBNZ",
        "IT",
        "ITE",
        "ITT",
        "UXTB",
        "UXTH",
        "SXTB",
        "SXTH",
        "REV",
        "REV16",
        "REVSH",
        "RBIT",
        "CLZ",
        "MLA",
        "MLS",
        "UMULL",
        "SMULL",
        "ADC",
        "SBC",
        "RSB",
        "RSC",
        "TST",
        "TEQ",
        "CMP",
        "CMN",
    ],
    "mips": [
        "LUI",
        "ORI",
        "ADDIU",
        "LW",
        "SW",
        "LB",
        "SB",
        "LH",
        "SH",
        "BEQ",
        "BNE",
        "BGEZ",
        "BGTZ",
        "BLEZ",
        "BLTZ",
        "JAL",
        "JALR",
        "JR",
        "SLT",
        "SLTU",
        "SLTI",
        "SLTIU",
        "MFHI",
        "MFLO",
        "MTHI",
        "MTLO",
        "MULT",
        "MULTU",
        "DIV",
        "DIVU",
        "SYNC",
        "BREAK",
        "SYSCALL",
    ],
    "riscv": [
        "LUI",
        "AUIPC",
        "JAL",
        "JALR",
        "BEQ",
        "BNE",
        "BLT",
        "BGE",
        "BLTU",
        "BGEU",
        "LB",
        "LH",
        "LW",
        "LD",
        "SB",
        "SH",
        "SW",
        "SD",
        "ADDI",
        "SLTI",
        "SLTIU",
        "XORI",
        "ORI",
        "ANDI",
        "SLLI",
        "SRLI",
        "SRAI",
        "ADD",
        "SUB",
        "SLL",
        "SLT",
        "SLTU",
        "XOR",
        "SRL",
        "SRA",
        "OR",
        "AND",
        "FENCE",
        "ECALL",
        "EBREAK",
    ],
    "ppc": [
        "LWZ",
        "STW",
        "LBZ",
        "STB",
        "LHZ",
        "STH",
        "ADDI",
        "ADDIS",
        "ADDIC",
        "BL",
        "BLR",
        "BCTR",
        "BCTRL",
        "CMP",
        "CMPI",
        "CMPL",
        "CMPLI",
        "MFLR",
        "MTLR",
        "MFCTR",
        "MTCTR",
        "RLWINM",
        "RLWIMI",
        "EXTSB",
        "EXTSH",
        "EXTSW",
        "CNTLZW",
        "CNTLZD",
    ],
}


@dataclass
class Vocabulary:
    """Token vocabulary for IR tokenization.

    Manages the mapping between tokens and integer IDs,
    with support for saving/loading and dynamic extension.
    """

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)

    # Vocabulary metadata
    version: str = "1.0"
    size: int = 0

    # Special token IDs
    pad_token_id: int = 0
    unk_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    mask_token_id: int = 4

    def __post_init__(self):
        if not self.token_to_id:
            self._build_vocabulary()

    def _build_vocabulary(self) -> None:
        """Build the full vocabulary."""
        current_id = 0

        # Special tokens first
        for token, tid in SPECIAL_TOKENS.items():
            self.token_to_id[token] = tid
            self.id_to_token[tid] = token
            current_id = max(current_id, tid + 1)

        # Opcodes
        for opcode in OPCODE_TOKENS:
            if opcode not in self.token_to_id:
                self.token_to_id[opcode] = current_id
                self.id_to_token[current_id] = opcode
                current_id += 1

        # Register roles
        for reg in REGISTER_ROLE_TOKENS:
            if reg not in self.token_to_id:
                self.token_to_id[reg] = current_id
                self.id_to_token[current_id] = reg
                current_id += 1

        # Immediates
        for imm in IMMEDIATE_TOKENS:
            if imm not in self.token_to_id:
                self.token_to_id[imm] = current_id
                self.id_to_token[current_id] = imm
                current_id += 1

        # Memory patterns
        for mem in MEMORY_TOKENS:
            if mem not in self.token_to_id:
                self.token_to_id[mem] = current_id
                self.id_to_token[current_id] = mem
                current_id += 1

        # Block references
        for block in BLOCK_TOKENS:
            if block not in self.token_to_id:
                self.token_to_id[block] = current_id
                self.id_to_token[current_id] = block
                current_id += 1

        # ISA extensions
        for isa, tokens in ISA_EXTENSION_TOKENS.items():
            for token in tokens:
                prefixed = f"{isa.upper()}_{token}"
                if prefixed not in self.token_to_id:
                    self.token_to_id[prefixed] = current_id
                    self.id_to_token[current_id] = prefixed
                    current_id += 1

        self.size = current_id

    def encode(self, token: str) -> int:
        """Encode a token to its ID."""
        return self.token_to_id.get(token, self.unk_token_id)

    def decode(self, token_id: int) -> str:
        """Decode a token ID to its string representation."""
        return self.id_to_token.get(token_id, "[UNK]")

    def encode_batch(self, tokens: list[str]) -> list[int]:
        """Encode a batch of tokens."""
        return [self.encode(t) for t in tokens]

    def decode_batch(self, token_ids: list[int]) -> list[str]:
        """Decode a batch of token IDs."""
        return [self.decode(t) for t in token_ids]

    def add_token(self, token: str) -> int:
        """Add a new token to the vocabulary.

        Returns the token ID (existing or new).
        """
        if token in self.token_to_id:
            return self.token_to_id[token]

        new_id = self.size
        self.token_to_id[token] = new_id
        self.id_to_token[new_id] = token
        self.size += 1
        return new_id

    def __len__(self) -> int:
        return self.size

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def save(self, path: Path) -> None:
        """Save vocabulary to JSON file."""
        data = {
            "version": self.version,
            "size": self.size,
            "special_tokens": {
                "pad": self.pad_token_id,
                "unk": self.unk_token_id,
                "cls": self.cls_token_id,
                "sep": self.sep_token_id,
                "mask": self.mask_token_id,
            },
            "tokens": self.token_to_id,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Load vocabulary from JSON file."""
        with open(path) as f:
            data = json.load(f)

        vocab = cls()
        vocab.version = data.get("version", "1.0")
        vocab.token_to_id = data["tokens"]
        vocab.id_to_token = {int(v): k for k, v in vocab.token_to_id.items()}
        vocab.size = len(vocab.token_to_id)

        special = data.get("special_tokens", {})
        vocab.pad_token_id = special.get("pad", 0)
        vocab.unk_token_id = special.get("unk", 1)
        vocab.cls_token_id = special.get("cls", 2)
        vocab.sep_token_id = special.get("sep", 3)
        vocab.mask_token_id = special.get("mask", 4)

        return vocab


def build_vocabulary(
    include_isa_extensions: Optional[list[str]] = None,
) -> Vocabulary:
    """Build the full vocabulary.

    Args:
        include_isa_extensions: List of ISAs to include extensions for.
                               If None, includes all.

    Returns:
        Configured Vocabulary instance
    """
    vocab = Vocabulary()

    # If specific ISAs requested, only include those extensions
    if include_isa_extensions is not None:
        for isa in include_isa_extensions:
            isa_lower = isa.lower()
            if isa_lower in ISA_EXTENSION_TOKENS:
                for token in ISA_EXTENSION_TOKENS[isa_lower]:
                    vocab.add_token(f"{isa.upper()}_{token}")

    return vocab


# Default vocabulary instance
_default_vocab: Optional[Vocabulary] = None


def get_default_vocabulary() -> Vocabulary:
    """Get the default vocabulary instance (lazy initialization)."""
    global _default_vocab
    if _default_vocab is None:
        _default_vocab = Vocabulary()
    return _default_vocab
