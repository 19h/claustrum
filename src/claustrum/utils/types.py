"""Core type definitions for CLAUSTRUM.

This module defines the fundamental data types used throughout the system,
including ISA enumeration, architecture specifications, and binary/function
metadata structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Any


class ISA(str, Enum):
    """Instruction Set Architecture enumeration.

    Covers the 30+ ISAs supported by the system, organized into tiers
    based on lifting backend support and training data availability.
    """

    # Tier 1: VEX IR primary support (best quality)
    X86 = "x86"
    X86_64 = "x86_64"
    ARM32 = "arm32"
    ARM64 = "arm64"
    MIPS32 = "mips32"
    MIPS64 = "mips64"
    PPC32 = "ppc32"
    PPC64 = "ppc64"

    # Tier 2: VEX + ESIL support
    RISCV32 = "riscv32"
    RISCV64 = "riscv64"
    S390X = "s390x"
    SPARC32 = "sparc32"
    SPARC64 = "sparc64"

    # Tier 3: ESIL/P-Code primary support
    AVR = "avr"
    MSP430 = "msp430"
    XTENSA = "xtensa"
    ARC = "arc"
    TRICORE = "tricore"
    M68K = "m68k"
    SH4 = "sh4"  # SuperH
    BLACKFIN = "blackfin"

    # Tier 4: P-Code only or limited support
    HEXAGON = "hexagon"  # VLIW
    LOONGARCH = "loongarch"
    VAX = "vax"
    PIC = "pic"
    Z80 = "z80"
    M6502 = "6502"

    # Stack machines (special handling)
    JVM = "jvm"
    DALVIK = "dalvik"
    WASM = "wasm"
    EBPF = "ebpf"

    @property
    def tier(self) -> int:
        """Return the ISA tier (1-4) based on lifting support quality."""
        tier1 = {
            self.X86,
            self.X86_64,
            self.ARM32,
            self.ARM64,
            self.MIPS32,
            self.MIPS64,
            self.PPC32,
            self.PPC64,
        }
        tier2 = {self.RISCV32, self.RISCV64, self.S390X, self.SPARC32, self.SPARC64}
        tier3 = {
            self.AVR,
            self.MSP430,
            self.XTENSA,
            self.ARC,
            self.TRICORE,
            self.M68K,
            self.SH4,
            self.BLACKFIN,
        }

        if self in tier1:
            return 1
        elif self in tier2:
            return 2
        elif self in tier3:
            return 3
        else:
            return 4

    @property
    def is_stack_machine(self) -> bool:
        """Check if this ISA uses stack-based execution model."""
        return self in {self.JVM, self.DALVIK, self.WASM, self.EBPF}

    @property
    def is_vliw(self) -> bool:
        """Check if this is a VLIW architecture."""
        return self in {self.HEXAGON}

    @property
    def has_delay_slots(self) -> bool:
        """Check if this architecture has branch delay slots."""
        return self in {self.MIPS32, self.MIPS64, self.SPARC32, self.SPARC64}

    @property
    def has_predication(self) -> bool:
        """Check if this architecture supports predicated instructions."""
        return self in {self.ARM32}  # ARM64 uses conditional branches instead

    @property
    def word_size(self) -> int:
        """Return the native word size in bits."""
        is_64bit = self in {
            self.X86_64,
            self.ARM64,
            self.MIPS64,
            self.PPC64,
            self.RISCV64,
            self.S390X,
            self.SPARC64,
            self.LOONGARCH,
        }
        return 64 if is_64bit else 32

    @property
    def endianness(self) -> str:
        """Return default endianness ('little' or 'big')."""
        big_endian = {
            self.PPC32,
            self.PPC64,
            self.S390X,
            self.SPARC32,
            self.SPARC64,
            self.M68K,
            self.JVM,
            self.DALVIK,
        }
        # Note: Many ISAs are bi-endian, this is just the common default
        return "big" if self in big_endian else "little"


class RegisterRole(str, Enum):
    """Abstract register roles for architecture-neutral representation.

    Based on common calling conventions, we abstract concrete registers
    to role-based identifiers enabling cross-architecture comparison.
    """

    # Function arguments (position-based)
    ARG0 = "ARG0"
    ARG1 = "ARG1"
    ARG2 = "ARG2"
    ARG3 = "ARG3"
    ARG4 = "ARG4"
    ARG5 = "ARG5"
    ARG6 = "ARG6"
    ARG7 = "ARG7"

    # Return value
    RET = "RET"
    RET_HI = "RET_HI"  # For 64-bit returns on 32-bit architectures

    # Stack management
    SP = "SP"  # Stack pointer
    FP = "FP"  # Frame pointer
    LR = "LR"  # Link register (return address)

    # Program counter (rarely used in IR)
    PC = "PC"

    # General purpose (for non-ABI registers)
    GENERAL = "GENERAL"

    # Floating point
    FP_ARG0 = "FP_ARG0"
    FP_ARG1 = "FP_ARG1"
    FP_ARG2 = "FP_ARG2"
    FP_ARG3 = "FP_ARG3"
    FP_RET = "FP_RET"

    # Status/flags
    FLAGS = "FLAGS"

    # Temporaries (IR-specific)
    TEMP = "TEMP"


class MemoryAccessSize(int, Enum):
    """Memory access sizes for IR normalization."""

    MEM8 = 8
    MEM16 = 16
    MEM32 = 32
    MEM64 = 64
    MEM128 = 128  # SIMD
    MEM256 = 256  # AVX
    MEM512 = 512  # AVX-512


class CallingConvention(str, Enum):
    """Common calling conventions across architectures."""

    # x86 conventions
    CDECL = "cdecl"
    STDCALL = "stdcall"
    FASTCALL = "fastcall"
    THISCALL = "thiscall"

    # x86-64 conventions
    SYSV_AMD64 = "sysv_amd64"
    MS_X64 = "ms_x64"

    # ARM conventions
    AAPCS = "aapcs"  # ARM 32-bit
    AAPCS64 = "aapcs64"  # ARM 64-bit

    # MIPS conventions
    O32 = "o32"
    N32 = "n32"
    N64 = "n64"

    # RISC-V conventions
    RV_ILP32 = "rv_ilp32"
    RV_LP64 = "rv_lp64"

    # Generic
    UNKNOWN = "unknown"


class Architecture(str, Enum):
    """Higher-level architecture families for grouping."""

    X86 = "x86"
    ARM = "arm"
    MIPS = "mips"
    PPC = "ppc"
    RISCV = "riscv"
    SPARC = "sparc"
    EMBEDDED = "embedded"  # AVR, MSP430, etc.
    STACK = "stack"  # JVM, WASM, etc.
    OTHER = "other"


@dataclass
class BinaryInfo:
    """Metadata about a binary file."""

    path: Path
    sha256: str
    isa: ISA
    architecture: Architecture
    endianness: str
    word_size: int
    file_format: str  # ELF, PE, Mach-O, raw
    base_address: int
    entry_point: int
    sections: list[dict[str, Any]] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)

    # Optional metadata
    compiler: Optional[str] = None
    optimization_level: Optional[str] = None  # O0, O1, O2, O3, Os
    stripped: bool = True
    pic: bool = False  # Position independent code

    @property
    def is_64bit(self) -> bool:
        return self.word_size == 64


@dataclass
class FunctionInfo:
    """Metadata about a function within a binary."""

    binary_hash: str
    address: int
    size: int
    name: Optional[str] = None

    # Control flow information
    num_basic_blocks: int = 0
    num_edges: int = 0
    num_instructions: int = 0

    # Calling information
    calling_convention: CallingConvention = CallingConvention.UNKNOWN
    num_args: int = 0
    has_return: bool = True

    # Graph properties
    cyclomatic_complexity: int = 0
    max_loop_depth: int = 0

    # References
    callees: list[int] = field(default_factory=list)
    callers: list[int] = field(default_factory=list)
    string_refs: list[str] = field(default_factory=list)

    # Source provenance (for training data)
    source_file: Optional[str] = None
    source_function: Optional[str] = None

    @property
    def function_id(self) -> str:
        """Unique identifier for this function."""
        return f"{self.binary_hash}:{self.address:016x}"


# ISA similarity for curriculum learning
ISA_SIMILARITY_GROUPS = {
    # Same architecture family (easiest pairs)
    "arm_family": {ISA.ARM32, ISA.ARM64},
    "x86_family": {ISA.X86, ISA.X86_64},
    "mips_family": {ISA.MIPS32, ISA.MIPS64},
    "ppc_family": {ISA.PPC32, ISA.PPC64},
    "riscv_family": {ISA.RISCV32, ISA.RISCV64},
    "sparc_family": {ISA.SPARC32, ISA.SPARC64},
    # RISC architectures (medium difficulty)
    "risc_64": {ISA.ARM64, ISA.RISCV64, ISA.MIPS64, ISA.PPC64},
    "risc_32": {ISA.ARM32, ISA.RISCV32, ISA.MIPS32, ISA.PPC32},
    # Embedded architectures
    "embedded": {ISA.AVR, ISA.MSP430, ISA.XTENSA, ISA.PIC},
    # Stack machines (hardest, fundamentally different)
    "stack": {ISA.JVM, ISA.DALVIK, ISA.WASM, ISA.EBPF},
}


def get_isa_similarity_score(isa1: ISA, isa2: ISA) -> float:
    """Compute similarity score between two ISAs for curriculum learning.

    Returns a score from 0.0 (completely different) to 1.0 (same ISA).
    """
    if isa1 == isa2:
        return 1.0

    # Same family (e.g., ARM32 <-> ARM64)
    for group in ISA_SIMILARITY_GROUPS.values():
        if isa1 in group and isa2 in group:
            return 0.8

    # Both RISC or both CISC
    cisc = {ISA.X86, ISA.X86_64}
    risc = {
        ISA.ARM32,
        ISA.ARM64,
        ISA.MIPS32,
        ISA.MIPS64,
        ISA.RISCV32,
        ISA.RISCV64,
        ISA.PPC32,
        ISA.PPC64,
    }

    if isa1 in cisc and isa2 in cisc:
        return 0.7
    if isa1 in risc and isa2 in risc:
        return 0.6

    # CISC <-> RISC
    if (isa1 in cisc and isa2 in risc) or (isa1 in risc and isa2 in cisc):
        return 0.4

    # Stack machine vs register machine (hardest)
    stack = {ISA.JVM, ISA.DALVIK, ISA.WASM, ISA.EBPF}
    if (isa1 in stack) != (isa2 in stack):
        return 0.2

    return 0.3  # Default for other pairs
