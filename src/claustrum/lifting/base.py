"""Base classes and data structures for IR lifting.

Defines the abstract lifter interface and unified IR data structures
that all backends (VEX, ESIL, P-Code) produce.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, Union


class IROpcode(str, Enum):
    """Unified IR operation codes.

    A minimal but complete set of operations that can represent
    computations across all supported architectures.
    """

    # Memory operations
    LOAD = "LOAD"  # Load from memory
    STORE = "STORE"  # Store to memory

    # Arithmetic operations
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    SDIV = "SDIV"  # Signed division
    MOD = "MOD"
    SMOD = "SMOD"  # Signed modulo
    NEG = "NEG"  # Negation

    # Bitwise operations
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NOT = "NOT"
    SHL = "SHL"  # Shift left
    SHR = "SHR"  # Logical shift right
    SAR = "SAR"  # Arithmetic shift right (preserves sign)
    ROL = "ROL"  # Rotate left
    ROR = "ROR"  # Rotate right

    # Comparison operations
    CMP_EQ = "CMP_EQ"  # Equal
    CMP_NE = "CMP_NE"  # Not equal
    CMP_LT = "CMP_LT"  # Less than (signed)
    CMP_LE = "CMP_LE"  # Less than or equal (signed)
    CMP_ULT = "CMP_ULT"  # Less than (unsigned)
    CMP_ULE = "CMP_ULE"  # Less than or equal (unsigned)

    # Control flow
    BRANCH = "BRANCH"  # Conditional branch
    JUMP = "JUMP"  # Unconditional jump
    CALL = "CALL"  # Function call
    RET = "RET"  # Return

    # Data movement
    MOV = "MOV"  # Copy value
    ZEXT = "ZEXT"  # Zero extend
    SEXT = "SEXT"  # Sign extend
    TRUNC = "TRUNC"  # Truncate

    # Floating point
    FADD = "FADD"
    FSUB = "FSUB"
    FMUL = "FMUL"
    FDIV = "FDIV"
    FCMP = "FCMP"
    FCVT = "FCVT"  # Float conversion

    # Special
    NOP = "NOP"
    UNKNOWN = "UNKNOWN"  # Unhandled instruction
    PHI = "PHI"  # SSA phi node

    # Architecture-specific (for operations that don't map cleanly)
    CPUID = "CPUID"
    SYSCALL = "SYSCALL"
    INT = "INT"  # Software interrupt
    HLT = "HLT"  # Halt

    # Atomic operations
    CMPXCHG = "CMPXCHG"  # Compare and exchange
    XCHG = "XCHG"  # Exchange

    # SIMD/Vector (simplified)
    VEC_ADD = "VEC_ADD"
    VEC_MUL = "VEC_MUL"
    VEC_LOAD = "VEC_LOAD"
    VEC_STORE = "VEC_STORE"


@dataclass
class IROperand:
    """An operand in an IR instruction.

    Operands can be:
    - Registers (architecture-specific or abstract)
    - Temporaries (IR-specific)
    - Immediates (constants)
    - Memory references
    """

    class Kind(str, Enum):
        REGISTER = "reg"
        TEMPORARY = "tmp"
        IMMEDIATE = "imm"
        MEMORY = "mem"
        LABEL = "label"

    kind: Kind
    value: Any
    size: int = 64  # Size in bits

    # For memory operands
    base: Optional[str] = None
    index: Optional[str] = None
    scale: int = 1
    offset: int = 0

    @classmethod
    def reg(cls, name: str, size: int = 64) -> "IROperand":
        """Create a register operand."""
        return cls(kind=cls.Kind.REGISTER, value=name, size=size)

    @classmethod
    def tmp(cls, idx: int, size: int = 64) -> "IROperand":
        """Create a temporary operand."""
        return cls(kind=cls.Kind.TEMPORARY, value=f"t{idx}", size=size)

    @classmethod
    def imm(cls, value: int, size: int = 64) -> "IROperand":
        """Create an immediate operand."""
        return cls(kind=cls.Kind.IMMEDIATE, value=value, size=size)

    @classmethod
    def mem(
        cls,
        base: Optional[str] = None,
        offset: int = 0,
        index: Optional[str] = None,
        scale: int = 1,
        size: int = 64,
    ) -> "IROperand":
        """Create a memory operand."""
        return cls(
            kind=cls.Kind.MEMORY,
            value=None,
            size=size,
            base=base,
            index=index,
            scale=scale,
            offset=offset,
        )

    @classmethod
    def label(cls, target: Union[int, str]) -> "IROperand":
        """Create a label operand (for branches/calls)."""
        return cls(kind=cls.Kind.LABEL, value=target)

    def __repr__(self) -> str:
        if self.kind == self.Kind.REGISTER:
            return f"${self.value}:{self.size}"
        elif self.kind == self.Kind.TEMPORARY:
            return f"{self.value}:{self.size}"
        elif self.kind == self.Kind.IMMEDIATE:
            if isinstance(self.value, int):
                return f"0x{self.value:x}" if self.value >= 16 else str(self.value)
            return str(self.value)
        elif self.kind == self.Kind.MEMORY:
            parts = []
            if self.base:
                parts.append(f"${self.base}")
            if self.index:
                if self.scale != 1:
                    parts.append(f"${self.index}*{self.scale}")
                else:
                    parts.append(f"${self.index}")
            if self.offset:
                parts.append(f"{self.offset:+d}")
            return f"MEM{self.size}[{'+'.join(parts) if parts else '0'}]"
        else:
            return f"@{self.value}"


@dataclass
class IRInstruction:
    """A single instruction in unified IR.

    Each instruction has an opcode, optional destination, and source operands.
    The instruction maps back to the original address for debugging/analysis.
    """

    opcode: IROpcode
    dest: Optional[IROperand] = None
    src: list[IROperand] = field(default_factory=list)

    # Original instruction info
    address: int = 0
    size: int = 0  # Original instruction size in bytes

    # Additional metadata
    condition: Optional[str] = None  # For conditional branches
    comment: Optional[str] = None  # Original disassembly or note

    def __repr__(self) -> str:
        parts = [f"{self.opcode.value}"]
        if self.dest:
            parts.append(f"{self.dest}")
        if self.src:
            parts.append(", ".join(str(s) for s in self.src))
        if self.condition:
            parts.insert(1, f"[{self.condition}]")
        return " ".join(parts)


@dataclass
class LiftedBlock:
    """A basic block of lifted IR instructions.

    Basic blocks have single entry (first instruction) and single exit
    (last instruction), with no internal branches.
    """

    address: int
    size: int
    instructions: list[IRInstruction] = field(default_factory=list)

    # CFG information
    successors: list[int] = field(default_factory=list)  # Addresses of successor blocks
    predecessors: list[int] = field(default_factory=list)  # Addresses of predecessor blocks

    # Block properties
    is_entry: bool = False
    is_exit: bool = False
    is_call_site: bool = False

    @property
    def num_instructions(self) -> int:
        return len(self.instructions)

    @property
    def end_address(self) -> int:
        return self.address + self.size

    def __repr__(self) -> str:
        return f"Block(0x{self.address:x}, {self.num_instructions} instrs)"


@dataclass
class LiftedFunction:
    """A lifted function containing IR blocks and CFG.

    This is the primary output of the lifting process, containing
    normalized IR and control flow graph for embedding generation.
    """

    address: int
    size: int
    name: Optional[str] = None

    # IR content
    blocks: dict[int, LiftedBlock] = field(default_factory=dict)  # addr -> block
    entry_block: Optional[int] = None

    # CFG as adjacency list
    cfg_edges: list[tuple[int, int]] = field(default_factory=list)

    # Function metadata
    num_args: int = 0
    return_type: Optional[str] = None
    calling_convention: Optional[str] = None

    # Original binary info
    binary_hash: Optional[str] = None
    isa: Optional[str] = None

    # Lifting metadata
    lifter_backend: Optional[str] = None
    lift_time_ms: float = 0.0

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_edges(self) -> int:
        return len(self.cfg_edges)

    @property
    def num_instructions(self) -> int:
        return sum(b.num_instructions for b in self.blocks.values())

    @property
    def cyclomatic_complexity(self) -> int:
        """Calculate McCabe cyclomatic complexity: E - N + 2"""
        return self.num_edges - self.num_blocks + 2

    def get_block_at(self, addr: int) -> Optional[LiftedBlock]:
        """Get block at specific address."""
        return self.blocks.get(addr)

    def iter_instructions(self):
        """Iterate over all instructions in address order."""
        for addr in sorted(self.blocks.keys()):
            for instr in self.blocks[addr].instructions:
                yield instr

    def to_ir_string(self) -> str:
        """Convert to human-readable IR string."""
        lines = [f"function {self.name or f'sub_{self.address:x}'} @ 0x{self.address:x}:"]

        for addr in sorted(self.blocks.keys()):
            block = self.blocks[addr]
            lines.append(f"\n  block_0x{addr:x}:")
            for instr in block.instructions:
                lines.append(f"    {instr}")
            if block.successors:
                succs = ", ".join(f"0x{s:x}" for s in block.successors)
                lines.append(f"    -> [{succs}]")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"LiftedFunction({self.name or f'sub_{self.address:x}'}, {self.num_blocks} blocks, {self.num_instructions} instrs)"


class IRLifter(ABC):
    """Abstract base class for IR lifters.

    All lifting backends (VEX, ESIL, P-Code) implement this interface.
    """

    def __init__(self):
        self.name: str = "base"
        self.supported_isas: set[str] = set()

    @abstractmethod
    def lift_function(
        self,
        binary_path: Path,
        function_addr: int,
        function_size: Optional[int] = None,
    ) -> LiftedFunction:
        """Lift a single function to IR.

        Args:
            binary_path: Path to the binary file
            function_addr: Address of the function start
            function_size: Optional size hint

        Returns:
            LiftedFunction with IR and CFG
        """
        pass

    @abstractmethod
    def lift_block(
        self,
        binary_path: Path,
        block_addr: int,
        max_size: int = 4096,
    ) -> LiftedBlock:
        """Lift a single basic block.

        Args:
            binary_path: Path to the binary file
            block_addr: Address of the block start
            max_size: Maximum size to lift

        Returns:
            LiftedBlock with IR instructions
        """
        pass

    def supports_isa(self, isa: str) -> bool:
        """Check if this lifter supports the given ISA."""
        return isa.lower() in self.supported_isas

    def get_priority(self, isa: str) -> int:
        """Get lifting priority for ISA (higher = preferred)."""
        return 1 if self.supports_isa(isa) else 0


class LiftingError(Exception):
    """Raised when IR lifting fails."""

    def __init__(
        self,
        message: str,
        address: Optional[int] = None,
        backend: Optional[str] = None,
    ):
        self.address = address
        self.backend = backend
        super().__init__(message)
