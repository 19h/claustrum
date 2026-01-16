"""Normalized IR data structures.

Defines the normalized representation of IR instructions, blocks, and functions
after applying architecture-neutral transformations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class NormalizedOperand:
    """A normalized IR operand.

    After normalization:
    - Registers use role-based names (ARG0, RET, SP, etc.)
    - Large immediates are abstracted (IMM16, IMM32, IMM64)
    - Memory operands use canonical form MEM[base+offset]
    - Addresses are replaced with ADDR token
    """

    class Kind(str):
        REGISTER = "REG"
        IMMEDIATE = "IMM"
        MEMORY = "MEM"
        ADDRESS = "ADDR"
        FUNCTION = "FUNC"
        TEMP = "TMP"

    kind: str
    value: str
    size: int = 64

    # For memory operands
    base: Optional[str] = None
    offset: Optional[str] = None  # Can be register or normalized immediate

    def __str__(self) -> str:
        if self.kind == self.Kind.MEMORY:
            parts = []
            if self.base:
                parts.append(self.base)
            if self.offset:
                parts.append(self.offset)
            return f"MEM{self.size}[{'+'.join(parts) if parts else '0'}]"
        elif self.kind == self.Kind.REGISTER:
            return f"${self.value}"
        elif self.kind == self.Kind.IMMEDIATE:
            return self.value
        elif self.kind == self.Kind.TEMP:
            return f"t{self.value}"
        else:
            return self.value

    @classmethod
    def register(cls, role: str, size: int = 64) -> "NormalizedOperand":
        """Create a register operand with role-based name."""
        return cls(kind=cls.Kind.REGISTER, value=role, size=size)

    @classmethod
    def immediate(cls, value: str, size: int = 64) -> "NormalizedOperand":
        """Create an immediate operand (normalized)."""
        return cls(kind=cls.Kind.IMMEDIATE, value=value, size=size)

    @classmethod
    def memory(
        cls,
        base: Optional[str] = None,
        offset: Optional[str] = None,
        size: int = 64,
    ) -> "NormalizedOperand":
        """Create a memory operand."""
        return cls(kind=cls.Kind.MEMORY, value="", size=size, base=base, offset=offset)

    @classmethod
    def address(cls) -> "NormalizedOperand":
        """Create an address placeholder."""
        return cls(kind=cls.Kind.ADDRESS, value="ADDR")

    @classmethod
    def function(cls) -> "NormalizedOperand":
        """Create a function call target placeholder."""
        return cls(kind=cls.Kind.FUNCTION, value="FUNC")

    @classmethod
    def temp(cls, idx: int, size: int = 64) -> "NormalizedOperand":
        """Create a temporary operand."""
        return cls(kind=cls.Kind.TEMP, value=str(idx), size=size)


@dataclass
class NormalizedInstruction:
    """A normalized IR instruction.

    Represents a single operation in the normalized representation,
    with architecture-independent opcode and operands.
    """

    opcode: str  # Unified opcode (ADD, SUB, LOAD, etc.)
    dest: Optional[NormalizedOperand] = None
    src: list[NormalizedOperand] = field(default_factory=list)

    # Original instruction info (for debugging/analysis)
    original_address: int = 0
    original_size: int = 0
    original_asm: Optional[str] = None

    def __str__(self) -> str:
        parts = [self.opcode]
        if self.dest:
            parts.append(str(self.dest))
        if self.src:
            parts.append(", ".join(str(s) for s in self.src))
        return " ".join(parts)

    def to_tokens(self) -> list[str]:
        """Convert instruction to token sequence for embedding."""
        tokens = [self.opcode]
        if self.dest:
            tokens.append(str(self.dest))
        for src in self.src:
            tokens.append(str(src))
        return tokens


@dataclass
class NormalizedBlock:
    """A normalized basic block."""

    block_id: int  # Sequential block ID (not address)
    instructions: list[NormalizedInstruction] = field(default_factory=list)

    # CFG information (using block IDs, not addresses)
    successors: list[int] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)

    # Block properties
    is_entry: bool = False
    is_exit: bool = False
    is_loop_header: bool = False
    loop_depth: int = 0

    # Original address (for reference)
    original_address: int = 0

    @property
    def num_instructions(self) -> int:
        return len(self.instructions)

    def to_tokens(self) -> list[str]:
        """Convert block to token sequence."""
        tokens = [f"[BLOCK_{self.block_id}]"]
        for instr in self.instructions:
            tokens.extend(instr.to_tokens())
        return tokens


@dataclass
class NormalizedFunction:
    """A normalized function representation.

    This is the primary output of the normalization pipeline,
    ready for tokenization and embedding.
    """

    blocks: list[NormalizedBlock] = field(default_factory=list)

    # CFG as adjacency list (block_id -> block_id)
    cfg_edges: list[tuple[int, int]] = field(default_factory=list)

    # Function metadata
    num_args: int = 0
    has_return: bool = True

    # Source information
    original_address: int = 0
    original_name: Optional[str] = None
    isa: Optional[str] = None
    binary_hash: Optional[str] = None
    source_function: Optional[str] = None  # For training data

    # Normalization metadata
    normalization_version: str = "1.0"

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_edges(self) -> int:
        return len(self.cfg_edges)

    @property
    def num_instructions(self) -> int:
        return sum(b.num_instructions for b in self.blocks)

    @property
    def cyclomatic_complexity(self) -> int:
        """McCabe cyclomatic complexity: E - N + 2"""
        return self.num_edges - self.num_blocks + 2

    def get_block_by_id(self, block_id: int) -> Optional[NormalizedBlock]:
        """Get block by ID."""
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None

    def get_entry_block(self) -> Optional[NormalizedBlock]:
        """Get the entry block."""
        for block in self.blocks:
            if block.is_entry:
                return block
        return self.blocks[0] if self.blocks else None

    def iter_instructions(self):
        """Iterate over all instructions in order."""
        for block in self.blocks:
            for instr in block.instructions:
                yield instr

    def to_tokens(self) -> list[str]:
        """Convert entire function to token sequence."""
        tokens = ["[FUNC]"]
        for block in self.blocks:
            tokens.extend(block.to_tokens())
        tokens.append("[/FUNC]")
        return tokens

    def to_ir_string(self) -> str:
        """Convert to human-readable normalized IR string."""
        lines = [f"function @ {self.original_address:x} ({self.isa}):"]
        lines.append(
            f"  blocks: {self.num_blocks}, edges: {self.num_edges}, complexity: {self.cyclomatic_complexity}"
        )
        lines.append("")

        for block in self.blocks:
            flags = []
            if block.is_entry:
                flags.append("entry")
            if block.is_exit:
                flags.append("exit")
            if block.is_loop_header:
                flags.append(f"loop_head(depth={block.loop_depth})")

            flag_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"  block_{block.block_id}{flag_str}:")

            for instr in block.instructions:
                lines.append(f"    {instr}")

            if block.successors:
                succs = ", ".join(f"block_{s}" for s in block.successors)
                lines.append(f"    -> [{succs}]")
            lines.append("")

        return "\n".join(lines)


# Alias for backwards compatibility
NormalizedIR = NormalizedFunction


@dataclass
class DefUseInfo:
    """Definition-Use information for an instruction.

    Used for the Def-Use Prediction (DUP) pretraining task.
    """

    instruction_idx: int
    defines: list[str] = field(default_factory=list)  # Variables/registers defined
    uses: list[str] = field(default_factory=list)  # Variables/registers used

    def has_def_use_relationship(self, other: "DefUseInfo") -> bool:
        """Check if this instruction defines something the other uses."""
        return bool(set(self.defines) & set(other.uses))


def compute_def_use(function: NormalizedFunction) -> list[DefUseInfo]:
    """Compute def-use information for all instructions.

    Args:
        function: Normalized function

    Returns:
        List of DefUseInfo for each instruction
    """
    result = []
    idx = 0

    for block in function.blocks:
        for instr in block.instructions:
            defines = []
            uses = []

            # Destination is a definition
            if instr.dest:
                if instr.dest.kind == NormalizedOperand.Kind.REGISTER:
                    defines.append(str(instr.dest))
                elif instr.dest.kind == NormalizedOperand.Kind.TEMP:
                    defines.append(str(instr.dest))

            # Sources are uses
            for src in instr.src:
                if src.kind in (NormalizedOperand.Kind.REGISTER, NormalizedOperand.Kind.TEMP):
                    uses.append(str(src))
                elif src.kind == NormalizedOperand.Kind.MEMORY:
                    if src.base:
                        uses.append(f"${src.base}")
                    if src.offset and not src.offset.startswith("IMM"):
                        uses.append(f"${src.offset}")

            result.append(
                DefUseInfo(
                    instruction_idx=idx,
                    defines=defines,
                    uses=uses,
                )
            )
            idx += 1

    return result
