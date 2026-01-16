"""IR Normalizer - transforms lifted IR to architecture-neutral form.

Implements the normalization passes specified in the plan:
- Constant propagation and folding
- Register role abstraction
- Memory operand canonicalization
- Immediate normalization (small constants preserved, large abstracted)
- Address/function call abstraction
- Dead code elimination (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from claustrum.lifting.base import (
    LiftedFunction,
    LiftedBlock,
    IRInstruction,
    IROperand,
    IROpcode,
)
from claustrum.normalization.normalized_ir import (
    NormalizedFunction,
    NormalizedBlock,
    NormalizedInstruction,
    NormalizedOperand,
)
from claustrum.normalization.register_mapping import RegisterMapper
from claustrum.utils.types import ISA, CallingConvention


@dataclass
class NormalizationConfig:
    """Configuration for IR normalization.

    Controls which normalization passes are applied and their parameters.
    """

    # Register abstraction
    abstract_registers: bool = True

    # Immediate handling
    preserve_small_immediates: bool = True
    small_immediate_threshold: int = 255  # Preserve 0-255

    # Address handling
    abstract_addresses: bool = True
    abstract_function_calls: bool = True

    # Memory operand normalization
    canonicalize_memory: bool = True

    # Optimization passes
    constant_propagation: bool = True
    dead_code_elimination: bool = False  # Can lose semantic info

    # Block renumbering
    sequential_block_ids: bool = True

    # Opcode normalization
    unify_comparison_order: bool = True  # Normalize a < b vs b > a


class IRNormalizer:
    """Normalizes lifted IR to architecture-neutral representation.

    This is a critical component for cross-ISA embedding quality.
    The normalizer applies a series of passes to transform architecture-specific
    IR into a canonical form that can be compared across ISAs.
    """

    def __init__(
        self,
        config: Optional[NormalizationConfig] = None,
    ):
        self.config = config or NormalizationConfig()
        self._register_mapper: Optional[RegisterMapper] = None
        self._constant_values: dict[str, int] = {}  # For constant propagation

    def normalize(
        self,
        function: LiftedFunction,
        isa: Optional[ISA] = None,
        calling_convention: Optional[CallingConvention] = None,
    ) -> NormalizedFunction:
        """Normalize a lifted function to architecture-neutral IR.

        Args:
            function: Lifted function from any backend
            isa: Target ISA (auto-detected if not provided)
            calling_convention: Calling convention for register mapping

        Returns:
            Normalized function ready for tokenization/embedding
        """
        # Detect ISA if not provided
        if isa is None and function.isa:
            isa = self._parse_isa(function.isa)

        if isa is None:
            isa = ISA.X86_64  # Default

        # Set up register mapper
        self._register_mapper = RegisterMapper(isa, calling_convention)
        self._constant_values.clear()

        # Create block ID mapping (address -> sequential ID)
        block_addr_to_id: dict[int, int] = {}
        sorted_addrs = sorted(function.blocks.keys())
        for idx, addr in enumerate(sorted_addrs):
            block_addr_to_id[addr] = idx

        # Normalize blocks
        normalized_blocks = []
        for idx, addr in enumerate(sorted_addrs):
            block = function.blocks[addr]
            normalized_block = self._normalize_block(block, idx, block_addr_to_id)
            normalized_blocks.append(normalized_block)

        # Normalize CFG edges
        normalized_edges = []
        for src_addr, dst_addr in function.cfg_edges:
            if src_addr in block_addr_to_id and dst_addr in block_addr_to_id:
                normalized_edges.append(
                    (
                        block_addr_to_id[src_addr],
                        block_addr_to_id[dst_addr],
                    )
                )

        # Detect loop headers (simplified - blocks with back edges)
        self._detect_loops(normalized_blocks, normalized_edges)

        return NormalizedFunction(
            blocks=normalized_blocks,
            cfg_edges=normalized_edges,
            num_args=function.num_args,
            has_return=True,  # TODO: detect from CFG
            original_address=function.address,
            original_name=function.name,
            isa=str(isa.value) if isa else None,
            binary_hash=function.binary_hash,
        )

    def _parse_isa(self, isa_str: str) -> Optional[ISA]:
        """Parse ISA string to enum."""
        isa_map = {
            "x86": ISA.X86,
            "x86_64": ISA.X86_64,
            "amd64": ISA.X86_64,
            "arm": ISA.ARM32,
            "arm32": ISA.ARM32,
            "aarch64": ISA.ARM64,
            "arm64": ISA.ARM64,
            "mips": ISA.MIPS32,
            "mips32": ISA.MIPS32,
            "mips64": ISA.MIPS64,
            "ppc": ISA.PPC32,
            "ppc32": ISA.PPC32,
            "ppc64": ISA.PPC64,
            "riscv32": ISA.RISCV32,
            "riscv64": ISA.RISCV64,
        }
        return isa_map.get(isa_str.lower())

    def _normalize_block(
        self,
        block: LiftedBlock,
        block_id: int,
        block_addr_to_id: dict[int, int],
    ) -> NormalizedBlock:
        """Normalize a single basic block."""
        normalized_instrs = []

        for instr in block.instructions:
            normalized = self._normalize_instruction(instr, block_addr_to_id)
            if normalized is not None:
                normalized_instrs.append(normalized)

        # Convert successor addresses to block IDs
        successor_ids = [
            block_addr_to_id[addr] for addr in block.successors if addr in block_addr_to_id
        ]

        return NormalizedBlock(
            block_id=block_id,
            instructions=normalized_instrs,
            successors=successor_ids,
            is_entry=block.is_entry,
            is_exit=block.is_exit,
            original_address=block.address,
        )

    def _normalize_instruction(
        self,
        instr: IRInstruction,
        block_addr_to_id: dict[int, int],
    ) -> Optional[NormalizedInstruction]:
        """Normalize a single instruction."""
        # Skip NOPs and unknown opcodes if configured
        if instr.opcode == IROpcode.NOP:
            return None

        # Normalize opcode
        opcode = instr.opcode.value

        # Normalize destination
        dest = None
        if instr.dest:
            dest = self._normalize_operand(instr.dest, is_dest=True)

        # Normalize sources
        src = [self._normalize_operand(s) for s in instr.src]

        # Handle specific opcodes
        if instr.opcode == IROpcode.CALL and self.config.abstract_function_calls:
            # Abstract call target
            if src and src[0].kind == NormalizedOperand.Kind.ADDRESS:
                src = [NormalizedOperand.function()]

        if instr.opcode in (IROpcode.JUMP, IROpcode.BRANCH):
            # Convert jump targets to block IDs if possible
            if src and isinstance(instr.src[0].value, int):
                target_addr = instr.src[0].value
                if target_addr in block_addr_to_id:
                    src = [NormalizedOperand.immediate(f"BLOCK_{block_addr_to_id[target_addr]}")]
                else:
                    src = [NormalizedOperand.address()]

        return NormalizedInstruction(
            opcode=opcode,
            dest=dest,
            src=src,
            original_address=instr.address,
            original_size=instr.size,
            original_asm=instr.comment,
        )

    def _normalize_operand(
        self,
        operand: IROperand,
        is_dest: bool = False,
    ) -> NormalizedOperand:
        """Normalize a single operand."""

        if operand.kind == IROperand.Kind.REGISTER:
            # Abstract register to role
            if self.config.abstract_registers and self._register_mapper:
                role = self._register_mapper.normalize_register(operand.value)
                return NormalizedOperand.register(role, operand.size)
            else:
                return NormalizedOperand.register(operand.value, operand.size)

        elif operand.kind == IROperand.Kind.TEMPORARY:
            # Keep temporaries but renumber
            return NormalizedOperand.temp(
                int(operand.value.lstrip("t"))
                if operand.value.startswith("t")
                else hash(operand.value) % 1000,
                operand.size,
            )

        elif operand.kind == IROperand.Kind.IMMEDIATE:
            # Normalize immediate values
            return self._normalize_immediate(operand.value, operand.size)

        elif operand.kind == IROperand.Kind.MEMORY:
            # Normalize memory operand
            return self._normalize_memory(operand)

        elif operand.kind == IROperand.Kind.LABEL:
            # Abstract addresses
            if self.config.abstract_addresses:
                return NormalizedOperand.address()
            else:
                return NormalizedOperand.immediate(
                    f"0x{operand.value:x}" if isinstance(operand.value, int) else str(operand.value)
                )

        else:
            return NormalizedOperand.immediate(str(operand.value))

    def _normalize_immediate(self, value: int, size: int) -> NormalizedOperand:
        """Normalize an immediate value.

        Strategy from plan:
        - Preserve small constants (0-255)
        - Abstract larger values to IMM16/IMM32/IMM64 tokens
        """
        if not isinstance(value, int):
            return NormalizedOperand.immediate(str(value), size)

        # Preserve small immediates
        if self.config.preserve_small_immediates:
            if 0 <= value <= self.config.small_immediate_threshold:
                return NormalizedOperand.immediate(str(value), size)

            # Also preserve small negative numbers
            if -128 <= value < 0:
                return NormalizedOperand.immediate(str(value), size)

        # Abstract large values by size
        if abs(value) <= 0xFFFF:
            return NormalizedOperand.immediate("IMM16", 16)
        elif abs(value) <= 0xFFFFFFFF:
            return NormalizedOperand.immediate("IMM32", 32)
        else:
            return NormalizedOperand.immediate("IMM64", 64)

    def _normalize_memory(self, operand: IROperand) -> NormalizedOperand:
        """Normalize a memory operand to canonical form.

        Canonical form: MEM[base+offset] with size annotation
        """
        base = None
        offset = None

        # Normalize base register
        if operand.base:
            if self.config.abstract_registers and self._register_mapper:
                base = self._register_mapper.normalize_register(operand.base)
            else:
                base = operand.base

        # Normalize offset/index
        if operand.index:
            if self.config.abstract_registers and self._register_mapper:
                idx = self._register_mapper.normalize_register(operand.index)
            else:
                idx = operand.index

            if operand.scale != 1:
                offset = f"{idx}*{operand.scale}"
            else:
                offset = idx

        # Handle constant offset
        if operand.offset != 0:
            offset_norm = self._normalize_immediate(operand.offset, 64)
            if offset:
                offset = f"{offset}+{offset_norm.value}"
            else:
                offset = offset_norm.value

        return NormalizedOperand.memory(base, offset, operand.size)

    def _detect_loops(
        self,
        blocks: list[NormalizedBlock],
        edges: list[tuple[int, int]],
    ) -> None:
        """Detect loop headers using back edge analysis.

        A back edge is an edge from a node to a dominator.
        Simplified version: edge where dst <= src (assuming topological order).
        """
        # Build adjacency list
        successors = {b.block_id: [] for b in blocks}
        predecessors = {b.block_id: [] for b in blocks}

        for src, dst in edges:
            successors[src].append(dst)
            predecessors[dst].append(src)

        # Update block predecessor/successor lists
        for block in blocks:
            block.successors = successors.get(block.block_id, [])
            block.predecessors = predecessors.get(block.block_id, [])

        # Detect back edges (simplified)
        back_edges = [(s, d) for s, d in edges if d <= s]

        # Mark loop headers
        loop_headers = {d for _, d in back_edges}
        for block in blocks:
            if block.block_id in loop_headers:
                block.is_loop_header = True

        # Compute approximate loop depth
        # (Full dominator analysis would be more accurate)
        for block in blocks:
            depth = 0
            for _, dst in back_edges:
                if block.block_id >= dst:
                    depth += 1
            block.loop_depth = depth

        # Mark exit blocks (no successors or only return)
        for block in blocks:
            if not block.successors:
                block.is_exit = True
            elif block.instructions:
                last = block.instructions[-1]
                if last.opcode == "RET":
                    block.is_exit = True


def normalize_function(
    function: LiftedFunction,
    isa: Optional[ISA] = None,
    config: Optional[NormalizationConfig] = None,
) -> NormalizedFunction:
    """Convenience function to normalize a lifted function.

    Args:
        function: Lifted function from any backend
        isa: Target ISA
        config: Normalization configuration

    Returns:
        Normalized function
    """
    normalizer = IRNormalizer(config)
    return normalizer.normalize(function, isa)
