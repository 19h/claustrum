"""Micro-trace collection via forced execution.

Implements Trex-style trace collection by executing instruction sequences
while ignoring control flow constraints, capturing dynamic register and
memory values.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Callable
from abc import ABC, abstractmethod


class AccessType(Enum):
    """Memory access type."""

    READ = "read"
    WRITE = "write"


@dataclass
class MemoryAccess:
    """A single memory access during execution."""

    address: int
    size: int
    value: int
    access_type: AccessType
    instruction_index: int


@dataclass
class RegisterState:
    """Snapshot of register state after an instruction."""

    # General purpose registers (normalized to role-based names)
    general: dict[str, int] = field(default_factory=dict)
    # Status/flag registers
    flags: dict[str, bool] = field(default_factory=dict)
    # Special registers (SP, PC, etc.)
    special: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "general": self.general,
            "flags": self.flags,
            "special": self.special,
        }

    def hash(self) -> str:
        """Compute hash of register state."""
        content = str(sorted(self.general.items()))
        content += str(sorted(self.flags.items()))
        content += str(sorted(self.special.items()))
        return hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class TracePoint:
    """A single point in an execution trace.

    Captures the state after executing one instruction.
    """

    instruction_index: int
    instruction_address: int
    instruction_bytes: bytes
    register_state: RegisterState
    memory_accesses: list[MemoryAccess] = field(default_factory=list)

    # Additional semantic info
    opcode: Optional[str] = None
    operands: Optional[list[str]] = None


@dataclass
class ExecutionTrace:
    """A complete execution trace for a function.

    Contains sequence of TracePoints from forced execution.
    """

    function_id: str
    isa: str
    trace_points: list[TracePoint]
    input_state: dict[str, int]  # Initial register/memory values

    # Metadata
    num_instructions: int = 0
    coverage: float = 0.0  # Fraction of instructions executed

    def __post_init__(self):
        self.num_instructions = len(self.trace_points)

    def get_register_sequence(self, register: str) -> list[Optional[int]]:
        """Get sequence of values for a specific register."""
        return [tp.register_state.general.get(register) for tp in self.trace_points]

    def get_memory_accesses(self) -> list[MemoryAccess]:
        """Get all memory accesses in order."""
        accesses = []
        for tp in self.trace_points:
            accesses.extend(tp.memory_accesses)
        return accesses


class EmulatorBackend(ABC):
    """Abstract base class for emulator backends."""

    @abstractmethod
    def initialize(self, isa: str) -> None:
        """Initialize emulator for given ISA."""
        pass

    @abstractmethod
    def set_code(self, code: bytes, base_address: int) -> None:
        """Set code to execute."""
        pass

    @abstractmethod
    def set_register(self, name: str, value: int) -> None:
        """Set register value."""
        pass

    @abstractmethod
    def get_register(self, name: str) -> int:
        """Get register value."""
        pass

    @abstractmethod
    def step(self) -> bool:
        """Execute one instruction. Returns False if execution failed."""
        pass

    @abstractmethod
    def get_memory(self, address: int, size: int) -> bytes:
        """Read memory."""
        pass

    @abstractmethod
    def set_memory(self, address: int, data: bytes) -> None:
        """Write memory."""
        pass


class UnicornBackend(EmulatorBackend):
    """Unicorn-based emulator backend.

    Supports: x86, x86_64, ARM, ARM64, MIPS, PPC, SPARC, M68K
    """

    # ISA to Unicorn constants mapping
    ISA_CONFIG = {
        "x86": {"arch": 3, "mode": 4},  # UC_ARCH_X86, UC_MODE_32
        "x86_64": {"arch": 3, "mode": 8},  # UC_ARCH_X86, UC_MODE_64
        "arm32": {"arch": 1, "mode": 0},  # UC_ARCH_ARM, UC_MODE_ARM
        "arm64": {"arch": 2, "mode": 0},  # UC_ARCH_ARM64, UC_MODE_ARM
        "mips32": {"arch": 4, "mode": 4},  # UC_ARCH_MIPS, UC_MODE_MIPS32
        "mips64": {"arch": 4, "mode": 8},  # UC_ARCH_MIPS, UC_MODE_MIPS64
        "ppc32": {"arch": 5, "mode": 4},  # UC_ARCH_PPC, UC_MODE_PPC32
        "ppc64": {"arch": 5, "mode": 8},  # UC_ARCH_PPC, UC_MODE_PPC64
    }

    # Register mappings (ISA -> role -> native name)
    REGISTER_MAPS = {
        "x86_64": {
            "ARG0": "rdi",
            "ARG1": "rsi",
            "ARG2": "rdx",
            "ARG3": "rcx",
            "ARG4": "r8",
            "ARG5": "r9",
            "RET": "rax",
            "SP": "rsp",
            "FP": "rbp",
            "G0": "rbx",
            "G1": "r10",
            "G2": "r11",
            "G3": "r12",
            "G4": "r13",
        },
        "arm64": {
            "ARG0": "x0",
            "ARG1": "x1",
            "ARG2": "x2",
            "ARG3": "x3",
            "ARG4": "x4",
            "ARG5": "x5",
            "ARG6": "x6",
            "ARG7": "x7",
            "RET": "x0",
            "SP": "sp",
            "FP": "x29",
            "LR": "x30",
        },
        # Add more ISAs...
    }

    def __init__(self):
        self.uc = None
        self.isa = None
        self._registers = {}

    def initialize(self, isa: str) -> None:
        """Initialize Unicorn emulator."""
        try:
            from unicorn import Uc
            from unicorn import UC_ARCH_X86, UC_MODE_64, UC_MODE_32
            from unicorn import UC_ARCH_ARM64, UC_ARCH_ARM
            from unicorn import UC_ARCH_MIPS, UC_ARCH_PPC
        except ImportError:
            raise ImportError("Unicorn engine not installed. Install with: pip install unicorn")

        if isa not in self.ISA_CONFIG:
            raise ValueError(f"Unsupported ISA for Unicorn: {isa}")

        config = self.ISA_CONFIG[isa]
        self.uc = Uc(config["arch"], config["mode"])
        self.isa = isa

        # Map 2MB for code and stack
        self.code_base = 0x10000
        self.stack_base = 0x80000

        self.uc.mem_map(self.code_base, 2 * 1024 * 1024)  # Code
        self.uc.mem_map(self.stack_base, 1 * 1024 * 1024)  # Stack

        # Initialize stack pointer
        self._init_stack()

    def _init_stack(self) -> None:
        """Initialize stack pointer."""
        sp_value = self.stack_base + 0x80000  # Middle of stack region

        if self.isa in ("x86_64",):
            self.set_register("rsp", sp_value)
        elif self.isa in ("x86",):
            self.set_register("esp", sp_value)
        elif self.isa in ("arm64",):
            self.set_register("sp", sp_value)
        elif self.isa in ("arm32",):
            self.set_register("sp", sp_value)

    def set_code(self, code: bytes, base_address: int) -> None:
        """Load code into emulator."""
        self.uc.mem_write(base_address, code)

        # Set program counter
        if self.isa in ("x86_64",):
            self.set_register("rip", base_address)
        elif self.isa in ("x86",):
            self.set_register("eip", base_address)
        elif self.isa in ("arm64", "arm32"):
            self.set_register("pc", base_address)

    def set_register(self, name: str, value: int) -> None:
        """Set register value."""
        reg_id = self._get_register_id(name)
        if reg_id is not None:
            self.uc.reg_write(reg_id, value)

    def get_register(self, name: str) -> int:
        """Get register value."""
        reg_id = self._get_register_id(name)
        if reg_id is not None:
            return self.uc.reg_read(reg_id)
        return 0

    def _get_register_id(self, name: str) -> Optional[int]:
        """Map register name to Unicorn constant."""
        # This would use unicorn.x86_const, unicorn.arm64_const, etc.
        # Simplified mapping here
        try:
            if self.isa == "x86_64":
                from unicorn.x86_const import (
                    UC_X86_REG_RAX,
                    UC_X86_REG_RBX,
                    UC_X86_REG_RCX,
                    UC_X86_REG_RDX,
                    UC_X86_REG_RSI,
                    UC_X86_REG_RDI,
                    UC_X86_REG_RSP,
                    UC_X86_REG_RBP,
                    UC_X86_REG_RIP,
                    UC_X86_REG_R8,
                    UC_X86_REG_R9,
                    UC_X86_REG_R10,
                    UC_X86_REG_R11,
                    UC_X86_REG_R12,
                    UC_X86_REG_R13,
                    UC_X86_REG_R14,
                    UC_X86_REG_R15,
                )

                mapping = {
                    "rax": UC_X86_REG_RAX,
                    "rbx": UC_X86_REG_RBX,
                    "rcx": UC_X86_REG_RCX,
                    "rdx": UC_X86_REG_RDX,
                    "rsi": UC_X86_REG_RSI,
                    "rdi": UC_X86_REG_RDI,
                    "rsp": UC_X86_REG_RSP,
                    "rbp": UC_X86_REG_RBP,
                    "rip": UC_X86_REG_RIP,
                    "r8": UC_X86_REG_R8,
                    "r9": UC_X86_REG_R9,
                    "r10": UC_X86_REG_R10,
                    "r11": UC_X86_REG_R11,
                    "r12": UC_X86_REG_R12,
                    "r13": UC_X86_REG_R13,
                    "r14": UC_X86_REG_R14,
                    "r15": UC_X86_REG_R15,
                }
                return mapping.get(name.lower())
        except ImportError:
            pass
        return None

    def step(self) -> bool:
        """Execute one instruction."""
        try:
            if self.isa in ("x86_64",):
                pc = self.get_register("rip")
            elif self.isa in ("arm64", "arm32"):
                pc = self.get_register("pc")
            else:
                return False

            # Execute one instruction
            self.uc.emu_start(pc, pc + 16, count=1)
            return True
        except Exception:
            return False

    def get_memory(self, address: int, size: int) -> bytes:
        """Read memory."""
        try:
            return self.uc.mem_read(address, size)
        except Exception:
            return b"\x00" * size

    def set_memory(self, address: int, data: bytes) -> None:
        """Write memory."""
        try:
            self.uc.mem_write(address, data)
        except Exception:
            pass


class MicroTraceCollector:
    """Collects micro-traces through forced execution.

    Implements Trex-style trace collection by executing instruction
    sequences linearly (ignoring control flow) and capturing dynamic
    state after each instruction.

    Args:
        isa: Target instruction set architecture
        backend: Emulator backend ("unicorn" or "qemu")
        max_instructions: Maximum instructions to trace
        timeout_ms: Execution timeout per instruction
    """

    def __init__(
        self,
        isa: str,
        backend: str = "unicorn",
        max_instructions: int = 256,
        timeout_ms: int = 100,
    ):
        self.isa = isa
        self.max_instructions = max_instructions
        self.timeout_ms = timeout_ms

        # Initialize emulator backend
        if backend == "unicorn":
            self.emulator = UnicornBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        try:
            self.emulator.initialize(isa)
            self._available = True
        except (ImportError, ValueError) as e:
            self._available = False
            self._error = str(e)

        # Disassembler for instruction info
        self._init_disassembler()

    def _init_disassembler(self) -> None:
        """Initialize Capstone disassembler."""
        try:
            import capstone

            arch_mode = {
                "x86": (capstone.CS_ARCH_X86, capstone.CS_MODE_32),
                "x86_64": (capstone.CS_ARCH_X86, capstone.CS_MODE_64),
                "arm32": (capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM),
                "arm64": (capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM),
                "mips32": (capstone.CS_ARCH_MIPS, capstone.CS_MODE_MIPS32),
                "mips64": (capstone.CS_ARCH_MIPS, capstone.CS_MODE_MIPS64),
            }

            if self.isa in arch_mode:
                arch, mode = arch_mode[self.isa]
                self.disasm = capstone.Cs(arch, mode)
                self.disasm.detail = True
            else:
                self.disasm = None
        except ImportError:
            self.disasm = None

    def collect_traces(
        self,
        function_bytes: bytes,
        base_address: int = 0x10000,
        num_traces: int = 10,
        randomize_inputs: bool = True,
    ) -> list[ExecutionTrace]:
        """Collect multiple execution traces for a function.

        Args:
            function_bytes: Raw machine code bytes
            base_address: Base address for code mapping
            num_traces: Number of traces to collect
            randomize_inputs: Whether to randomize initial register values

        Returns:
            List of execution traces
        """
        if not self._available:
            return []

        traces = []

        for trace_idx in range(num_traces):
            # Generate input state
            if randomize_inputs:
                input_state = self._generate_random_inputs()
            else:
                input_state = self._generate_default_inputs()

            trace = self._collect_single_trace(
                function_bytes=function_bytes,
                base_address=base_address,
                input_state=input_state,
                trace_id=f"trace_{trace_idx}",
            )

            if trace and trace.num_instructions > 0:
                traces.append(trace)

        return traces

    def _collect_single_trace(
        self,
        function_bytes: bytes,
        base_address: int,
        input_state: dict[str, int],
        trace_id: str,
    ) -> Optional[ExecutionTrace]:
        """Collect a single execution trace.

        Args:
            function_bytes: Raw machine code
            base_address: Load address
            input_state: Initial register values
            trace_id: Trace identifier

        Returns:
            ExecutionTrace or None if collection failed
        """
        try:
            # Reset and configure emulator
            self.emulator.initialize(self.isa)
            self.emulator.set_code(function_bytes, base_address)

            # Set initial register state
            for reg, value in input_state.items():
                self.emulator.set_register(reg, value)

            # Collect trace points
            trace_points = []
            current_address = base_address

            # Disassemble to get instruction boundaries
            instructions = []
            if self.disasm:
                for insn in self.disasm.disasm(function_bytes, base_address):
                    instructions.append(
                        {
                            "address": insn.address,
                            "size": insn.size,
                            "mnemonic": insn.mnemonic,
                            "op_str": insn.op_str,
                            "bytes": insn.bytes,
                        }
                    )

            if not instructions:
                # Fallback: assume fixed instruction size
                inst_size = 4 if self.isa in ("arm64", "arm32", "mips32", "mips64") else 1
                for i in range(0, len(function_bytes), inst_size):
                    instructions.append(
                        {
                            "address": base_address + i,
                            "size": min(inst_size, len(function_bytes) - i),
                            "mnemonic": "unknown",
                            "op_str": "",
                            "bytes": function_bytes[i : i + inst_size],
                        }
                    )

            # Execute instructions and collect state
            for idx, insn in enumerate(instructions[: self.max_instructions]):
                # Execute instruction
                success = self.emulator.step()

                if not success:
                    break

                # Capture register state
                reg_state = self._capture_register_state()

                # Create trace point
                tp = TracePoint(
                    instruction_index=idx,
                    instruction_address=insn["address"],
                    instruction_bytes=bytes(insn["bytes"]),
                    register_state=reg_state,
                    opcode=insn["mnemonic"],
                    operands=[insn["op_str"]] if insn["op_str"] else [],
                )
                trace_points.append(tp)

            return ExecutionTrace(
                function_id=trace_id,
                isa=self.isa,
                trace_points=trace_points,
                input_state=input_state,
                coverage=len(trace_points) / max(len(instructions), 1),
            )

        except Exception as e:
            return None

    def _capture_register_state(self) -> RegisterState:
        """Capture current register state from emulator."""
        general = {}
        flags = {}
        special = {}

        # Get role-mapped registers for the ISA
        if self.isa == "x86_64":
            role_regs = [
                ("ARG0", "rdi"),
                ("ARG1", "rsi"),
                ("ARG2", "rdx"),
                ("ARG3", "rcx"),
                ("ARG4", "r8"),
                ("ARG5", "r9"),
                ("RET", "rax"),
                ("SP", "rsp"),
                ("FP", "rbp"),
            ]
            for role, native in role_regs:
                value = self.emulator.get_register(native)
                if role in ("SP", "FP"):
                    special[role] = value
                else:
                    general[role] = value

        elif self.isa == "arm64":
            role_regs = [
                ("ARG0", "x0"),
                ("ARG1", "x1"),
                ("ARG2", "x2"),
                ("ARG3", "x3"),
                ("ARG4", "x4"),
                ("ARG5", "x5"),
                ("RET", "x0"),
                ("SP", "sp"),
                ("FP", "x29"),
            ]
            for role, native in role_regs:
                value = self.emulator.get_register(native)
                if role in ("SP", "FP"):
                    special[role] = value
                else:
                    general[role] = value

        return RegisterState(general=general, flags=flags, special=special)

    def _generate_random_inputs(self) -> dict[str, int]:
        """Generate randomized input register values."""
        import random

        inputs = {}

        if self.isa == "x86_64":
            arg_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
            for reg in arg_regs:
                inputs[reg] = random.randint(0, 2**32 - 1)

        elif self.isa == "arm64":
            arg_regs = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]
            for reg in arg_regs:
                inputs[reg] = random.randint(0, 2**32 - 1)

        return inputs

    def _generate_default_inputs(self) -> dict[str, int]:
        """Generate default input register values."""
        inputs = {}

        if self.isa == "x86_64":
            inputs = {"rdi": 1, "rsi": 2, "rdx": 3, "rcx": 4, "r8": 5, "r9": 6}
        elif self.isa == "arm64":
            inputs = {"x0": 1, "x1": 2, "x2": 3, "x3": 4, "x4": 5, "x5": 6}

        return inputs

    @property
    def available(self) -> bool:
        """Check if trace collection is available."""
        return self._available
