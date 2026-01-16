"""Register role mapping for architecture-neutral representation.

Maps concrete architecture-specific registers to abstract roles
(ARG0, ARG1, RET, SP, FP, etc.) based on calling conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from claustrum.utils.types import ISA, RegisterRole, CallingConvention


@dataclass
class CallingConventionSpec:
    """Specification of a calling convention's register usage."""

    name: CallingConvention

    # Integer argument registers in order
    int_args: list[str] = field(default_factory=list)

    # Floating point argument registers
    fp_args: list[str] = field(default_factory=list)

    # Return value registers
    int_return: list[str] = field(default_factory=list)
    fp_return: list[str] = field(default_factory=list)

    # Stack and frame pointers
    stack_pointer: str = ""
    frame_pointer: str = ""
    link_register: str = ""

    # Callee-saved registers (don't need to normalize these)
    callee_saved: list[str] = field(default_factory=list)

    # Caller-saved registers
    caller_saved: list[str] = field(default_factory=list)


# Calling convention specifications for common architectures
CALLING_CONVENTIONS: dict[tuple[ISA, CallingConvention], CallingConventionSpec] = {
    # x86_64 System V ABI
    (ISA.X86_64, CallingConvention.SYSV_AMD64): CallingConventionSpec(
        name=CallingConvention.SYSV_AMD64,
        int_args=["rdi", "rsi", "rdx", "rcx", "r8", "r9"],
        fp_args=["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"],
        int_return=["rax", "rdx"],
        fp_return=["xmm0", "xmm1"],
        stack_pointer="rsp",
        frame_pointer="rbp",
        callee_saved=["rbx", "rbp", "r12", "r13", "r14", "r15"],
        caller_saved=["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"],
    ),
    # x86_64 Microsoft ABI
    (ISA.X86_64, CallingConvention.MS_X64): CallingConventionSpec(
        name=CallingConvention.MS_X64,
        int_args=["rcx", "rdx", "r8", "r9"],
        fp_args=["xmm0", "xmm1", "xmm2", "xmm3"],
        int_return=["rax"],
        fp_return=["xmm0"],
        stack_pointer="rsp",
        frame_pointer="rbp",
        callee_saved=["rbx", "rbp", "rdi", "rsi", "r12", "r13", "r14", "r15"],
        caller_saved=["rax", "rcx", "rdx", "r8", "r9", "r10", "r11"],
    ),
    # x86 cdecl (all args on stack)
    (ISA.X86, CallingConvention.CDECL): CallingConventionSpec(
        name=CallingConvention.CDECL,
        int_args=[],  # All on stack
        fp_args=[],
        int_return=["eax", "edx"],
        fp_return=["st0"],
        stack_pointer="esp",
        frame_pointer="ebp",
        callee_saved=["ebx", "esi", "edi", "ebp"],
        caller_saved=["eax", "ecx", "edx"],
    ),
    # ARM64 AAPCS64
    (ISA.ARM64, CallingConvention.AAPCS64): CallingConventionSpec(
        name=CallingConvention.AAPCS64,
        int_args=["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"],
        fp_args=["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"],
        int_return=["x0", "x1"],
        fp_return=["d0", "d1"],
        stack_pointer="sp",
        frame_pointer="x29",
        link_register="x30",
        callee_saved=[
            "x19",
            "x20",
            "x21",
            "x22",
            "x23",
            "x24",
            "x25",
            "x26",
            "x27",
            "x28",
            "x29",
            "x30",
        ],
        caller_saved=[
            "x0",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
            "x14",
            "x15",
            "x16",
            "x17",
            "x18",
        ],
    ),
    # ARM32 AAPCS
    (ISA.ARM32, CallingConvention.AAPCS): CallingConventionSpec(
        name=CallingConvention.AAPCS,
        int_args=["r0", "r1", "r2", "r3"],
        fp_args=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"],
        int_return=["r0", "r1"],
        fp_return=["s0", "s1"],
        stack_pointer="sp",
        frame_pointer="fp",
        link_register="lr",
        callee_saved=["r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11"],
        caller_saved=["r0", "r1", "r2", "r3", "r12"],
    ),
    # MIPS O32
    (ISA.MIPS32, CallingConvention.O32): CallingConventionSpec(
        name=CallingConvention.O32,
        int_args=["a0", "a1", "a2", "a3"],
        fp_args=["f12", "f14"],
        int_return=["v0", "v1"],
        fp_return=["f0", "f2"],
        stack_pointer="sp",
        frame_pointer="fp",
        link_register="ra",
        callee_saved=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "fp"],
        caller_saved=[
            "at",
            "v0",
            "v1",
            "a0",
            "a1",
            "a2",
            "a3",
            "t0",
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
            "t7",
            "t8",
            "t9",
        ],
    ),
    # MIPS N64
    (ISA.MIPS64, CallingConvention.N64): CallingConventionSpec(
        name=CallingConvention.N64,
        int_args=["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"],
        fp_args=["f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19"],
        int_return=["v0", "v1"],
        fp_return=["f0", "f2"],
        stack_pointer="sp",
        frame_pointer="fp",
        link_register="ra",
        callee_saved=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "fp", "gp"],
        caller_saved=[
            "at",
            "v0",
            "v1",
            "a0",
            "a1",
            "a2",
            "a3",
            "a4",
            "a5",
            "a6",
            "a7",
            "t0",
            "t1",
            "t2",
            "t3",
            "t8",
            "t9",
        ],
    ),
    # RISC-V LP64
    (ISA.RISCV64, CallingConvention.RV_LP64): CallingConventionSpec(
        name=CallingConvention.RV_LP64,
        int_args=["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"],
        fp_args=["fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7"],
        int_return=["a0", "a1"],
        fp_return=["fa0", "fa1"],
        stack_pointer="sp",
        frame_pointer="fp",
        link_register="ra",
        callee_saved=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"],
        caller_saved=[
            "ra",
            "t0",
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
            "a0",
            "a1",
            "a2",
            "a3",
            "a4",
            "a5",
            "a6",
            "a7",
        ],
    ),
    # PPC64 ELFv2
    (ISA.PPC64, CallingConvention.UNKNOWN): CallingConventionSpec(
        name=CallingConvention.UNKNOWN,
        int_args=["r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"],
        fp_args=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"],
        int_return=["r3", "r4"],
        fp_return=["f1"],
        stack_pointer="r1",
        frame_pointer="r31",
        link_register="lr",
        callee_saved=[
            "r14",
            "r15",
            "r16",
            "r17",
            "r18",
            "r19",
            "r20",
            "r21",
            "r22",
            "r23",
            "r24",
            "r25",
            "r26",
            "r27",
            "r28",
            "r29",
            "r30",
            "r31",
        ],
        caller_saved=["r0", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12"],
    ),
}


# Default calling convention per ISA
DEFAULT_CALLING_CONVENTION: dict[ISA, CallingConvention] = {
    ISA.X86: CallingConvention.CDECL,
    ISA.X86_64: CallingConvention.SYSV_AMD64,
    ISA.ARM32: CallingConvention.AAPCS,
    ISA.ARM64: CallingConvention.AAPCS64,
    ISA.MIPS32: CallingConvention.O32,
    ISA.MIPS64: CallingConvention.N64,
    ISA.RISCV64: CallingConvention.RV_LP64,
    ISA.RISCV32: CallingConvention.RV_ILP32,
}


class RegisterMapper:
    """Maps architecture-specific registers to abstract roles.

    This enables comparing functions across architectures by abstracting
    away the concrete register names used by each ISA.
    """

    def __init__(
        self,
        isa: ISA,
        calling_convention: Optional[CallingConvention] = None,
    ):
        self.isa = isa

        # Get calling convention
        if calling_convention is None:
            calling_convention = DEFAULT_CALLING_CONVENTION.get(isa, CallingConvention.UNKNOWN)
        self.calling_convention = calling_convention

        # Get convention spec
        self.spec = CALLING_CONVENTIONS.get(
            (isa, calling_convention),
            self._make_default_spec(),
        )

        # Build reverse mapping
        self._role_map: dict[str, RegisterRole] = {}
        self._build_role_map()

    def _make_default_spec(self) -> CallingConventionSpec:
        """Create a default calling convention spec."""
        return CallingConventionSpec(
            name=CallingConvention.UNKNOWN,
            stack_pointer="sp",
            frame_pointer="fp",
        )

    def _build_role_map(self) -> None:
        """Build the register -> role mapping."""
        spec = self.spec

        # Map argument registers
        for i, reg in enumerate(spec.int_args):
            role = getattr(RegisterRole, f"ARG{i}", RegisterRole.GENERAL)
            self._role_map[reg.lower()] = role

        # Map FP argument registers
        for i, reg in enumerate(spec.fp_args):
            role = getattr(RegisterRole, f"FP_ARG{i}", RegisterRole.GENERAL)
            self._role_map[reg.lower()] = role

        # Map return registers
        if spec.int_return:
            self._role_map[spec.int_return[0].lower()] = RegisterRole.RET
            if len(spec.int_return) > 1:
                self._role_map[spec.int_return[1].lower()] = RegisterRole.RET_HI

        if spec.fp_return:
            self._role_map[spec.fp_return[0].lower()] = RegisterRole.FP_RET

        # Map special registers
        if spec.stack_pointer:
            self._role_map[spec.stack_pointer.lower()] = RegisterRole.SP
        if spec.frame_pointer:
            self._role_map[spec.frame_pointer.lower()] = RegisterRole.FP
        if spec.link_register:
            self._role_map[spec.link_register.lower()] = RegisterRole.LR

        # Add common aliases
        self._add_aliases()

    def _add_aliases(self) -> None:
        """Add common register aliases."""
        aliases = {
            # x86/x86_64
            "rsp": "sp",
            "esp": "sp",
            "rbp": "fp",
            "ebp": "fp",
            "rip": "pc",
            "eip": "pc",
            # ARM
            "r13": "sp",
            "r14": "lr",
            "r15": "pc",
            "x31": "sp",
            # MIPS
            "$sp": "sp",
            "$fp": "fp",
            "$ra": "ra",
            "$zero": "zero",
            "$0": "zero",
        }

        for alias, canonical in aliases.items():
            if canonical.lower() in self._role_map and alias.lower() not in self._role_map:
                self._role_map[alias.lower()] = self._role_map[canonical.lower()]

    def get_role(self, register: str) -> RegisterRole:
        """Get the abstract role for a register.

        Args:
            register: Register name (case-insensitive)

        Returns:
            Abstract register role
        """
        reg_lower = register.lower().lstrip("$%")

        # Direct mapping
        if reg_lower in self._role_map:
            return self._role_map[reg_lower]

        # Check for numbered registers (r0, r1, etc.)
        if reg_lower.startswith("r") and reg_lower[1:].isdigit():
            # VEX often uses r<offset> notation
            return RegisterRole.GENERAL

        # Temporary registers
        if reg_lower.startswith("t") and reg_lower[1:].isdigit():
            return RegisterRole.TEMP

        # Flags/status registers
        if reg_lower in ("flags", "eflags", "rflags", "cpsr", "nzcv"):
            return RegisterRole.FLAGS

        # Program counter
        if reg_lower in ("pc", "ip", "rip", "eip"):
            return RegisterRole.PC

        return RegisterRole.GENERAL

    def normalize_register(self, register: str) -> str:
        """Normalize register name to role-based identifier.

        Args:
            register: Architecture-specific register name

        Returns:
            Role-based identifier string
        """
        role = self.get_role(register)
        return role.value

    def is_argument_register(self, register: str) -> bool:
        """Check if register is used for passing arguments."""
        role = self.get_role(register)
        return role.value.startswith("ARG")

    def is_return_register(self, register: str) -> bool:
        """Check if register is used for return values."""
        role = self.get_role(register)
        return role in (RegisterRole.RET, RegisterRole.RET_HI, RegisterRole.FP_RET)

    def is_special_register(self, register: str) -> bool:
        """Check if register has a special role (SP, FP, LR, PC)."""
        role = self.get_role(register)
        return role in (RegisterRole.SP, RegisterRole.FP, RegisterRole.LR, RegisterRole.PC)


def get_register_mapper(
    isa: ISA,
    calling_convention: Optional[CallingConvention] = None,
) -> RegisterMapper:
    """Get a register mapper for the given ISA.

    Args:
        isa: Target ISA
        calling_convention: Optional specific calling convention

    Returns:
        Configured RegisterMapper
    """
    return RegisterMapper(isa, calling_convention)
