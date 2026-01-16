"""IR Lifting module for CLAUSTRUM.

This module provides multi-backend support for lifting binary code to
architecture-neutral intermediate representations:

- VEXLifter: Uses angr/pyvex for VEX IR (Tier 1 architectures)
- ESILLifter: Uses radare2 for ESIL (broad architecture support)
- PCodeLifter: Uses Ghidra for P-Code (broadest coverage)

The tiered lifting strategy uses VEX IR for well-supported architectures
(x86, ARM, MIPS, RISC-V, PPC) and falls back to ESIL/P-Code for others.
"""

from claustrum.lifting.base import IRLifter, LiftedFunction, LiftedBlock, IRInstruction
from claustrum.lifting.vex import VEXLifter
from claustrum.lifting.esil import ESILLifter
from claustrum.lifting.pcode import PCodeLifter
from claustrum.lifting.factory import create_lifter, TieredLifter

__all__ = [
    "IRLifter",
    "LiftedFunction",
    "LiftedBlock",
    "IRInstruction",
    "VEXLifter",
    "ESILLifter",
    "PCodeLifter",
    "create_lifter",
    "TieredLifter",
]
