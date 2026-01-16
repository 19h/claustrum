"""Factory for creating appropriate lifters based on ISA.

Implements the tiered lifting strategy:
- Tier 1: VEX IR for x86, ARM, MIPS, RISC-V, PPC
- Tier 2: VEX with ESIL fallback
- Tier 3: ESIL for embedded architectures
- Tier 4: P-Code or ESIL for exotic ISAs
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

from claustrum.lifting.base import (
    IRLifter,
    LiftedFunction,
    LiftedBlock,
    LiftingError,
)
from claustrum.lifting.vex import VEXLifter
from claustrum.lifting.esil import ESILLifter
from claustrum.lifting.pcode import PCodeLifter
from claustrum.utils.types import ISA


# Lifter registry
_LIFTERS: dict[str, type[IRLifter]] = {
    "vex": VEXLifter,
    "esil": ESILLifter,
    "pcode": PCodeLifter,
}


def register_lifter(name: str, lifter_class: type[IRLifter]) -> None:
    """Register a new lifter backend.

    Args:
        name: Unique name for the lifter
        lifter_class: Lifter class implementing IRLifter
    """
    _LIFTERS[name] = lifter_class


def get_available_lifters() -> list[str]:
    """Get list of available lifter backends."""
    return list(_LIFTERS.keys())


def create_lifter(backend: str = "auto", isa: Optional[str] = None) -> IRLifter:
    """Create a lifter for the specified backend.

    Args:
        backend: Lifter backend name ("vex", "esil", "auto")
        isa: Target ISA (used for "auto" selection)

    Returns:
        Configured lifter instance
    """
    if backend == "auto":
        if isa:
            return TieredLifter().get_best_lifter(isa)
        return TieredLifter()

    if backend not in _LIFTERS:
        raise ValueError(f"Unknown lifter backend: {backend}. Available: {list(_LIFTERS.keys())}")

    return _LIFTERS[backend]()


class TieredLifter(IRLifter):
    """Multi-backend lifter that selects the best backend per ISA.

    Implements the tiered lifting strategy from the plan:
    - Use VEX IR for Tier 1 architectures (best quality)
    - Fall back to ESIL for less common ISAs
    - Supports adding P-Code backend for broadest coverage
    """

    def __init__(self, enable_pcode: bool = True):
        super().__init__()
        self.name = "tiered"

        # Initialize available lifters
        self._lifters: list[IRLifter] = []

        # Try to initialize VEX lifter (highest quality for supported ISAs)
        try:
            self._lifters.append(VEXLifter())
        except ImportError:
            pass

        # Try to initialize ESIL lifter (good fallback)
        try:
            self._lifters.append(ESILLifter())
        except ImportError:
            pass

        # Try to initialize P-Code lifter (broadest coverage)
        if enable_pcode:
            try:
                self._lifters.append(PCodeLifter())
            except (ImportError, Exception):
                pass  # pyhidra not available or Ghidra not installed

        if not self._lifters:
            raise RuntimeError(
                "No lifting backends available. Install angr (for VEX), r2pipe (for ESIL), "
                "or pyhidra (for P-Code)."
            )

        # Build supported ISA set from all lifters
        self.supported_isas = set()
        for lifter in self._lifters:
            self.supported_isas.update(lifter.supported_isas)

    def get_best_lifter(self, isa: str) -> IRLifter:
        """Get the best lifter for a given ISA.

        Args:
            isa: Target ISA name

        Returns:
            Best available lifter for the ISA
        """
        isa_lower = isa.lower()

        # Sort lifters by priority for this ISA
        ranked = sorted(
            self._lifters,
            key=lambda l: l.get_priority(isa_lower),
            reverse=True,
        )

        # Return highest priority lifter that supports this ISA
        for lifter in ranked:
            if lifter.supports_isa(isa_lower):
                return lifter

        # Fall back to first available
        if self._lifters:
            return self._lifters[0]

        raise LiftingError(f"No lifter available for ISA: {isa}")

    def lift_block(
        self,
        binary_path: Path,
        block_addr: int,
        max_size: int = 4096,
        isa: Optional[str] = None,
    ) -> LiftedBlock:
        """Lift a block using the best available backend.

        Args:
            binary_path: Path to binary file
            block_addr: Address of block start
            max_size: Maximum bytes to lift
            isa: Optional ISA hint for backend selection

        Returns:
            Lifted basic block
        """
        if isa:
            lifter = self.get_best_lifter(isa)
        else:
            # Try lifters in priority order until one works
            errors = []
            for lifter in self._lifters:
                try:
                    return lifter.lift_block(binary_path, block_addr, max_size)
                except LiftingError as e:
                    errors.append(str(e))
                except Exception as e:
                    errors.append(f"{lifter.name}: {e}")

            raise LiftingError(
                f"All lifters failed for block at 0x{block_addr:x}: " + "; ".join(errors),
                address=block_addr,
            )

        return lifter.lift_block(binary_path, block_addr, max_size)

    def lift_function(
        self,
        binary_path: Path,
        function_addr: int,
        function_size: Optional[int] = None,
        isa: Optional[str] = None,
    ) -> LiftedFunction:
        """Lift a function using the best available backend.

        Args:
            binary_path: Path to binary file
            function_addr: Address of function start
            function_size: Optional size hint
            isa: Optional ISA hint for backend selection

        Returns:
            Lifted function with CFG
        """
        if isa:
            lifter = self.get_best_lifter(isa)
            return lifter.lift_function(binary_path, function_addr, function_size)

        # Try lifters in priority order until one works
        errors = []
        for lifter in self._lifters:
            try:
                result = lifter.lift_function(binary_path, function_addr, function_size)
                return result
            except LiftingError as e:
                errors.append(str(e))
            except Exception as e:
                errors.append(f"{lifter.name}: {e}")

        raise LiftingError(
            f"All lifters failed for function at 0x{function_addr:x}: " + "; ".join(errors),
            address=function_addr,
        )

    def detect_isa(self, binary_path: Path) -> str:
        """Detect the ISA of a binary file.

        Args:
            binary_path: Path to binary file

        Returns:
            ISA name string
        """
        # Try to detect using available lifters
        for lifter in self._lifters:
            if isinstance(lifter, VEXLifter):
                try:
                    import angr

                    proj = angr.Project(str(binary_path), auto_load_libs=False)
                    return proj.arch.name.lower()
                except Exception:
                    pass

            elif isinstance(lifter, ESILLifter):
                try:
                    import r2pipe

                    r2 = r2pipe.open(str(binary_path), flags=["-2"])
                    info = r2.cmdj("iAj")
                    r2.quit()
                    if info and "arch" in info:
                        return info["arch"].lower()
                except Exception:
                    pass

        raise LiftingError(f"Could not detect ISA for {binary_path}")


# Convenience functions
def lift_function(
    binary_path: Path,
    function_addr: int,
    function_size: Optional[int] = None,
    isa: Optional[str] = None,
    backend: str = "auto",
) -> LiftedFunction:
    """Convenience function to lift a function.

    Args:
        binary_path: Path to binary file
        function_addr: Address of function start
        function_size: Optional size hint
        isa: Optional ISA hint
        backend: Lifter backend ("vex", "esil", "auto")

    Returns:
        Lifted function
    """
    lifter = create_lifter(backend, isa)

    if isinstance(lifter, TieredLifter):
        return lifter.lift_function(binary_path, function_addr, function_size, isa)

    return lifter.lift_function(binary_path, function_addr, function_size)


def lift_block(
    binary_path: Path,
    block_addr: int,
    max_size: int = 4096,
    isa: Optional[str] = None,
    backend: str = "auto",
) -> LiftedBlock:
    """Convenience function to lift a basic block.

    Args:
        binary_path: Path to binary file
        block_addr: Address of block start
        max_size: Maximum bytes to lift
        isa: Optional ISA hint
        backend: Lifter backend

    Returns:
        Lifted basic block
    """
    lifter = create_lifter(backend, isa)

    if isinstance(lifter, TieredLifter):
        return lifter.lift_block(binary_path, block_addr, max_size, isa)

    return lifter.lift_block(binary_path, block_addr, max_size)
