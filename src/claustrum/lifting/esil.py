"""ESIL (Evaluable Strings Intermediate Language) lifter using radare2.

ESIL provides broad architecture support through radare2's extensive
processor specifications, covering embedded architectures often missing
from VEX (AVR, MSP430, Xtensa, etc.).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional, Any

from claustrum.lifting.base import (
    IRLifter,
    IRInstruction,
    IROpcode,
    IROperand,
    LiftedBlock,
    LiftedFunction,
    LiftingError,
)


# ESIL operation mapping to unified IR
# ESIL is stack-based: a,b,+ means push a, push b, add and push result
ESIL_OPCODE_MAP = {
    # Arithmetic
    "+": IROpcode.ADD,
    "-": IROpcode.SUB,
    "*": IROpcode.MUL,
    "/": IROpcode.DIV,
    "~/": IROpcode.SDIV,
    "%": IROpcode.MOD,
    "~%": IROpcode.SMOD,
    # Bitwise
    "&": IROpcode.AND,
    "|": IROpcode.OR,
    "^": IROpcode.XOR,
    "!": IROpcode.NOT,
    "<<": IROpcode.SHL,
    ">>": IROpcode.SHR,
    ">>>>": IROpcode.SAR,  # Arithmetic shift
    "<<<": IROpcode.ROL,
    ">>>": IROpcode.ROR,
    # Comparison
    "==": IROpcode.CMP_EQ,
    "!=": IROpcode.CMP_NE,
    "<": IROpcode.CMP_LT,
    "<=": IROpcode.CMP_LE,
    ">": IROpcode.CMP_LT,  # Reversed operand order
    ">=": IROpcode.CMP_LE,  # Reversed operand order
    # Memory
    "=[1]": IROpcode.STORE,
    "=[2]": IROpcode.STORE,
    "=[4]": IROpcode.STORE,
    "=[8]": IROpcode.STORE,
    "[1]": IROpcode.LOAD,
    "[2]": IROpcode.LOAD,
    "[4]": IROpcode.LOAD,
    "[8]": IROpcode.LOAD,
}


class ESILParser:
    """Parser for ESIL expressions.

    ESIL is a stack-based IL where operations pop operands from
    the stack and push results. Example:
        "rax,rbx,+,rcx,=" means rcx = rax + rbx
    """

    def __init__(self):
        self.stack: list[IROperand] = []
        self.instructions: list[IRInstruction] = []
        self.temp_counter = 0
        self.address = 0

    def reset(self, address: int = 0):
        """Reset parser state."""
        self.stack = []
        self.instructions = []
        self.temp_counter = 0
        self.address = address

    def _new_temp(self, size: int = 64) -> IROperand:
        """Create a new temporary operand."""
        tmp = IROperand.tmp(self.temp_counter, size)
        self.temp_counter += 1
        return tmp

    def _emit(
        self,
        opcode: IROpcode,
        dest: Optional[IROperand] = None,
        src: Optional[list[IROperand]] = None,
    ) -> None:
        """Emit an IR instruction."""
        self.instructions.append(
            IRInstruction(
                opcode=opcode,
                dest=dest,
                src=src or [],
                address=self.address,
            )
        )

    def parse(self, esil: str, address: int = 0) -> list[IRInstruction]:
        """Parse ESIL string to IR instructions.

        Args:
            esil: ESIL expression string
            address: Address of the original instruction

        Returns:
            List of IR instructions
        """
        self.reset(address)

        if not esil or esil == "":
            return []

        # Split by comma (ESIL separator)
        tokens = esil.split(",")

        for token in tokens:
            token = token.strip()
            if not token:
                continue

            self._process_token(token)

        return self.instructions

    def _process_token(self, token: str) -> None:
        """Process a single ESIL token."""

        # Check for memory operations first
        mem_store_match = re.match(r"=\[(\d+)\]", token)
        mem_load_match = re.match(r"\[(\d+)\]", token)

        if token == "=":
            # Assignment: pop value, pop destination (register), store
            if len(self.stack) >= 2:
                value = self.stack.pop()
                dest_name = self.stack.pop()

                if dest_name.kind == IROperand.Kind.REGISTER:
                    self._emit(IROpcode.MOV, dest_name, [value])
                elif dest_name.kind == IROperand.Kind.TEMPORARY:
                    self._emit(IROpcode.MOV, dest_name, [value])

        elif mem_store_match:
            # Memory store: =[size]
            size = int(mem_store_match.group(1)) * 8
            if len(self.stack) >= 2:
                value = self.stack.pop()
                addr = self.stack.pop()
                self._emit(IROpcode.STORE, None, [addr, value])

        elif mem_load_match:
            # Memory load: [size]
            size = int(mem_load_match.group(1)) * 8
            if len(self.stack) >= 1:
                addr = self.stack.pop()
                result = self._new_temp(size)
                self._emit(IROpcode.LOAD, result, [addr])
                self.stack.append(result)

        elif token in ESIL_OPCODE_MAP:
            # Binary/unary operations
            opcode = ESIL_OPCODE_MAP[token]

            if token == "!":
                # Unary NOT
                if self.stack:
                    operand = self.stack.pop()
                    result = self._new_temp()
                    self._emit(opcode, result, [operand])
                    self.stack.append(result)
            else:
                # Binary operation
                if len(self.stack) >= 2:
                    right = self.stack.pop()
                    left = self.stack.pop()
                    result = self._new_temp()
                    self._emit(opcode, result, [left, right])
                    self.stack.append(result)

        elif token == "GOTO":
            # Unconditional jump
            if self.stack:
                target = self.stack.pop()
                self._emit(IROpcode.JUMP, target)

        elif token == "?{":
            # Start of conditional block
            pass

        elif token == "}":
            # End of conditional block
            pass

        elif token == "BREAK":
            # Break from block
            pass

        elif token == "LOOP":
            # Loop marker
            pass

        elif token.startswith("$"):
            # Internal ESIL variable (flags, etc.)
            self.stack.append(IROperand.reg(token, 64))

        elif re.match(r"^0x[0-9a-fA-F]+$", token):
            # Hexadecimal immediate
            value = int(token, 16)
            self.stack.append(IROperand.imm(value))

        elif re.match(r"^-?[0-9]+$", token):
            # Decimal immediate
            value = int(token)
            self.stack.append(IROperand.imm(value))

        elif re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", token):
            # Register name
            self.stack.append(IROperand.reg(token, 64))

        else:
            # Unknown token - treat as register
            self.stack.append(IROperand.reg(token, 64))


class ESILLifter(IRLifter):
    """Lifter using radare2's ESIL.

    ESIL provides excellent coverage for embedded and exotic
    architectures through radare2's extensive architecture support.
    """

    def __init__(self):
        super().__init__()
        self.name = "esil"
        self.supported_isas = {
            # Tier 1-2 (also supported by VEX but ESIL can be fallback)
            "x86",
            "x86_64",
            "arm32",
            "arm64",
            "mips32",
            "mips64",
            "ppc32",
            "ppc64",
            "riscv32",
            "riscv64",
            "sparc",
            "sparc64",
            # Tier 3 (ESIL primary)
            "avr",
            "msp430",
            "xtensa",
            "arc",
            "tricore",
            "m68k",
            "sh4",
            "blackfin",
            "pic",
            "z80",
            "6502",
            # Tier 4
            "hexagon",
            "loongarch",
            "v850",
            "cris",
            "lm32",
            "or1k",
            # Additional embedded
            "arm_thumb",
            "arm_cortex",
            "rl78",
            "rx",
            "s390x",
        }
        self._r2_sessions: dict[str, Any] = {}
        self._parser = ESILParser()

    def _get_r2(self, binary_path: Path) -> Any:
        """Get or create r2pipe session for binary."""
        import r2pipe

        key = str(binary_path)
        if key not in self._r2_sessions:
            r2 = r2pipe.open(str(binary_path), flags=["-2"])
            r2.cmd("aaa")  # Analyze all
            self._r2_sessions[key] = r2
        return self._r2_sessions[key]

    def _close_r2(self, binary_path: Path) -> None:
        """Close r2pipe session."""
        key = str(binary_path)
        if key in self._r2_sessions:
            self._r2_sessions[key].quit()
            del self._r2_sessions[key]

    def lift_block(
        self,
        binary_path: Path,
        block_addr: int,
        max_size: int = 4096,
    ) -> LiftedBlock:
        """Lift a basic block using radare2 ESIL."""
        r2 = self._get_r2(binary_path)

        try:
            # Seek to address and get ESIL for instructions
            r2.cmd(f"s {block_addr}")

            # Get basic block info
            block_json = r2.cmdj(f"pdbj @ {block_addr}")

            if not block_json:
                # Fall back to linear disassembly
                block_json = r2.cmdj(f"pdj 32 @ {block_addr}")

            if not block_json:
                raise LiftingError(
                    f"Could not disassemble block at 0x{block_addr:x}",
                    address=block_addr,
                    backend=self.name,
                )

            instructions = []
            block_size = 0
            successors = []

            for instr_info in block_json:
                addr = instr_info.get("offset", 0)
                size = instr_info.get("size", 0)
                esil = instr_info.get("esil", "")
                opcode = instr_info.get("opcode", "")
                instr_type = instr_info.get("type", "")

                # Parse ESIL to IR
                ir_instrs = self._parser.parse(esil, addr)

                # Add comment with original disassembly
                for ir_instr in ir_instrs:
                    ir_instr.comment = opcode
                    ir_instr.size = size

                instructions.extend(ir_instrs)
                block_size += size

                # Check for control flow changes
                if instr_type in ("jmp", "cjmp", "call", "ret"):
                    if "jump" in instr_info:
                        successors.append(instr_info["jump"])
                    if instr_type == "cjmp" and addr + size not in successors:
                        successors.append(addr + size)  # Fall-through
                    break

            return LiftedBlock(
                address=block_addr,
                size=block_size,
                instructions=instructions,
                successors=successors,
            )

        except LiftingError:
            raise
        except Exception as e:
            raise LiftingError(
                f"Failed to lift block at 0x{block_addr:x}: {e}",
                address=block_addr,
                backend=self.name,
            )

    def lift_function(
        self,
        binary_path: Path,
        function_addr: int,
        function_size: Optional[int] = None,
    ) -> LiftedFunction:
        """Lift a function using radare2 analysis."""
        start_time = time.time()

        r2 = self._get_r2(binary_path)

        try:
            # Get function info
            r2.cmd(f"s {function_addr}")
            func_info = r2.cmdj(f"afij @ {function_addr}")

            if not func_info or len(func_info) == 0:
                # Try to define function
                r2.cmd(f"af @ {function_addr}")
                func_info = r2.cmdj(f"afij @ {function_addr}")

            if not func_info or len(func_info) == 0:
                raise LiftingError(
                    f"Could not find function at 0x{function_addr:x}",
                    address=function_addr,
                    backend=self.name,
                )

            func_info = func_info[0]
            func_name = func_info.get("name", None)
            func_size = func_info.get("size", 0)

            # Get basic blocks
            bb_json = r2.cmdj(f"afbj @ {function_addr}")

            if not bb_json:
                raise LiftingError(
                    f"Could not analyze basic blocks for function at 0x{function_addr:x}",
                    address=function_addr,
                    backend=self.name,
                )

            blocks = {}
            cfg_edges = []

            for bb_info in bb_json:
                bb_addr = bb_info.get("addr", 0)
                bb_size = bb_info.get("size", 0)

                # Get ESIL for this block
                block_disasm = r2.cmdj(f"pdbj @ {bb_addr}")

                if not block_disasm:
                    continue

                instructions = []
                for instr_info in block_disasm:
                    addr = instr_info.get("offset", 0)
                    size = instr_info.get("size", 0)
                    esil = instr_info.get("esil", "")
                    opcode = instr_info.get("opcode", "")

                    ir_instrs = self._parser.parse(esil, addr)
                    for ir_instr in ir_instrs:
                        ir_instr.comment = opcode
                        ir_instr.size = size
                    instructions.extend(ir_instrs)

                # Get successors from radare2 analysis
                successors = []
                if "jump" in bb_info:
                    successors.append(bb_info["jump"])
                if "fail" in bb_info:
                    successors.append(bb_info["fail"])

                block = LiftedBlock(
                    address=bb_addr,
                    size=bb_size,
                    instructions=instructions,
                    successors=successors,
                )
                blocks[bb_addr] = block

                # Add CFG edges
                for succ in successors:
                    cfg_edges.append((bb_addr, succ))

            # Update predecessors
            for src, dst in cfg_edges:
                if dst in blocks:
                    blocks[dst].predecessors.append(src)

            # Mark entry block
            if function_addr in blocks:
                blocks[function_addr].is_entry = True

            lift_time = (time.time() - start_time) * 1000

            # Get architecture info
            arch_info = r2.cmdj("iAj")
            isa = arch_info.get("arch", "unknown") if arch_info else "unknown"

            return LiftedFunction(
                address=function_addr,
                size=func_size,
                name=func_name if func_name and not func_name.startswith("fcn.") else None,
                blocks=blocks,
                entry_block=function_addr,
                cfg_edges=cfg_edges,
                num_args=func_info.get("nargs", 0),
                lifter_backend=self.name,
                lift_time_ms=lift_time,
                isa=isa,
            )

        except LiftingError:
            raise
        except Exception as e:
            raise LiftingError(
                f"Failed to lift function at 0x{function_addr:x}: {e}",
                address=function_addr,
                backend=self.name,
            )

    def get_priority(self, isa: str) -> int:
        """ESIL has priority for embedded architectures."""
        tier3 = {
            "avr",
            "msp430",
            "xtensa",
            "arc",
            "tricore",
            "m68k",
            "sh4",
            "blackfin",
            "pic",
            "z80",
            "6502",
        }

        isa_lower = isa.lower()
        if isa_lower in tier3:
            return 10  # Best choice for these
        elif self.supports_isa(isa_lower):
            return 3  # Fallback for others
        return 0

    def __del__(self):
        """Clean up r2pipe sessions."""
        for r2 in self._r2_sessions.values():
            try:
                r2.quit()
            except Exception:
                pass
