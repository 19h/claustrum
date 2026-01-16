"""VEX IR lifter using angr/pyvex.

VEX IR is the intermediate representation from Valgrind, providing
high-quality lifting for Tier 1 architectures: x86, x86_64, ARM32/64,
MIPS32/64, PPC32/64, and S390X.
"""

from __future__ import annotations

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


# VEX opcode mapping to unified IR
VEX_OPCODE_MAP = {
    # Integer arithmetic
    "Iop_Add8": IROpcode.ADD,
    "Iop_Add16": IROpcode.ADD,
    "Iop_Add32": IROpcode.ADD,
    "Iop_Add64": IROpcode.ADD,
    "Iop_Sub8": IROpcode.SUB,
    "Iop_Sub16": IROpcode.SUB,
    "Iop_Sub32": IROpcode.SUB,
    "Iop_Sub64": IROpcode.SUB,
    "Iop_Mul8": IROpcode.MUL,
    "Iop_Mul16": IROpcode.MUL,
    "Iop_Mul32": IROpcode.MUL,
    "Iop_Mul64": IROpcode.MUL,
    "Iop_DivU32": IROpcode.DIV,
    "Iop_DivU64": IROpcode.DIV,
    "Iop_DivS32": IROpcode.SDIV,
    "Iop_DivS64": IROpcode.SDIV,
    # Bitwise
    "Iop_And8": IROpcode.AND,
    "Iop_And16": IROpcode.AND,
    "Iop_And32": IROpcode.AND,
    "Iop_And64": IROpcode.AND,
    "Iop_Or8": IROpcode.OR,
    "Iop_Or16": IROpcode.OR,
    "Iop_Or32": IROpcode.OR,
    "Iop_Or64": IROpcode.OR,
    "Iop_Xor8": IROpcode.XOR,
    "Iop_Xor16": IROpcode.XOR,
    "Iop_Xor32": IROpcode.XOR,
    "Iop_Xor64": IROpcode.XOR,
    "Iop_Not8": IROpcode.NOT,
    "Iop_Not16": IROpcode.NOT,
    "Iop_Not32": IROpcode.NOT,
    "Iop_Not64": IROpcode.NOT,
    "Iop_Shl8": IROpcode.SHL,
    "Iop_Shl16": IROpcode.SHL,
    "Iop_Shl32": IROpcode.SHL,
    "Iop_Shl64": IROpcode.SHL,
    "Iop_Shr8": IROpcode.SHR,
    "Iop_Shr16": IROpcode.SHR,
    "Iop_Shr32": IROpcode.SHR,
    "Iop_Shr64": IROpcode.SHR,
    "Iop_Sar8": IROpcode.SAR,
    "Iop_Sar16": IROpcode.SAR,
    "Iop_Sar32": IROpcode.SAR,
    "Iop_Sar64": IROpcode.SAR,
    # Comparisons
    "Iop_CmpEQ8": IROpcode.CMP_EQ,
    "Iop_CmpEQ16": IROpcode.CMP_EQ,
    "Iop_CmpEQ32": IROpcode.CMP_EQ,
    "Iop_CmpEQ64": IROpcode.CMP_EQ,
    "Iop_CmpNE8": IROpcode.CMP_NE,
    "Iop_CmpNE16": IROpcode.CMP_NE,
    "Iop_CmpNE32": IROpcode.CMP_NE,
    "Iop_CmpNE64": IROpcode.CMP_NE,
    "Iop_CmpLT32S": IROpcode.CMP_LT,
    "Iop_CmpLT64S": IROpcode.CMP_LT,
    "Iop_CmpLE32S": IROpcode.CMP_LE,
    "Iop_CmpLE64S": IROpcode.CMP_LE,
    "Iop_CmpLT32U": IROpcode.CMP_ULT,
    "Iop_CmpLT64U": IROpcode.CMP_ULT,
    "Iop_CmpLE32U": IROpcode.CMP_ULE,
    "Iop_CmpLE64U": IROpcode.CMP_ULE,
    # Conversions
    "Iop_8Uto16": IROpcode.ZEXT,
    "Iop_8Uto32": IROpcode.ZEXT,
    "Iop_8Uto64": IROpcode.ZEXT,
    "Iop_16Uto32": IROpcode.ZEXT,
    "Iop_16Uto64": IROpcode.ZEXT,
    "Iop_32Uto64": IROpcode.ZEXT,
    "Iop_8Sto16": IROpcode.SEXT,
    "Iop_8Sto32": IROpcode.SEXT,
    "Iop_8Sto64": IROpcode.SEXT,
    "Iop_16Sto32": IROpcode.SEXT,
    "Iop_16Sto64": IROpcode.SEXT,
    "Iop_32Sto64": IROpcode.SEXT,
    "Iop_64to32": IROpcode.TRUNC,
    "Iop_32to16": IROpcode.TRUNC,
    "Iop_16to8": IROpcode.TRUNC,
    # Floating point
    "Iop_AddF32": IROpcode.FADD,
    "Iop_AddF64": IROpcode.FADD,
    "Iop_SubF32": IROpcode.FSUB,
    "Iop_SubF64": IROpcode.FSUB,
    "Iop_MulF32": IROpcode.FMUL,
    "Iop_MulF64": IROpcode.FMUL,
    "Iop_DivF32": IROpcode.FDIV,
    "Iop_DivF64": IROpcode.FDIV,
    "Iop_CmpF32": IROpcode.FCMP,
    "Iop_CmpF64": IROpcode.FCMP,
}


class VEXLifter(IRLifter):
    """Lifter using angr/pyvex for VEX IR.

    Provides high-quality lifting for well-supported architectures.
    Uses angr's CFG analysis for function discovery and block boundaries.
    """

    def __init__(self):
        super().__init__()
        self.name = "vex"
        self.supported_isas = {
            "x86",
            "x86_64",
            "amd64",
            "arm32",
            "arm",
            "armel",
            "armhf",
            "arm64",
            "aarch64",
            "mips32",
            "mips",
            "mipsel",
            "mips64",
            "mips64el",
            "ppc32",
            "ppc",
            "ppc64",
            "ppc64le",
            "s390x",
        }
        self._project_cache: dict[str, Any] = {}

    def _get_project(self, binary_path: Path) -> Any:
        """Get or create angr project for binary."""
        import angr

        key = str(binary_path)
        if key not in self._project_cache:
            self._project_cache[key] = angr.Project(
                str(binary_path),
                auto_load_libs=False,
                load_options={"auto_load_libs": False},
            )
        return self._project_cache[key]

    def _convert_vex_expr(
        self, expr: Any, temp_counter: int
    ) -> tuple[IROperand, list[IRInstruction], int]:
        """Convert VEX expression to IR operand and any needed instructions.

        Returns (operand, extra_instructions, new_temp_counter)
        """
        import pyvex

        extra_instrs = []

        if isinstance(expr, pyvex.IRExpr.RdTmp):
            return IROperand.tmp(expr.tmp, 64), extra_instrs, temp_counter

        elif isinstance(expr, pyvex.IRExpr.Const):
            value = expr.con.value
            size = expr.con.size
            return IROperand.imm(value, size), extra_instrs, temp_counter

        elif isinstance(expr, pyvex.IRExpr.Get):
            # Register read
            offset = expr.offset
            size = expr.result_size(None) if hasattr(expr, "result_size") else 64  # type: ignore[arg-type]
            return IROperand.reg(f"r{offset}", size), extra_instrs, temp_counter

        elif isinstance(expr, pyvex.IRExpr.Load):
            # Memory load - create load instruction
            addr_op, addr_instrs, temp_counter = self._convert_vex_expr(expr.addr, temp_counter)
            extra_instrs.extend(addr_instrs)

            result = IROperand.tmp(temp_counter, 64)
            temp_counter += 1

            load_instr = IRInstruction(
                opcode=IROpcode.LOAD,
                dest=result,
                src=[addr_op],
            )
            extra_instrs.append(load_instr)
            return result, extra_instrs, temp_counter

        elif isinstance(expr, pyvex.IRExpr.Unop):
            # Unary operation
            arg_op, arg_instrs, temp_counter = self._convert_vex_expr(expr.args[0], temp_counter)
            extra_instrs.extend(arg_instrs)

            op_name = expr.op
            ir_opcode = VEX_OPCODE_MAP.get(op_name, IROpcode.UNKNOWN)

            result = IROperand.tmp(temp_counter, 64)
            temp_counter += 1

            unop_instr = IRInstruction(
                opcode=ir_opcode,
                dest=result,
                src=[arg_op],
            )
            extra_instrs.append(unop_instr)
            return result, extra_instrs, temp_counter

        elif isinstance(expr, pyvex.IRExpr.Binop):
            # Binary operation
            left_op, left_instrs, temp_counter = self._convert_vex_expr(expr.args[0], temp_counter)
            extra_instrs.extend(left_instrs)

            right_op, right_instrs, temp_counter = self._convert_vex_expr(
                expr.args[1], temp_counter
            )
            extra_instrs.extend(right_instrs)

            op_name = expr.op
            ir_opcode = VEX_OPCODE_MAP.get(op_name, IROpcode.UNKNOWN)

            result = IROperand.tmp(temp_counter, 64)
            temp_counter += 1

            binop_instr = IRInstruction(
                opcode=ir_opcode,
                dest=result,
                src=[left_op, right_op],
            )
            extra_instrs.append(binop_instr)
            return result, extra_instrs, temp_counter

        elif isinstance(expr, pyvex.IRExpr.ITE):
            # If-then-else expression
            cond_op, cond_instrs, temp_counter = self._convert_vex_expr(expr.cond, temp_counter)
            extra_instrs.extend(cond_instrs)

            true_op, true_instrs, temp_counter = self._convert_vex_expr(expr.iftrue, temp_counter)
            extra_instrs.extend(true_instrs)

            false_op, false_instrs, temp_counter = self._convert_vex_expr(
                expr.iffalse, temp_counter
            )
            extra_instrs.extend(false_instrs)

            # Represent as conditional move
            result = IROperand.tmp(temp_counter, 64)
            temp_counter += 1

            # Simplified: just return the true operand for now
            return true_op, extra_instrs, temp_counter

        else:
            # Unknown expression type
            return IROperand.imm(0, 64), extra_instrs, temp_counter

    def _convert_vex_stmt(
        self,
        stmt: Any,
        temp_counter: int,
        block_addr: int,
    ) -> tuple[list[IRInstruction], int]:
        """Convert VEX statement to IR instructions.

        Returns (instructions, new_temp_counter)
        """
        import pyvex

        instrs = []

        if isinstance(stmt, pyvex.IRStmt.WrTmp):
            # Temporary assignment
            result_op = IROperand.tmp(stmt.tmp, 64)
            data_op, extra_instrs, temp_counter = self._convert_vex_expr(stmt.data, temp_counter)
            instrs.extend(extra_instrs)

            # The result is already computed in extra_instrs, just need assignment
            if not extra_instrs:
                instr = IRInstruction(
                    opcode=IROpcode.MOV,
                    dest=result_op,
                    src=[data_op],
                    address=block_addr,
                )
                instrs.append(instr)

        elif isinstance(stmt, pyvex.IRStmt.Put):
            # Register write
            reg_op = IROperand.reg(f"r{stmt.offset}", 64)
            data_op, extra_instrs, temp_counter = self._convert_vex_expr(stmt.data, temp_counter)
            instrs.extend(extra_instrs)

            instr = IRInstruction(
                opcode=IROpcode.MOV,
                dest=reg_op,
                src=[data_op],
                address=block_addr,
            )
            instrs.append(instr)

        elif isinstance(stmt, pyvex.IRStmt.Store):
            # Memory store
            addr_op, addr_instrs, temp_counter = self._convert_vex_expr(stmt.addr, temp_counter)
            instrs.extend(addr_instrs)

            data_op, data_instrs, temp_counter = self._convert_vex_expr(stmt.data, temp_counter)
            instrs.extend(data_instrs)

            instr = IRInstruction(
                opcode=IROpcode.STORE,
                dest=None,
                src=[addr_op, data_op],
                address=block_addr,
            )
            instrs.append(instr)

        elif isinstance(stmt, pyvex.IRStmt.Exit):
            # Conditional exit (branch)
            guard_op, guard_instrs, temp_counter = self._convert_vex_expr(stmt.guard, temp_counter)
            instrs.extend(guard_instrs)

            target = stmt.dst.value if hasattr(stmt.dst, "value") else 0

            instr = IRInstruction(
                opcode=IROpcode.BRANCH,
                dest=IROperand.label(target),
                src=[guard_op],
                address=block_addr,
                condition=str(stmt.jk) if hasattr(stmt, "jk") else None,
            )
            instrs.append(instr)

        elif isinstance(stmt, pyvex.IRStmt.IMark):
            # Instruction marker - useful for address tracking
            pass

        elif isinstance(stmt, pyvex.IRStmt.AbiHint):
            # ABI hint - skip
            pass

        elif isinstance(stmt, pyvex.IRStmt.NoOp):
            # No operation
            pass

        return instrs, temp_counter

    def _convert_vex_block(self, vex_block: Any, block_addr: int) -> LiftedBlock:
        """Convert a pyvex IRSB to LiftedBlock."""
        instructions = []
        temp_counter = 1000  # Start high to avoid conflicts

        for stmt in vex_block.statements:
            stmt_instrs, temp_counter = self._convert_vex_stmt(stmt, temp_counter, block_addr)
            instructions.extend(stmt_instrs)

        # Handle the block's exit (jump target)
        successors = []
        if vex_block.jumpkind == "Ijk_Boring":
            # Unconditional jump
            if hasattr(vex_block, "next") and hasattr(vex_block.next, "con"):
                target = vex_block.next.con.value
                successors.append(target)

                instr = IRInstruction(
                    opcode=IROpcode.JUMP,
                    dest=IROperand.label(target),
                    address=block_addr,
                )
                instructions.append(instr)

        elif vex_block.jumpkind == "Ijk_Call":
            # Function call
            if hasattr(vex_block, "next") and hasattr(vex_block.next, "con"):
                target = vex_block.next.con.value

                instr = IRInstruction(
                    opcode=IROpcode.CALL,
                    dest=IROperand.label(target),
                    address=block_addr,
                )
                instructions.append(instr)

        elif vex_block.jumpkind == "Ijk_Ret":
            # Return
            instr = IRInstruction(
                opcode=IROpcode.RET,
                address=block_addr,
            )
            instructions.append(instr)

        return LiftedBlock(
            address=block_addr,
            size=vex_block.size,
            instructions=instructions,
            successors=successors,
        )

    def lift_block(
        self,
        binary_path: Path,
        block_addr: int,
        max_size: int = 4096,
    ) -> LiftedBlock:
        """Lift a single basic block using pyvex."""
        import pyvex

        proj = self._get_project(binary_path)

        try:
            # Get the bytes at the address
            state = proj.factory.blank_state(addr=block_addr)

            # Lift to VEX
            vex_block = pyvex.lift(
                proj.loader.memory.load(block_addr, max_size),
                block_addr,
                proj.arch,
            )

            return self._convert_vex_block(vex_block, block_addr)

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
        """Lift a complete function using angr CFG analysis."""
        start_time = time.time()

        proj = self._get_project(binary_path)

        try:
            # Use angr's CFG for function analysis
            cfg = proj.analyses.CFGFast(
                regions=[(function_addr, function_addr + (function_size or 0x10000))],
                normalize=True,
                data_references=False,
            )

            # Find the function
            func = cfg.kb.functions.get(function_addr)
            if func is None:
                # Try to find function containing this address
                for f in cfg.kb.functions.values():
                    if f.addr <= function_addr < f.addr + f.size:
                        func = f
                        break

            if func is None:
                raise LiftingError(
                    f"Could not find function at 0x{function_addr:x}",
                    address=function_addr,
                    backend=self.name,
                )

            # Lift all blocks in the function
            blocks = {}
            cfg_edges = []

            for block_node in func.graph.nodes():
                block_addr = block_node.addr

                try:
                    # Get VEX block from angr
                    angr_block = proj.factory.block(block_addr)
                    vex_block = angr_block.vex

                    lifted_block = self._convert_vex_block(vex_block, block_addr)
                    blocks[block_addr] = lifted_block
                except Exception:
                    # Skip problematic blocks
                    continue

            # Extract CFG edges
            for src, dst in func.graph.edges():
                cfg_edges.append((src.addr, dst.addr))
                if src.addr in blocks:
                    blocks[src.addr].successors.append(dst.addr)
                if dst.addr in blocks:
                    blocks[dst.addr].predecessors.append(src.addr)

            # Mark entry block
            if function_addr in blocks:
                blocks[function_addr].is_entry = True

            lift_time = (time.time() - start_time) * 1000

            return LiftedFunction(
                address=function_addr,
                size=func.size,
                name=func.name if func.name != f"sub_{function_addr:x}" else None,
                blocks=blocks,
                entry_block=function_addr,
                cfg_edges=cfg_edges,
                lifter_backend=self.name,
                lift_time_ms=lift_time,
                isa=proj.arch.name.lower(),
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
        """VEX has highest priority for Tier 1 architectures."""
        tier1 = {
            "x86",
            "x86_64",
            "amd64",
            "arm32",
            "arm",
            "arm64",
            "aarch64",
            "mips32",
            "mips",
            "mips64",
            "ppc32",
            "ppc",
            "ppc64",
        }

        isa_lower = isa.lower()
        if isa_lower in tier1:
            return 10
        elif self.supports_isa(isa_lower):
            return 5
        return 0
