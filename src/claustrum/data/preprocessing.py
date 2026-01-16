"""Preprocessing pipeline for building CLAUSTRUM datasets.

Converts raw binaries into tokenized IR sequences ready for training.
Supports:
    - Single binary processing
    - Batch processing from directories
    - Cross-compilation dataset building from paired data
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from claustrum.lifting import TieredLifter
from claustrum.normalization import IRNormalizer
from claustrum.tokenization import IRTokenizer
from claustrum.utils.types import ISA


@dataclass
class ProcessedFunction:
    """Result of processing a single binary function."""

    function_id: str
    source_id: str
    isa: str
    ir_tokens: list[int]
    cfg_edges: list[tuple[int, int]]
    raw_bytes: Optional[bytes] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BinaryProcessor:
    """Processes binary files into tokenized IR sequences.

    Pipeline:
        1. Lift binary to IR (VEX/ESIL)
        2. Normalize IR to architecture-neutral form
        3. Tokenize normalized IR
        4. Extract CFG structure

    Args:
        tokenizer: IR tokenizer instance
        lifter: IR lifter (auto-selects if None)
        normalizer: IR normalizer (auto-configured if None)
        max_length: Maximum token sequence length
    """

    def __init__(
        self,
        tokenizer: Optional[IRTokenizer] = None,
        lifter: Optional[TieredLifter] = None,
        normalizer: Optional[IRNormalizer] = None,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer or IRTokenizer()
        self.lifter = lifter or TieredLifter()
        self.normalizer = normalizer or IRNormalizer()
        self.max_length = max_length

    def process_binary(
        self,
        binary_path: Path,
        isa: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> list[ProcessedFunction]:
        """Process all functions in a binary file.

        Args:
            binary_path: Path to binary file
            isa: ISA name (auto-detected if None)
            source_id: Source identifier for linking cross-ISA pairs

        Returns:
            List of processed functions
        """
        results = []
        binary_path = Path(binary_path)

        # Generate source_id from path if not provided
        if source_id is None:
            source_id = binary_path.stem

        try:
            # Lift binary to IR
            lifted_functions = self.lifter.lift_binary(
                str(binary_path),
                isa=isa,
            )

            for func in lifted_functions:
                try:
                    # Normalize IR
                    normalized = self.normalizer.normalize(func.ir_blocks)

                    # Tokenize
                    tokens = self.tokenizer.encode(normalized)
                    tokens = tokens[: self.max_length]

                    # Extract CFG edges
                    cfg_edges = self._extract_cfg_edges(func)

                    # Generate function ID
                    func_id = self._generate_function_id(binary_path, func.name, func.address)

                    results.append(
                        ProcessedFunction(
                            function_id=func_id,
                            source_id=source_id,
                            isa=func.isa or isa or "unknown",
                            ir_tokens=tokens,
                            cfg_edges=cfg_edges,
                            metadata={
                                "name": func.name,
                                "address": hex(func.address) if func.address else None,
                                "size": func.size,
                                "binary": binary_path.name,
                            },
                        )
                    )

                except Exception as e:
                    results.append(
                        ProcessedFunction(
                            function_id=f"{binary_path.stem}_{func.name}",
                            source_id=source_id,
                            isa=isa or "unknown",
                            ir_tokens=[],
                            cfg_edges=[],
                            error=str(e),
                        )
                    )

        except Exception as e:
            # Binary-level error
            results.append(
                ProcessedFunction(
                    function_id=f"{binary_path.stem}_error",
                    source_id=source_id,
                    isa=isa or "unknown",
                    ir_tokens=[],
                    cfg_edges=[],
                    error=f"Binary processing failed: {e}",
                )
            )

        return results

    def process_function_bytes(
        self,
        func_bytes: bytes,
        isa: str,
        address: int = 0,
        source_id: Optional[str] = None,
        function_name: Optional[str] = None,
    ) -> ProcessedFunction:
        """Process raw function bytes directly.

        Args:
            func_bytes: Raw machine code bytes
            isa: Target ISA
            address: Base address of function
            source_id: Source identifier
            function_name: Function name

        Returns:
            Processed function
        """
        try:
            # Lift bytes to IR
            ir_blocks = self.lifter.lift_bytes(
                func_bytes,
                isa=isa,
                address=address,
            )

            # Normalize
            normalized = self.normalizer.normalize(ir_blocks)

            # Tokenize
            tokens = self.tokenizer.encode(normalized)
            tokens = tokens[: self.max_length]

            # Generate IDs
            func_id = self._generate_function_id_from_bytes(func_bytes, isa)
            source_id = source_id or func_id

            return ProcessedFunction(
                function_id=func_id,
                source_id=source_id,
                isa=isa,
                ir_tokens=tokens,
                cfg_edges=[],  # CFG not available for raw bytes
                raw_bytes=func_bytes,
                metadata={
                    "name": function_name,
                    "address": hex(address),
                    "size": len(func_bytes),
                },
            )

        except Exception as e:
            return ProcessedFunction(
                function_id=self._generate_function_id_from_bytes(func_bytes, isa),
                source_id=source_id or "unknown",
                isa=isa,
                ir_tokens=[],
                cfg_edges=[],
                error=str(e),
            )

    def _extract_cfg_edges(self, func: Any) -> list[tuple[int, int]]:
        """Extract CFG edges from lifted function."""
        edges = []

        if hasattr(func, "cfg") and func.cfg:
            # If we have a proper CFG object
            for block_idx, block in enumerate(func.cfg.blocks):
                for succ in block.successors:
                    succ_idx = func.cfg.block_to_index.get(succ.address, -1)
                    if succ_idx >= 0:
                        edges.append((block_idx, succ_idx))
        elif hasattr(func, "ir_blocks"):
            # Infer simple sequential edges from IR blocks
            for i in range(len(func.ir_blocks) - 1):
                edges.append((i, i + 1))

        return edges

    def _generate_function_id(
        self,
        binary_path: Path,
        func_name: str,
        address: Optional[int],
    ) -> str:
        """Generate unique function identifier."""
        components = [binary_path.name, func_name]
        if address:
            components.append(hex(address))
        content = "_".join(str(c) for c in components)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _generate_function_id_from_bytes(
        self,
        func_bytes: bytes,
        isa: str,
    ) -> str:
        """Generate function ID from bytes content."""
        content = isa.encode() + func_bytes
        return hashlib.sha256(content).hexdigest()[:16]


class DatasetBuilder:
    """Builds training datasets from binary collections.

    Supports:
        - Directories of binaries organized by ISA
        - Cross-compilation metadata files
        - Sharded output for large datasets

    Args:
        processor: Binary processor instance
        output_dir: Directory for output files
        shard_size: Number of samples per shard file
        num_workers: Parallel processing workers
    """

    def __init__(
        self,
        processor: Optional[BinaryProcessor] = None,
        output_dir: str = "data/processed",
        shard_size: int = 10000,
        num_workers: int = 4,
    ):
        self.processor = processor or BinaryProcessor()
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.num_workers = num_workers

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_from_directory(
        self,
        input_dir: Path,
        isa_mapping: Optional[dict[str, str]] = None,
    ) -> dict[str, int]:
        """Build dataset from directory of binaries.

        Expected structure:
            input_dir/
                x86/
                    binary1, binary2, ...
                arm64/
                    binary1, binary2, ...

        Or flat structure with isa_mapping providing ISA info.

        Args:
            input_dir: Directory containing binaries
            isa_mapping: Optional mapping of subdirectory names to ISA names

        Returns:
            Statistics about processed data
        """
        input_dir = Path(input_dir)

        # Collect binaries with ISA info
        binary_tasks = []

        # Check for ISA subdirectories
        subdirs = [d for d in input_dir.iterdir() if d.is_dir()]

        if subdirs:
            for subdir in subdirs:
                isa = isa_mapping.get(subdir.name, subdir.name) if isa_mapping else subdir.name

                for binary_path in subdir.glob("*"):
                    if binary_path.is_file():
                        binary_tasks.append((binary_path, isa))
        else:
            # Flat directory - need ISA info
            for binary_path in input_dir.glob("*"):
                if binary_path.is_file():
                    # Try to infer ISA from filename
                    isa = self._infer_isa(binary_path)
                    binary_tasks.append((binary_path, isa))

        return self._process_binaries(binary_tasks)

    def build_from_cross_compilation(
        self,
        metadata_file: Path,
        binaries_dir: Path,
    ) -> dict[str, int]:
        """Build dataset from cross-compilation metadata.

        Metadata JSON format:
            {
                "source_functions": [
                    {
                        "source_id": "func_123",
                        "source_file": "foo.c",
                        "function_name": "bar",
                        "compilations": [
                            {"isa": "x86_64", "binary": "foo_x64.o", "address": "0x1000"},
                            {"isa": "arm64", "binary": "foo_arm64.o", "address": "0x800"},
                            ...
                        ]
                    },
                    ...
                ]
            }

        Args:
            metadata_file: Path to cross-compilation metadata JSON
            binaries_dir: Directory containing compiled binaries

        Returns:
            Processing statistics
        """
        metadata_file = Path(metadata_file)
        binaries_dir = Path(binaries_dir)

        with open(metadata_file) as f:
            metadata = json.load(f)

        binary_tasks = []
        source_id_map = {}  # binary_path -> source_id

        for source_func in metadata.get("source_functions", []):
            source_id = source_func["source_id"]

            for comp in source_func.get("compilations", []):
                binary_path = binaries_dir / comp["binary"]
                isa = comp["isa"]

                if binary_path.exists():
                    binary_tasks.append((binary_path, isa))
                    source_id_map[str(binary_path)] = source_id

        return self._process_binaries(binary_tasks, source_id_map)

    def build_from_pairs(
        self,
        pairs: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Build dataset from explicit function pairs.

        Each pair dict should have:
            - source_id: str
            - functions: list of {isa, bytes, address, name}

        Args:
            pairs: List of function pair specifications

        Returns:
            Processing statistics
        """
        all_results = []

        for pair in tqdm(pairs, desc="Processing pairs"):
            source_id = pair["source_id"]

            for func_spec in pair.get("functions", []):
                result = self.processor.process_function_bytes(
                    func_bytes=func_spec["bytes"],
                    isa=func_spec["isa"],
                    address=func_spec.get("address", 0),
                    source_id=source_id,
                    function_name=func_spec.get("name"),
                )
                all_results.append(result)

        return self._save_results(all_results)

    def _process_binaries(
        self,
        binary_tasks: list[tuple[Path, str]],
        source_id_map: Optional[dict[str, str]] = None,
    ) -> dict[str, int]:
        """Process binary files in parallel.

        Args:
            binary_tasks: List of (binary_path, isa) tuples
            source_id_map: Optional mapping of binary paths to source IDs

        Returns:
            Processing statistics
        """
        source_id_map = source_id_map or {}
        all_results = []

        if self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {}

                for binary_path, isa in binary_tasks:
                    source_id = source_id_map.get(str(binary_path))
                    future = executor.submit(
                        self.processor.process_binary,
                        binary_path,
                        isa,
                        source_id,
                    )
                    futures[future] = binary_path

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing binaries",
                ):
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        binary_path = futures[future]
                        print(f"Error processing {binary_path}: {e}")
        else:
            for binary_path, isa in tqdm(binary_tasks, desc="Processing binaries"):
                source_id = source_id_map.get(str(binary_path))
                results = self.processor.process_binary(binary_path, isa, source_id)
                all_results.extend(results)

        return self._save_results(all_results)

    def _save_results(
        self,
        results: list[ProcessedFunction],
    ) -> dict[str, int]:
        """Save processed results to Parquet shards.

        Args:
            results: List of processed functions

        Returns:
            Statistics dict
        """
        # Filter out errors
        valid_results = [r for r in results if not r.error and r.ir_tokens]
        error_count = len(results) - len(valid_results)

        # Convert to Arrow table
        schema = pa.schema(
            [
                ("function_id", pa.string()),
                ("source_id", pa.string()),
                ("isa", pa.string()),
                ("ir_tokens", pa.list_(pa.int32())),
                ("cfg_edges", pa.string()),  # JSON-encoded
                ("metadata", pa.string()),  # JSON-encoded
            ]
        )

        # Write shards
        shard_idx = 0

        for i in range(0, len(valid_results), self.shard_size):
            shard_results = valid_results[i : i + self.shard_size]

            arrays = [
                pa.array([r.function_id for r in shard_results]),
                pa.array([r.source_id for r in shard_results]),
                pa.array([r.isa for r in shard_results]),
                pa.array([r.ir_tokens for r in shard_results]),
                pa.array([json.dumps(r.cfg_edges) for r in shard_results]),
                pa.array([json.dumps(r.metadata) for r in shard_results]),
            ]

            table = pa.Table.from_arrays(arrays, schema=schema)

            shard_path = self.output_dir / f"shard_{shard_idx:05d}.parquet"
            pq.write_table(table, shard_path)

            shard_idx += 1

        # Write metadata
        metadata = {
            "total_functions": len(valid_results),
            "total_errors": error_count,
            "num_shards": shard_idx,
            "isas": list({r.isa for r in valid_results}),
            "source_functions": len({r.source_id for r in valid_results}),
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _infer_isa(self, binary_path: Path) -> str:
        """Attempt to infer ISA from binary filename or content."""
        name = binary_path.name.lower()

        isa_patterns = {
            "x86_64": ["x86_64", "x64", "amd64"],
            "x86": ["x86", "i386", "i686"],
            "arm64": ["arm64", "aarch64"],
            "arm32": ["arm32", "armv7", "arm"],
            "mips64": ["mips64"],
            "mips32": ["mips32", "mips"],
            "riscv64": ["riscv64", "rv64"],
            "riscv32": ["riscv32", "rv32"],
            "ppc64": ["ppc64", "powerpc64"],
            "ppc32": ["ppc32", "powerpc"],
        }

        for isa, patterns in isa_patterns.items():
            for pattern in patterns:
                if pattern in name:
                    return isa

        return "unknown"


def create_train_val_split(
    data_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Split processed data into train/validation sets.

    Splits by source_id to ensure no cross-ISA pairs are split.

    Args:
        data_dir: Directory with processed Parquet files
        output_dir: Output directory for split data
        val_ratio: Fraction for validation
        seed: Random seed
    """
    import random

    random.seed(seed)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Load all samples
    samples = []
    for pq_file in data_dir.glob("*.parquet"):
        table = pq.read_table(pq_file)
        samples.extend(table.to_pydict())

    # Group by source_id
    source_to_samples: dict[str, list[int]] = {}
    for idx, sample in enumerate(samples):
        source_id = sample["source_id"]
        if source_id not in source_to_samples:
            source_to_samples[source_id] = []
        source_to_samples[source_id].append(idx)

    # Split source_ids
    source_ids = list(source_to_samples.keys())
    random.shuffle(source_ids)

    val_count = int(len(source_ids) * val_ratio)
    val_sources = set(source_ids[:val_count])
    train_sources = set(source_ids[val_count:])

    # Write splits
    for split_name, split_sources in [("train", train_sources), ("val", val_sources)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_indices = []
        for source_id in split_sources:
            split_indices.extend(source_to_samples[source_id])

        # TODO: Write split to Parquet
        print(f"{split_name}: {len(split_indices)} samples from {len(split_sources)} sources")
