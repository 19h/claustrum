"""Dataset classes for CLAUSTRUM training and evaluation.

Provides dataset implementations for:
- Binary function samples with IR sequences
- Cross-ISA paired data for contrastive learning
- Pretraining datasets with masking
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator, Any, Union

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset, IterableDataset

from claustrum.utils.types import ISA, FunctionMetadata
from claustrum.normalization.normalized_ir import NormalizedInstruction


@dataclass
class FunctionSample:
    """A single function sample for training/inference.

    Attributes:
        function_id: Unique identifier for the function
        source_id: Identifier linking functions from same source code
        isa: Target instruction set architecture
        ir_tokens: Tokenized normalized IR sequence
        cfg_edges: Control flow graph edge list (optional)
        metadata: Additional function metadata
    """

    function_id: str
    source_id: str  # Links cross-ISA pairs from same source
    isa: str
    ir_tokens: list[int]
    cfg_edges: Optional[list[tuple[int, int]]] = None
    metadata: Optional[dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self.ir_tokens)


@dataclass
class CrossISABatch:
    """Batch of cross-ISA samples grouped by source function.

    For contrastive learning, each source function has multiple ISA variants.
    """

    source_ids: list[str]
    samples: list[FunctionSample]
    source_to_indices: dict[str, list[int]]  # Maps source_id to sample indices


class BinaryFunctionDataset(Dataset):
    """Dataset for binary function samples.

    Loads pre-processed function data from Parquet files with structure:
        - function_id: str
        - source_id: str
        - isa: str
        - ir_tokens: list[int]
        - cfg_edges: list[tuple[int, int]] (optional)
        - metadata: dict (optional)

    Args:
        data_path: Path to data directory or Parquet file
        max_length: Maximum sequence length (truncates longer sequences)
        filter_isas: Only include samples from these ISAs (None = all)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        max_length: int = 512,
        filter_isas: Optional[list[str]] = None,
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.filter_isas = set(filter_isas) if filter_isas else None

        self.samples: list[FunctionSample] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load samples from Parquet files."""
        if self.data_path.is_file():
            parquet_files = [self.data_path]
        else:
            parquet_files = list(self.data_path.glob("**/*.parquet"))

        if not parquet_files:
            # Try JSON format as fallback
            json_files = list(self.data_path.glob("**/*.json"))
            if json_files:
                self._load_json_data(json_files)
                return
            raise ValueError(f"No data files found in {self.data_path}")

        for pq_file in parquet_files:
            table = pq.read_table(pq_file)
            df = table.to_pandas()

            for _, row in df.iterrows():
                if self.filter_isas and row["isa"] not in self.filter_isas:
                    continue

                ir_tokens = row["ir_tokens"]
                if isinstance(ir_tokens, str):
                    ir_tokens = json.loads(ir_tokens)

                cfg_edges = row.get("cfg_edges")
                if isinstance(cfg_edges, str):
                    cfg_edges = json.loads(cfg_edges) if cfg_edges else None

                sample = FunctionSample(
                    function_id=str(row["function_id"]),
                    source_id=str(row["source_id"]),
                    isa=str(row["isa"]),
                    ir_tokens=ir_tokens[: self.max_length],
                    cfg_edges=cfg_edges,
                    metadata=row.get("metadata"),
                )
                self.samples.append(sample)

    def _load_json_data(self, json_files: list[Path]) -> None:
        """Load samples from JSON files."""
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            # Handle both single sample and list formats
            samples_data = data if isinstance(data, list) else [data]

            for item in samples_data:
                if self.filter_isas and item["isa"] not in self.filter_isas:
                    continue

                sample = FunctionSample(
                    function_id=str(item["function_id"]),
                    source_id=str(item["source_id"]),
                    isa=str(item["isa"]),
                    ir_tokens=item["ir_tokens"][: self.max_length],
                    cfg_edges=item.get("cfg_edges"),
                    metadata=item.get("metadata"),
                )
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> FunctionSample:
        return self.samples[idx]

    def get_source_ids(self) -> set[str]:
        """Get all unique source function IDs."""
        return {s.source_id for s in self.samples}

    def get_isas(self) -> set[str]:
        """Get all ISAs present in the dataset."""
        return {s.isa for s in self.samples}


class CrossISADataset(Dataset):
    """Dataset for cross-ISA contrastive learning.

    Groups samples by source function, ensuring each batch contains
    multiple ISA variants of the same functions for positive pairing.

    Args:
        data_path: Path to data directory or Parquet file
        max_length: Maximum sequence length
        isas_per_sample: Number of ISA variants to sample per source function
        filter_isas: Only include samples from these ISAs
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        max_length: int = 512,
        isas_per_sample: int = 4,
        filter_isas: Optional[list[str]] = None,
    ):
        self.base_dataset = BinaryFunctionDataset(
            data_path=data_path,
            max_length=max_length,
            filter_isas=filter_isas,
        )
        self.isas_per_sample = isas_per_sample

        # Build index: source_id -> list of sample indices
        self.source_to_samples: dict[str, list[int]] = {}
        for idx, sample in enumerate(self.base_dataset.samples):
            if sample.source_id not in self.source_to_samples:
                self.source_to_samples[sample.source_id] = []
            self.source_to_samples[sample.source_id].append(idx)

        # Filter to sources with multiple ISAs
        self.source_ids = [
            sid
            for sid, indices in self.source_to_samples.items()
            if len(indices) >= 2  # Need at least 2 ISA variants
        ]

    def __len__(self) -> int:
        return len(self.source_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get samples for a source function across ISAs.

        Returns:
            Dictionary with:
                - source_id: str
                - samples: list of FunctionSample for different ISAs
                - source_label: int (index for contrastive loss)
        """
        source_id = self.source_ids[idx]
        sample_indices = self.source_to_samples[source_id]

        # Sample ISA variants (or take all if fewer than requested)
        if len(sample_indices) > self.isas_per_sample:
            selected_indices = random.sample(sample_indices, self.isas_per_sample)
        else:
            selected_indices = sample_indices

        samples = [self.base_dataset[i] for i in selected_indices]

        return {
            "source_id": source_id,
            "samples": samples,
            "source_label": idx,  # Used as label for contrastive loss
        }

    @classmethod
    def from_directory(
        cls,
        data_dir: Union[str, Path],
        **kwargs,
    ) -> "CrossISADataset":
        """Create dataset from a data directory."""
        return cls(data_path=data_dir, **kwargs)


class PretrainingDataset(Dataset):
    """Dataset for pretraining with MIM/CWP/DUP tasks.

    Prepares samples for:
        - Masked Instruction Modeling (MIM): Random token masking
        - Context Window Prediction (CWP): Same-block instruction pairs
        - Def-Use Prediction (DUP): Data dependency pairs

    Args:
        data_path: Path to data directory
        max_length: Maximum sequence length
        mlm_probability: Probability of masking each token
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        max_length: int = 512,
        mlm_probability: float = 0.15,
    ):
        self.base_dataset = BinaryFunctionDataset(
            data_path=data_path,
            max_length=max_length,
        )
        self.mlm_probability = mlm_probability

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample with masking applied.

        Returns:
            Dictionary with:
                - input_ids: Masked token sequence
                - labels: Original tokens (for loss computation)
                - attention_mask: Attention mask
        """
        sample = self.base_dataset[idx]
        input_ids = sample.ir_tokens.copy()
        labels = sample.ir_tokens.copy()

        # Apply masking (details handled by collator with tokenizer info)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "function_id": sample.function_id,
            "isa": sample.isa,
        }


class StreamingCrossISADataset(IterableDataset):
    """Streaming dataset for large-scale cross-ISA training.

    Loads data on-the-fly from sharded Parquet files, suitable for
    datasets too large to fit in memory.

    Args:
        data_path: Path to sharded data directory
        max_length: Maximum sequence length
        shuffle: Whether to shuffle samples
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        max_length: int = 512,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.shuffle = shuffle
        self.seed = seed

        # Find all shard files
        self.shard_files = sorted(self.data_path.glob("*.parquet"))
        if not self.shard_files:
            raise ValueError(f"No parquet shards found in {data_path}")

    def __iter__(self) -> Iterator[FunctionSample]:
        """Iterate over samples from shards."""
        worker_info = None
        try:
            from torch.utils.data import get_worker_info

            worker_info = get_worker_info()
        except ImportError:
            pass

        # Handle multi-worker data loading
        if worker_info is not None:
            # Split shards among workers
            per_worker = len(self.shard_files) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = (
                start + per_worker
                if worker_id < worker_info.num_workers - 1
                else len(self.shard_files)
            )
            shard_files = self.shard_files[start:end]
        else:
            shard_files = self.shard_files

        # Shuffle shards
        if self.shuffle:
            rng = random.Random(self.seed)
            shard_files = list(shard_files)
            rng.shuffle(shard_files)

        for shard_file in shard_files:
            table = pq.read_table(shard_file)
            df = table.to_pandas()

            indices = list(range(len(df)))
            if self.shuffle:
                random.shuffle(indices)

            for idx in indices:
                row = df.iloc[idx]

                ir_tokens = row["ir_tokens"]
                if isinstance(ir_tokens, str):
                    ir_tokens = json.loads(ir_tokens)

                yield FunctionSample(
                    function_id=str(row["function_id"]),
                    source_id=str(row["source_id"]),
                    isa=str(row["isa"]),
                    ir_tokens=ir_tokens[: self.max_length],
                    cfg_edges=row.get("cfg_edges"),
                )


class BenchmarkDataset(Dataset):
    """Dataset for evaluation benchmarks.

    Loads standard benchmark data (POJ-104, BinKit, Cisco, etc.) in
    a unified format for evaluation.

    Args:
        benchmark_name: Name of benchmark ("poj104", "binkit", "cisco", "trex")
        data_path: Path to benchmark data
        split: Data split ("train", "val", "test")
        pool_size: Size of retrieval pool for evaluation
    """

    SUPPORTED_BENCHMARKS = {"poj104", "binkit", "cisco", "trex", "binarycorp"}

    def __init__(
        self,
        benchmark_name: str,
        data_path: Union[str, Path],
        split: str = "test",
        pool_size: Optional[int] = None,
    ):
        if benchmark_name not in self.SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark: {benchmark_name}. Supported: {self.SUPPORTED_BENCHMARKS}"
            )

        self.benchmark_name = benchmark_name
        self.data_path = Path(data_path)
        self.split = split
        self.pool_size = pool_size

        self.samples: list[FunctionSample] = []
        self.queries: list[FunctionSample] = []
        self.ground_truth: dict[str, list[str]] = {}  # query_id -> relevant_ids

        self._load_benchmark()

    def _load_benchmark(self) -> None:
        """Load benchmark data in appropriate format."""
        benchmark_path = self.data_path / self.benchmark_name / self.split

        # Load samples
        samples_file = benchmark_path / "samples.parquet"
        if samples_file.exists():
            table = pq.read_table(samples_file)
            df = table.to_pandas()

            for _, row in df.iterrows():
                ir_tokens = row["ir_tokens"]
                if isinstance(ir_tokens, str):
                    ir_tokens = json.loads(ir_tokens)

                sample = FunctionSample(
                    function_id=str(row["function_id"]),
                    source_id=str(row["source_id"]),
                    isa=str(row["isa"]),
                    ir_tokens=ir_tokens,
                )
                self.samples.append(sample)

        # Load queries
        queries_file = benchmark_path / "queries.parquet"
        if queries_file.exists():
            table = pq.read_table(queries_file)
            df = table.to_pandas()

            for _, row in df.iterrows():
                ir_tokens = row["ir_tokens"]
                if isinstance(ir_tokens, str):
                    ir_tokens = json.loads(ir_tokens)

                query = FunctionSample(
                    function_id=str(row["function_id"]),
                    source_id=str(row["source_id"]),
                    isa=str(row["isa"]),
                    ir_tokens=ir_tokens,
                )
                self.queries.append(query)
        else:
            # Use a portion of samples as queries
            self.queries = self.samples[: len(self.samples) // 10]

        # Load ground truth
        gt_file = benchmark_path / "ground_truth.json"
        if gt_file.exists():
            with open(gt_file) as f:
                self.ground_truth = json.load(f)
        else:
            # Generate ground truth from source_ids
            source_to_ids: dict[str, list[str]] = {}
            for sample in self.samples:
                if sample.source_id not in source_to_ids:
                    source_to_ids[sample.source_id] = []
                source_to_ids[sample.source_id].append(sample.function_id)

            for query in self.queries:
                relevant = [
                    fid
                    for fid in source_to_ids.get(query.source_id, [])
                    if fid != query.function_id
                ]
                self.ground_truth[query.function_id] = relevant

        # Limit pool size if specified
        if self.pool_size and len(self.samples) > self.pool_size:
            # Keep query-relevant samples plus random samples
            relevant_ids = set()
            for relevant in self.ground_truth.values():
                relevant_ids.update(relevant)

            relevant_samples = [s for s in self.samples if s.function_id in relevant_ids]
            other_samples = [s for s in self.samples if s.function_id not in relevant_ids]

            remaining = self.pool_size - len(relevant_samples)
            if remaining > 0:
                sampled_others = random.sample(other_samples, min(remaining, len(other_samples)))
                self.samples = relevant_samples + sampled_others
            else:
                self.samples = relevant_samples[: self.pool_size]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> FunctionSample:
        return self.samples[idx]

    def get_queries(self) -> list[FunctionSample]:
        """Get query samples for evaluation."""
        return self.queries

    def get_ground_truth(self) -> dict[str, list[str]]:
        """Get ground truth mapping for evaluation."""
        return self.ground_truth
