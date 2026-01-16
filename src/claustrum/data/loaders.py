"""DataLoader factory functions for CLAUSTRUM training.

Provides convenient functions to create configured DataLoaders for
different training stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from torch.utils.data import DataLoader, DistributedSampler

from claustrum.data.dataset import (
    BinaryFunctionDataset,
    CrossISADataset,
    PretrainingDataset,
    StreamingCrossISADataset,
    BenchmarkDataset,
)
from claustrum.data.collators import (
    BaseCollator,
    ContrastiveCollator,
    PretrainingCollator,
    CFGCollator,
)
from claustrum.tokenization import IRTokenizer


def create_train_dataloader(
    data_path: Union[str, Path],
    tokenizer: IRTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    isas_per_sample: int = 4,
    num_workers: int = 4,
    filter_isas: Optional[list[str]] = None,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """Create DataLoader for contrastive training.

    Args:
        data_path: Path to training data
        tokenizer: IR tokenizer
        batch_size: Samples per batch (source functions, not total ISA variants)
        max_length: Maximum sequence length
        isas_per_sample: Number of ISA variants per source function
        num_workers: DataLoader workers
        filter_isas: Only include these ISAs
        distributed: Use distributed sampler
        world_size: Number of distributed processes
        rank: Current process rank

    Returns:
        Configured DataLoader
    """
    dataset = CrossISADataset(
        data_path=data_path,
        max_length=max_length,
        isas_per_sample=isas_per_sample,
        filter_isas=filter_isas,
    )

    collator = ContrastiveCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    sampler = None
    shuffle = True

    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )


def create_eval_dataloader(
    data_path: Union[str, Path],
    tokenizer: IRTokenizer,
    batch_size: int = 64,
    max_length: int = 512,
    num_workers: int = 4,
    filter_isas: Optional[list[str]] = None,
) -> DataLoader:
    """Create DataLoader for evaluation.

    Args:
        data_path: Path to evaluation data
        tokenizer: IR tokenizer
        batch_size: Samples per batch
        max_length: Maximum sequence length
        num_workers: DataLoader workers
        filter_isas: Only include these ISAs

    Returns:
        Configured DataLoader
    """
    dataset = BinaryFunctionDataset(
        data_path=data_path,
        max_length=max_length,
        filter_isas=filter_isas,
    )

    collator = BaseCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


def create_pretraining_dataloader(
    data_path: Union[str, Path],
    tokenizer: IRTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    mlm_probability: float = 0.15,
    num_workers: int = 4,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """Create DataLoader for pretraining.

    Args:
        data_path: Path to pretraining data
        tokenizer: IR tokenizer
        batch_size: Samples per batch
        max_length: Maximum sequence length
        mlm_probability: Token masking probability
        num_workers: DataLoader workers
        distributed: Use distributed sampler
        world_size: Number of distributed processes
        rank: Current process rank

    Returns:
        Configured DataLoader
    """
    dataset = PretrainingDataset(
        data_path=data_path,
        max_length=max_length,
        mlm_probability=mlm_probability,
    )

    collator = PretrainingCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        mlm_probability=mlm_probability,
    )

    sampler = None
    shuffle = True

    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )


def create_streaming_dataloader(
    data_path: Union[str, Path],
    tokenizer: IRTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """Create streaming DataLoader for large datasets.

    Args:
        data_path: Path to sharded data
        tokenizer: IR tokenizer
        batch_size: Samples per batch
        max_length: Maximum sequence length
        num_workers: DataLoader workers
        seed: Random seed for shuffling

    Returns:
        Streaming DataLoader
    """
    dataset = StreamingCrossISADataset(
        data_path=data_path,
        max_length=max_length,
        shuffle=True,
        seed=seed,
    )

    collator = BaseCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


def create_benchmark_dataloader(
    benchmark_name: str,
    data_path: Union[str, Path],
    tokenizer: IRTokenizer,
    split: str = "test",
    batch_size: int = 64,
    max_length: int = 512,
    pool_size: Optional[int] = None,
    num_workers: int = 4,
) -> tuple[DataLoader, BenchmarkDataset]:
    """Create DataLoader for benchmark evaluation.

    Args:
        benchmark_name: Name of benchmark (poj104, binkit, cisco, trex)
        data_path: Path to benchmark data
        tokenizer: IR tokenizer
        split: Data split (train, val, test)
        batch_size: Samples per batch
        max_length: Maximum sequence length
        pool_size: Size of retrieval pool
        num_workers: DataLoader workers

    Returns:
        Tuple of (DataLoader, BenchmarkDataset)
    """
    dataset = BenchmarkDataset(
        benchmark_name=benchmark_name,
        data_path=data_path,
        split=split,
        pool_size=pool_size,
    )

    collator = BaseCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    return dataloader, dataset


def create_cfg_dataloader(
    data_path: Union[str, Path],
    tokenizer: IRTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4,
    filter_isas: Optional[list[str]] = None,
) -> DataLoader:
    """Create DataLoader with CFG information for GNN.

    Args:
        data_path: Path to data with CFG edges
        tokenizer: IR tokenizer
        batch_size: Samples per batch
        max_length: Maximum sequence length
        num_workers: DataLoader workers
        filter_isas: Only include these ISAs

    Returns:
        DataLoader with CFG edge batching
    """
    dataset = BinaryFunctionDataset(
        data_path=data_path,
        max_length=max_length,
        filter_isas=filter_isas,
    )

    collator = CFGCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
