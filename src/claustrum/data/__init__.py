"""Data loading and processing pipeline for CLAUSTRUM.

This module provides dataset classes and data loaders for training and evaluation:

    - BinaryFunctionDataset: Base dataset for binary function samples
    - CrossISADataset: Dataset for cross-ISA contrastive learning
    - PretrainingDataset: Dataset for MIM/CWP/DUP pretraining tasks
    - Collators: Batch collation functions for different training modes

Supported data formats:
    - Raw binaries with symbol information
    - Pre-processed IR sequences (Parquet/Arrow)
    - Paired cross-compilation data (JSON metadata + binaries)

Example:
    >>> from claustrum.data import CrossISADataset, ContrastiveCollator
    >>> dataset = CrossISADataset.from_directory("data/processed/train")
    >>> collator = ContrastiveCollator(tokenizer)
    >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collator)
"""

from claustrum.data.dataset import (
    BinaryFunctionDataset,
    CrossISADataset,
    PretrainingDataset,
    FunctionSample,
)
from claustrum.data.collators import (
    BaseCollator,
    ContrastiveCollator,
    PretrainingCollator,
)
from claustrum.data.preprocessing import (
    BinaryProcessor,
    DatasetBuilder,
)
from claustrum.data.loaders import (
    create_train_dataloader,
    create_eval_dataloader,
    create_pretraining_dataloader,
)

__all__ = [
    # Datasets
    "BinaryFunctionDataset",
    "CrossISADataset",
    "PretrainingDataset",
    "FunctionSample",
    # Collators
    "BaseCollator",
    "ContrastiveCollator",
    "PretrainingCollator",
    # Preprocessing
    "BinaryProcessor",
    "DatasetBuilder",
    # Loaders
    "create_train_dataloader",
    "create_eval_dataloader",
    "create_pretraining_dataloader",
]
