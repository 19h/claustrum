"""Batch collation functions for CLAUSTRUM data loading.

Collators transform lists of samples into batched tensors suitable
for model training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union, cast

import torch
import numpy as np

from claustrum.data.dataset import FunctionSample
from claustrum.tokenization import IRTokenizer


@dataclass
class BaseCollator:
    """Base collator for batching function samples.

    Pads sequences to uniform length and creates attention masks.

    Args:
        tokenizer: Tokenizer for special token IDs
        max_length: Maximum sequence length
        pad_to_multiple_of: Pad to multiple of this value (for efficiency)
    """

    tokenizer: IRTokenizer
    max_length: int = 512
    pad_to_multiple_of: Optional[int] = 8

    def __call__(
        self, features: list[Union[dict[str, Any], FunctionSample]]
    ) -> dict[str, torch.Tensor]:
        """Collate a batch of samples.

        Args:
            features: List of sample dictionaries or FunctionSample objects

        Returns:
            Batched tensors with padding
        """
        # Handle FunctionSample objects or dicts
        if features and isinstance(features[0], FunctionSample):
            samples = cast(list[FunctionSample], features)
            input_ids_list = [f.ir_tokens for f in samples]
        else:
            dicts = cast(list[dict[str, Any]], features)
            input_ids_list = [f["input_ids"] for f in dicts]

        # Find max length in batch
        batch_max_len = min(max(len(ids) for ids in input_ids_list), self.max_length)

        # Pad to multiple
        if self.pad_to_multiple_of:
            batch_max_len = (
                (batch_max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Pad sequences
        padded_input_ids = []
        attention_masks = []

        pad_id = self.tokenizer.pad_token_id

        for ids in input_ids_list:
            # Truncate if needed
            ids = ids[: self.max_length]

            # Compute padding
            padding_length = batch_max_len - len(ids)

            padded_ids = ids + [pad_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length

            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)

        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }

        return batch


@dataclass
class ContrastiveCollator:
    """Collator for contrastive learning with cross-ISA pairs.

    Handles batches where each item contains multiple ISA variants
    of the same source function.

    Args:
        tokenizer: Tokenizer for special token IDs
        max_length: Maximum sequence length
        flatten: Whether to flatten all samples into single batch
    """

    tokenizer: IRTokenizer
    max_length: int = 512
    flatten: bool = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate cross-ISA batch.

        Args:
            features: List of {"source_id", "samples", "source_label"} dicts

        Returns:
            Batched tensors with source labels for contrastive loss
        """
        all_samples = []
        all_labels = []

        for item in features:
            samples = item["samples"]
            source_label = item["source_label"]

            for sample in samples:
                all_samples.append(sample)
                all_labels.append(source_label)

        # Pad sequences
        input_ids_list = [s.ir_tokens for s in all_samples]

        batch_max_len = min(max(len(ids) for ids in input_ids_list), self.max_length)

        # Round up to multiple of 8 for efficiency
        batch_max_len = ((batch_max_len + 7) // 8) * 8

        padded_input_ids = []
        attention_masks = []

        pad_id = self.tokenizer.pad_token_id

        for ids in input_ids_list:
            ids = ids[: self.max_length]
            padding_length = batch_max_len - len(ids)

            padded_ids = ids + [pad_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length

            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)

        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "source_labels": torch.tensor(all_labels, dtype=torch.long),
        }

        # Add ISA information for analysis
        batch["isas"] = [s.isa for s in all_samples]  # type: ignore[assignment]

        return batch


@dataclass
class PretrainingCollator:
    """Collator for pretraining with MIM/CWP/DUP tasks.

    Applies masking and generates labels for pretraining objectives.

    Args:
        tokenizer: Tokenizer for masking
        max_length: Maximum sequence length
        mlm_probability: Probability of masking each token
    """

    tokenizer: IRTokenizer
    max_length: int = 512
    mlm_probability: float = 0.15

    def __call__(
        self, features: list[Union[dict[str, Any], FunctionSample]]
    ) -> dict[str, torch.Tensor]:
        """Collate and apply masking for pretraining.

        Args:
            features: List of sample dictionaries or FunctionSample objects

        Returns:
            Batched tensors with masked inputs and labels
        """
        # Get token sequences
        if isinstance(features[0], FunctionSample):
            samples = cast(list[FunctionSample], features)
            input_ids_list = [f.ir_tokens for f in samples]
        else:
            dicts = cast(list[dict[str, Any]], features)
            input_ids_list = [f["input_ids"] for f in dicts]

        batch_max_len = min(max(len(ids) for ids in input_ids_list), self.max_length)
        batch_max_len = ((batch_max_len + 7) // 8) * 8

        # Pad and mask
        padded_input_ids = []
        labels = []
        attention_masks = []

        pad_id = self.tokenizer.pad_token_id
        mask_id = self.tokenizer.mask_token_id

        for ids in input_ids_list:
            ids = list(ids[: self.max_length])  # Make a copy
            padding_length = batch_max_len - len(ids)

            # Create labels (original tokens)
            token_labels = ids.copy() + [-100] * padding_length  # -100 = ignore in loss

            # Apply masking
            masked_ids = self._apply_masking(ids, mask_id)

            # Pad
            masked_ids = masked_ids + [pad_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length

            padded_input_ids.append(masked_ids)
            labels.append(token_labels)
            attention_masks.append(mask)

        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        return batch

    def _apply_masking(self, token_ids: list[int], mask_id: int) -> list[int]:
        """Apply BERT-style masking to token sequence.

        80% mask token, 10% random token, 10% unchanged

        Args:
            token_ids: Original token IDs
            mask_id: Mask token ID

        Returns:
            Masked token sequence
        """
        masked = token_ids.copy()
        vocab_size = self.tokenizer.vocab_size

        # Special token IDs to not mask
        special_ids = {
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
        }

        for i in range(len(masked)):
            if masked[i] in special_ids:
                continue

            if np.random.random() < self.mlm_probability:
                prob = np.random.random()

                if prob < 0.8:
                    # Replace with [MASK]
                    masked[i] = mask_id
                elif prob < 0.9:
                    # Replace with random token
                    masked[i] = np.random.randint(0, vocab_size)
                # else: keep original (10%)

        return masked


@dataclass
class CFGCollator:
    """Collator that includes CFG edge information for GNN.

    Prepares batched graph data for the Graph Attention Network.

    Args:
        tokenizer: Tokenizer for special token IDs
        max_length: Maximum sequence length
        max_blocks: Maximum number of basic blocks per function
    """

    tokenizer: IRTokenizer
    max_length: int = 512
    max_blocks: int = 128

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate batch with CFG information.

        Args:
            features: List of samples with cfg_edges

        Returns:
            Batched tensors including edge indices for GNN
        """
        # Handle FunctionSample or dict
        if isinstance(features[0], FunctionSample):
            samples = features
        else:
            samples = [f.get("sample") or f for f in features]

        # Collate token sequences
        input_ids_list = [
            s.ir_tokens if isinstance(s, FunctionSample) else s["input_ids"] for s in samples
        ]

        batch_max_len = min(max(len(ids) for ids in input_ids_list), self.max_length)
        batch_max_len = ((batch_max_len + 7) // 8) * 8

        pad_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        attention_masks = []

        for ids in input_ids_list:
            ids = ids[: self.max_length]
            padding_length = batch_max_len - len(ids)

            padded_input_ids.append(ids + [pad_id] * padding_length)
            attention_masks.append([1] * len(ids) + [0] * padding_length)

        # Collate CFG edges
        # Convert to batched edge index format for PyG
        edge_indices = []
        batch_indices = []

        node_offset = 0
        for batch_idx, sample in enumerate(samples):
            cfg_edges = (
                sample.cfg_edges
                if isinstance(sample, FunctionSample)
                else sample.get("cfg_edges", [])
            )

            if cfg_edges:
                # Offset node indices for batch
                edges = torch.tensor(cfg_edges, dtype=torch.long).T
                edges = edges + node_offset
                edge_indices.append(edges)

                # Track which graph each node belongs to
                num_nodes = max(max(e) for e in cfg_edges) + 1 if cfg_edges else 1
                batch_indices.extend([batch_idx] * num_nodes)
                node_offset += num_nodes

        # Stack edges
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
            batch_idx = torch.tensor(batch_indices, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            batch_idx = torch.zeros(0, dtype=torch.long)

        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "edge_index": edge_index,
            "batch": batch_idx,
        }

        return batch
