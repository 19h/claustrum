"""Integration tests for the data loading pipeline."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from claustrum.data import (
    BinaryFunctionDataset,
    CrossISADataset,
    PretrainingDataset,
    FunctionSample,
    ContrastiveCollator,
    PretrainingCollator,
    create_train_dataloader,
    create_eval_dataloader,
)
from claustrum.tokenization import IRTokenizer


class TestFunctionSample:
    """Test FunctionSample dataclass."""

    def test_sample_creation(self):
        """Test creating a function sample."""
        sample = FunctionSample(
            function_id="func_001",
            source_id="source_001",
            isa="x86_64",
            ir_tokens=[1, 2, 3, 4, 5],
            cfg_edges=[(0, 1), (1, 2)],
        )

        assert sample.function_id == "func_001"
        assert sample.isa == "x86_64"
        assert len(sample) == 5

    def test_sample_without_cfg(self):
        """Test sample without CFG edges."""
        sample = FunctionSample(
            function_id="func_002",
            source_id="source_002",
            isa="arm64",
            ir_tokens=[10, 20, 30],
        )

        assert sample.cfg_edges is None


class TestBinaryFunctionDataset:
    """Test BinaryFunctionDataset loading."""

    @pytest.fixture
    def sample_parquet_dir(self, tmp_path):
        """Create sample Parquet data."""
        data = {
            "function_id": ["f1", "f2", "f3", "f4"],
            "source_id": ["s1", "s1", "s2", "s2"],
            "isa": ["x86_64", "arm64", "x86_64", "arm64"],
            "ir_tokens": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            "cfg_edges": ["[[0,1]]", "[[0,1]]", "[]", "[]"],
        }

        schema = pa.schema(
            [
                ("function_id", pa.string()),
                ("source_id", pa.string()),
                ("isa", pa.string()),
                ("ir_tokens", pa.list_(pa.int32())),
                ("cfg_edges", pa.string()),
            ]
        )

        arrays = [
            pa.array(data["function_id"]),
            pa.array(data["source_id"]),
            pa.array(data["isa"]),
            pa.array(data["ir_tokens"]),
            pa.array(data["cfg_edges"]),
        ]

        table = pa.Table.from_arrays(arrays, schema=schema)
        pq.write_table(table, tmp_path / "data.parquet")

        return tmp_path

    @pytest.fixture
    def sample_json_dir(self, tmp_path):
        """Create sample JSON data."""
        samples = [
            {"function_id": "f1", "source_id": "s1", "isa": "x86_64", "ir_tokens": [1, 2, 3]},
            {"function_id": "f2", "source_id": "s1", "isa": "arm64", "ir_tokens": [4, 5, 6]},
        ]

        with open(tmp_path / "data.json", "w") as f:
            json.dump(samples, f)

        return tmp_path

    def test_load_parquet(self, sample_parquet_dir):
        """Test loading from Parquet files."""
        dataset = BinaryFunctionDataset(sample_parquet_dir)

        assert len(dataset) == 4
        assert dataset[0].function_id == "f1"
        assert dataset[0].ir_tokens == [1, 2, 3]

    def test_load_json(self, sample_json_dir):
        """Test loading from JSON files."""
        dataset = BinaryFunctionDataset(sample_json_dir)

        assert len(dataset) == 2

    def test_filter_by_isa(self, sample_parquet_dir):
        """Test filtering by ISA."""
        dataset = BinaryFunctionDataset(
            sample_parquet_dir,
            filter_isas=["x86_64"],
        )

        assert len(dataset) == 2
        for sample in dataset.samples:
            assert sample.isa == "x86_64"

    def test_get_source_ids(self, sample_parquet_dir):
        """Test getting unique source IDs."""
        dataset = BinaryFunctionDataset(sample_parquet_dir)
        source_ids = dataset.get_source_ids()

        assert source_ids == {"s1", "s2"}

    def test_get_isas(self, sample_parquet_dir):
        """Test getting unique ISAs."""
        dataset = BinaryFunctionDataset(sample_parquet_dir)
        isas = dataset.get_isas()

        assert isas == {"x86_64", "arm64"}


class TestCrossISADataset:
    """Test CrossISADataset for contrastive learning."""

    @pytest.fixture
    def multi_isa_data(self, tmp_path):
        """Create multi-ISA paired data."""
        data = {
            "function_id": [f"f{i}" for i in range(8)],
            "source_id": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
            "isa": ["x86_64", "arm64", "mips32", "riscv64"] * 2,
            "ir_tokens": [[i] * 10 for i in range(8)],
            "cfg_edges": ["[]"] * 8,
        }

        schema = pa.schema(
            [
                ("function_id", pa.string()),
                ("source_id", pa.string()),
                ("isa", pa.string()),
                ("ir_tokens", pa.list_(pa.int32())),
                ("cfg_edges", pa.string()),
            ]
        )

        table = pa.Table.from_arrays(
            [pa.array(v) if not isinstance(v[0], list) else pa.array(v) for v in data.values()],
            schema=schema,
        )
        pq.write_table(table, tmp_path / "data.parquet")

        return tmp_path

    def test_cross_isa_grouping(self, multi_isa_data):
        """Test that samples are grouped by source."""
        dataset = CrossISADataset(multi_isa_data)

        # Should have 2 source functions
        assert len(dataset) == 2

    def test_get_item_returns_multiple_isas(self, multi_isa_data):
        """Test that __getitem__ returns samples from multiple ISAs."""
        dataset = CrossISADataset(multi_isa_data, isas_per_sample=4)

        item = dataset[0]

        assert "source_id" in item
        assert "samples" in item
        assert len(item["samples"]) <= 4

        # All samples should have same source_id
        isas = {s.isa for s in item["samples"]}
        assert len(isas) > 1  # Multiple ISAs


class TestCollators:
    """Test batch collation functions."""

    @pytest.fixture
    def tokenizer(self):
        """Create a mock tokenizer."""

        class MockTokenizer:
            pad_token_id = 0
            mask_token_id = 1
            cls_token_id = 2
            sep_token_id = 3
            vocab_size = 1000

        return MockTokenizer()

    def test_contrastive_collator(self, tokenizer):
        """Test ContrastiveCollator batching."""
        collator = ContrastiveCollator(tokenizer=tokenizer, max_length=32)

        features = [
            {
                "source_id": "s1",
                "samples": [
                    FunctionSample("f1", "s1", "x86_64", [1, 2, 3, 4, 5]),
                    FunctionSample("f2", "s1", "arm64", [6, 7, 8, 9, 10]),
                ],
                "source_label": 0,
            },
            {
                "source_id": "s2",
                "samples": [
                    FunctionSample("f3", "s2", "x86_64", [11, 12, 13]),
                    FunctionSample("f4", "s2", "arm64", [14, 15, 16]),
                ],
                "source_label": 1,
            },
        ]

        batch = collator(features)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "source_labels" in batch

        # 4 total samples (2 ISAs x 2 sources)
        assert batch["input_ids"].shape[0] == 4

        # Labels should identify source
        labels = batch["source_labels"].tolist()
        assert labels.count(0) == 2
        assert labels.count(1) == 2

    def test_pretraining_collator(self, tokenizer):
        """Test PretrainingCollator with masking."""
        collator = PretrainingCollator(
            tokenizer=tokenizer,
            max_length=32,
            mlm_probability=0.15,
        )

        features = [
            {"input_ids": list(range(10, 30))},
            {"input_ids": list(range(30, 50))},
        ]

        batch = collator(features)

        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch

        # Some positions should be masked
        labels = batch["labels"]
        assert (labels != -100).any()  # Some labels should be set


class TestDataLoaders:
    """Test DataLoader factory functions."""

    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create sample data directory."""
        data = {
            "function_id": [f"f{i}" for i in range(20)],
            "source_id": [f"s{i // 4}" for i in range(20)],  # 5 sources, 4 ISAs each
            "isa": ["x86_64", "arm64", "mips32", "riscv64"] * 5,
            "ir_tokens": [[i * 10 + j for j in range(20)] for i in range(20)],
            "cfg_edges": ["[]"] * 20,
        }

        schema = pa.schema(
            [
                ("function_id", pa.string()),
                ("source_id", pa.string()),
                ("isa", pa.string()),
                ("ir_tokens", pa.list_(pa.int32())),
                ("cfg_edges", pa.string()),
            ]
        )

        table = pa.Table.from_arrays(
            [pa.array(v) for v in data.values()],
            schema=schema,
        )
        pq.write_table(table, tmp_path / "train.parquet")

        return tmp_path

    def test_create_train_dataloader(self, data_dir):
        """Test creating training DataLoader."""
        tokenizer = IRTokenizer()

        dataloader = create_train_dataloader(
            data_path=data_dir,
            tokenizer=tokenizer,
            batch_size=2,
            num_workers=0,  # For testing
        )

        # Get one batch
        batch = next(iter(dataloader))

        assert "input_ids" in batch
        assert batch["input_ids"].dim() == 2

    def test_create_eval_dataloader(self, data_dir):
        """Test creating evaluation DataLoader."""
        tokenizer = IRTokenizer()

        dataloader = create_eval_dataloader(
            data_path=data_dir,
            tokenizer=tokenizer,
            batch_size=4,
            num_workers=0,
        )

        batch = next(iter(dataloader))
        assert "input_ids" in batch


class TestPretrainingDataset:
    """Test PretrainingDataset."""

    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create sample data."""
        data = {
            "function_id": ["f1", "f2"],
            "source_id": ["s1", "s2"],
            "isa": ["x86_64", "arm64"],
            "ir_tokens": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            "cfg_edges": ["[]", "[]"],
        }

        schema = pa.schema(
            [
                ("function_id", pa.string()),
                ("source_id", pa.string()),
                ("isa", pa.string()),
                ("ir_tokens", pa.list_(pa.int32())),
                ("cfg_edges", pa.string()),
            ]
        )

        table = pa.Table.from_arrays(
            [pa.array(v) for v in data.values()],
            schema=schema,
        )
        pq.write_table(table, tmp_path / "data.parquet")

        return tmp_path

    def test_pretraining_dataset(self, data_dir):
        """Test PretrainingDataset returns correct format."""
        dataset = PretrainingDataset(data_dir, mlm_probability=0.15)

        assert len(dataset) == 2

        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item
