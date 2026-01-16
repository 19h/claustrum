#!/usr/bin/env python3
"""Data preparation script for CLAUSTRUM.

Converts raw binaries or cross-compilation data into training-ready format.

Usage:
    # From binary directory organized by ISA
    python scripts/prepare_data.py from-binaries --input data/binaries --output data/processed
    
    # From cross-compilation metadata
    python scripts/prepare_data.py from-metadata --metadata data/compilation.json --binaries data/bins
    
    # Create train/val split
    python scripts/prepare_data.py split --input data/processed --output data/split --val-ratio 0.1
"""

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(help="CLAUSTRUM Data Preparation")
console = Console()


@app.command("from-binaries")
def prepare_from_binaries(
    input_dir: str = typer.Option(..., "--input", "-i", help="Input directory with binaries"),
    output_dir: str = typer.Option("data/processed", "--output", "-o", help="Output directory"),
    max_length: int = typer.Option(512, "--max-length", help="Max sequence length"),
    num_workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
    shard_size: int = typer.Option(10000, "--shard-size", help="Samples per shard"),
):
    """Process binaries organized by ISA subdirectories.
    
    Expected structure:
        input_dir/
            x86_64/
                binary1, binary2, ...
            arm64/
                binary1, binary2, ...
    """
    console.print("[bold blue]CLAUSTRUM Data Preparation[/bold blue]")
    console.print(f"Input: {input_dir}")
    console.print(f"Output: {output_dir}")
    
    from claustrum.data.preprocessing import BinaryProcessor, DatasetBuilder
    from claustrum.tokenization import IRTokenizer
    
    # Create processor and builder
    tokenizer = IRTokenizer()
    processor = BinaryProcessor(tokenizer=tokenizer, max_length=max_length)
    builder = DatasetBuilder(
        processor=processor,
        output_dir=output_dir,
        shard_size=shard_size,
        num_workers=num_workers,
    )
    
    # Process binaries
    console.print("\n[bold]Processing binaries...[/bold]")
    stats = builder.build_from_directory(Path(input_dir))
    
    # Print stats
    console.print("\n[bold green]Processing complete![/bold green]")
    console.print(f"Total functions: {stats.get('total_functions', 0):,}")
    console.print(f"Total errors: {stats.get('total_errors', 0):,}")
    console.print(f"Shards created: {stats.get('num_shards', 0)}")
    console.print(f"ISAs: {', '.join(stats.get('isas', []))}")
    console.print(f"Unique source functions: {stats.get('source_functions', 0):,}")
    console.print(f"Output saved to: {output_dir}")


@app.command("from-metadata")
def prepare_from_metadata(
    metadata: str = typer.Option(..., "--metadata", "-m", help="Cross-compilation metadata JSON"),
    binaries_dir: str = typer.Option(..., "--binaries", "-b", help="Directory with compiled binaries"),
    output_dir: str = typer.Option("data/processed", "--output", "-o", help="Output directory"),
    max_length: int = typer.Option(512, "--max-length", help="Max sequence length"),
    num_workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
):
    """Process cross-compilation data from metadata file.
    
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
                    ]
                },
            ]
        }
    """
    console.print("[bold blue]CLAUSTRUM Cross-Compilation Data Preparation[/bold blue]")
    console.print(f"Metadata: {metadata}")
    console.print(f"Binaries: {binaries_dir}")
    console.print(f"Output: {output_dir}")
    
    from claustrum.data.preprocessing import BinaryProcessor, DatasetBuilder
    from claustrum.tokenization import IRTokenizer
    
    # Create processor and builder
    tokenizer = IRTokenizer()
    processor = BinaryProcessor(tokenizer=tokenizer, max_length=max_length)
    builder = DatasetBuilder(
        processor=processor,
        output_dir=output_dir,
        num_workers=num_workers,
    )
    
    # Process from metadata
    console.print("\n[bold]Processing cross-compilation data...[/bold]")
    stats = builder.build_from_cross_compilation(
        metadata_file=Path(metadata),
        binaries_dir=Path(binaries_dir),
    )
    
    # Print stats
    console.print("\n[bold green]Processing complete![/bold green]")
    console.print(f"Total functions: {stats.get('total_functions', 0):,}")
    console.print(f"Total errors: {stats.get('total_errors', 0):,}")
    console.print(f"ISAs: {', '.join(stats.get('isas', []))}")
    console.print(f"Source functions: {stats.get('source_functions', 0):,}")
    console.print(f"Output saved to: {output_dir}")


@app.command("split")
def create_split(
    input_dir: str = typer.Option(..., "--input", "-i", help="Input processed data directory"),
    output_dir: str = typer.Option("data/split", "--output", "-o", help="Output directory"),
    val_ratio: float = typer.Option(0.1, "--val-ratio", help="Validation set ratio"),
    test_ratio: float = typer.Option(0.1, "--test-ratio", help="Test set ratio"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    """Split processed data into train/val/test sets.
    
    Splits by source_id to ensure no cross-ISA pairs are split across sets.
    """
    console.print("[bold blue]CLAUSTRUM Data Split[/bold blue]")
    console.print(f"Input: {input_dir}")
    console.print(f"Output: {output_dir}")
    console.print(f"Val ratio: {val_ratio}, Test ratio: {test_ratio}")
    
    import json
    import random
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    random.seed(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Load all samples
    console.print("\n[bold]Loading data...[/bold]")
    samples = []
    for pq_file in input_path.glob("*.parquet"):
        table = pq.read_table(pq_file)
        for i in range(len(table)):
            row = {col: table[col][i].as_py() for col in table.column_names}
            samples.append(row)
            
    console.print(f"Loaded {len(samples):,} samples")
    
    # Group by source_id
    source_to_samples = {}
    for idx, sample in enumerate(samples):
        source_id = sample["source_id"]
        if source_id not in source_to_samples:
            source_to_samples[source_id] = []
        source_to_samples[source_id].append(idx)
        
    source_ids = list(source_to_samples.keys())
    random.shuffle(source_ids)
    
    console.print(f"Unique source functions: {len(source_ids):,}")
    
    # Split source IDs
    n_total = len(source_ids)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val
    
    test_sources = set(source_ids[:n_test])
    val_sources = set(source_ids[n_test:n_test + n_val])
    train_sources = set(source_ids[n_test + n_val:])
    
    console.print(f"\nSplit sizes:")
    console.print(f"  Train: {n_train} sources")
    console.print(f"  Val: {n_val} sources")
    console.print(f"  Test: {n_test} sources")
    
    # Create splits
    for split_name, split_sources in [
        ("train", train_sources),
        ("val", val_sources),
        ("test", test_sources),
    ]:
        if not split_sources:
            continue
            
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Gather samples for this split
        split_indices = []
        for source_id in split_sources:
            split_indices.extend(source_to_samples[source_id])
            
        split_samples = [samples[i] for i in split_indices]
        
        # Write to parquet
        if split_samples:
            schema = pa.schema([
                ("function_id", pa.string()),
                ("source_id", pa.string()),
                ("isa", pa.string()),
                ("ir_tokens", pa.list_(pa.int32())),
                ("cfg_edges", pa.string()),
                ("metadata", pa.string()),
            ])
            
            arrays = [
                pa.array([s.get("function_id", "") for s in split_samples]),
                pa.array([s.get("source_id", "") for s in split_samples]),
                pa.array([s.get("isa", "") for s in split_samples]),
                pa.array([s.get("ir_tokens", []) for s in split_samples]),
                pa.array([s.get("cfg_edges", "[]") for s in split_samples]),
                pa.array([s.get("metadata", "{}") for s in split_samples]),
            ]
            
            table = pa.Table.from_arrays(arrays, schema=schema)
            pq.write_table(table, split_dir / "data.parquet")
            
            console.print(f"  {split_name}: {len(split_samples):,} samples -> {split_dir}")
    
    console.print("\n[bold green]Split complete![/bold green]")


@app.command("stats")
def show_stats(
    data_path: str = typer.Option(..., "--data", "-d", help="Data directory or parquet file"),
):
    """Show statistics about a processed dataset."""
    console.print("[bold blue]CLAUSTRUM Dataset Statistics[/bold blue]")
    
    import json
    from collections import Counter
    import pyarrow.parquet as pq
    
    data_path = Path(data_path)
    
    # Load samples
    samples = []
    if data_path.is_file():
        parquet_files = [data_path]
    else:
        parquet_files = list(data_path.glob("**/*.parquet"))
        
    for pq_file in parquet_files:
        table = pq.read_table(pq_file)
        for i in range(len(table)):
            row = {col: table[col][i].as_py() for col in table.column_names}
            samples.append(row)
            
    if not samples:
        console.print("[red]No samples found![/red]")
        return
        
    # Compute stats
    total_samples = len(samples)
    unique_sources = len({s["source_id"] for s in samples})
    isas = Counter(s["isa"] for s in samples)
    
    token_lengths = [len(s.get("ir_tokens", [])) for s in samples]
    avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    max_length = max(token_lengths) if token_lengths else 0
    min_length = min(token_lengths) if token_lengths else 0
    
    # Source ISA coverage
    source_isas = {}
    for s in samples:
        source_id = s["source_id"]
        if source_id not in source_isas:
            source_isas[source_id] = set()
        source_isas[source_id].add(s["isa"])
        
    isas_per_source = [len(isas) for isas in source_isas.values()]
    avg_isas_per_source = sum(isas_per_source) / len(isas_per_source) if isas_per_source else 0
    
    # Print stats
    console.print(f"\n[bold]Dataset Overview[/bold]")
    console.print(f"Total samples: {total_samples:,}")
    console.print(f"Unique source functions: {unique_sources:,}")
    console.print(f"Avg ISAs per source: {avg_isas_per_source:.1f}")
    
    console.print(f"\n[bold]Token Length Statistics[/bold]")
    console.print(f"Average length: {avg_length:.1f}")
    console.print(f"Min length: {min_length}")
    console.print(f"Max length: {max_length}")
    
    console.print(f"\n[bold]ISA Distribution[/bold]")
    for isa, count in isas.most_common():
        pct = count / total_samples * 100
        console.print(f"  {isa}: {count:,} ({pct:.1f}%)")
        
    # Cross-ISA pairs available
    sources_with_multiple_isas = sum(1 for isas in source_isas.values() if len(isas) >= 2)
    console.print(f"\n[bold]Cross-ISA Training[/bold]")
    console.print(f"Sources with 2+ ISAs: {sources_with_multiple_isas:,} ({sources_with_multiple_isas/unique_sources*100:.1f}%)")


@app.command("download-benchmark")
def download_benchmark(
    benchmark: str = typer.Argument(..., help="Benchmark name: poj104, binkit, cisco, trex"),
    output_dir: str = typer.Option("data/benchmarks", "--output", "-o", help="Output directory"),
):
    """Download and prepare a standard benchmark dataset."""
    console.print(f"[bold blue]Downloading {benchmark} benchmark[/bold blue]")
    
    benchmark_urls = {
        "poj104": "https://github.com/poj104-dataset/releases/...",  # placeholder
        "binkit": "https://github.com/SoftSec-KAIST/BinKit/...",
        "cisco": "https://github.com/cisco-research/...",
        "trex": "https://github.com/CUMLSec/trex/...",
    }
    
    if benchmark not in benchmark_urls:
        console.print(f"[red]Unknown benchmark: {benchmark}[/red]")
        console.print(f"Available: {', '.join(benchmark_urls.keys())}")
        return
        
    console.print(f"\n[yellow]Note: Benchmark downloads require manual setup.[/yellow]")
    console.print(f"Please download {benchmark} from:")
    console.print(f"  {benchmark_urls[benchmark]}")
    console.print(f"\nThen process with:")
    console.print(f"  python scripts/prepare_data.py from-binaries --input <downloaded> --output {output_dir}/{benchmark}")


if __name__ == "__main__":
    app()
