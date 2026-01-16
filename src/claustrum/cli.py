"""Command-line interface for CLAUSTRUM.

Provides commands for:
- Embedding functions from binaries
- Training models
- Evaluating models
- Building and querying indexes
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="claustrum",
    help="Cross-ISA Semantic Code Embedding System",
    add_completion=False,
)

console = Console()


@app.command()
def embed(
    binary: str = typer.Argument(..., help="Path to binary file"),
    address: str = typer.Argument(..., help="Function address (hex)"),
    model: str = typer.Option("claustrum-base", help="Model name or path"),
    output: Optional[str] = typer.Option(None, help="Output file for embedding"),
):
    """Embed a function from a binary."""
    from claustrum.inference import ClaustrumEmbedder

    addr = int(address, 16) if address.startswith("0x") else int(address)

    console.print(f"Loading model: {model}")
    embedder = ClaustrumEmbedder.from_pretrained(model)

    console.print(f"Embedding function at {addr:#x} from {binary}")
    embedding = embedder.embed_function(binary, addr)

    console.print(f"Embedding shape: {embedding.shape}")

    if output:
        import torch

        torch.save(embedding, output)
        console.print(f"Saved embedding to {output}")
    else:
        console.print(f"Embedding (first 10 values): {embedding[:10].tolist()}")


@app.command()
def search(
    query_binary: str = typer.Argument(..., help="Binary containing query function"),
    query_addr: str = typer.Argument(..., help="Query function address (hex)"),
    index_path: str = typer.Argument(..., help="Path to FAISS index"),
    k: int = typer.Option(10, help="Number of results"),
    model: str = typer.Option("claustrum-base", help="Model name or path"),
):
    """Search for similar functions."""
    from claustrum.inference import ClaustrumEmbedder

    addr = int(query_addr, 16) if query_addr.startswith("0x") else int(query_addr)

    console.print(f"Loading model and index...")
    embedder = ClaustrumEmbedder.from_pretrained(model)
    embedder.load_index(index_path)

    console.print(f"Embedding query function at {addr:#x}")
    query_embedding = embedder.embed_function(query_binary, addr)

    console.print(f"Searching for {k} similar functions...")
    results = embedder.search(query_embedding, k=k)

    # Display results
    table = Table(title="Search Results")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Index")
    table.add_column("Metadata")

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        meta_str = f"{metadata.get('name', 'unknown')} @ {metadata.get('binary', 'unknown')}"
        table.add_row(
            str(i),
            f"{result['score']:.4f}",
            str(result["index"]),
            meta_str,
        )

    console.print(table)


@app.command()
def train(
    config: str = typer.Argument(..., help="Path to config file"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory"),
    resume: Optional[str] = typer.Option(None, help="Resume from checkpoint"),
):
    """Train a CLAUSTRUM model."""
    import yaml

    console.print(f"Loading config from {config}")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    if output_dir:
        cfg["training"]["output_dir"] = output_dir

    console.print(f"Training configuration:")
    console.print(f"  Output: {cfg['training']['output_dir']}")
    console.print(f"  Epochs: {cfg['training']['num_epochs']}")
    console.print(f"  Batch size: {cfg['training']['per_device_train_batch_size']}")

    # Import and run training
    from claustrum.training import ClaustrumTrainer, TrainingConfig
    from claustrum.model import ClaustrumEncoder, ClaustrumConfig

    model_cfg = ClaustrumConfig(**cfg.get("model", {}))
    model = ClaustrumEncoder(model_cfg)

    train_cfg = TrainingConfig(**cfg.get("training", {}))
    trainer = ClaustrumTrainer(model, train_cfg)

    if resume:
        trainer.load_checkpoint(resume)

    console.print("Starting training...")
    metrics = trainer.train()

    console.print("Training complete!")
    console.print(f"Final loss: {metrics['train_loss'][-1]:.4f}")


@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to model"),
    data_path: str = typer.Argument(..., help="Path to evaluation data"),
    output: Optional[str] = typer.Option(None, help="Output file for results"),
):
    """Evaluate a CLAUSTRUM model."""
    from claustrum.inference import ClaustrumEmbedder
    from claustrum.evaluation import evaluate_retrieval
    import torch
    import numpy as np

    console.print(f"Loading model from {model_path}")
    embedder = ClaustrumEmbedder.from_pretrained(model_path)

    console.print(f"Loading evaluation data from {data_path}")
    # Load evaluation data (expected format: dict with 'embeddings', 'labels')
    data = torch.load(data_path)

    embeddings = data["embeddings"].numpy()
    labels = data["labels"].numpy()
    isa_labels = data.get("isa_labels", None)
    if isa_labels is not None:
        isa_labels = isa_labels.numpy()

    console.print(f"Evaluating on {len(embeddings)} samples...")
    metrics = evaluate_retrieval(embeddings, labels, isa_labels)

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for name, value in metrics.to_dict().items():
        if value > 0:
            table.add_row(name, f"{value:.4f}")

    console.print(table)

    if output:
        import json

        with open(output, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        console.print(f"Results saved to {output}")


@app.command()
def build_index(
    embeddings_path: str = typer.Argument(..., help="Path to embeddings file"),
    output_path: str = typer.Argument(..., help="Output path for index"),
    index_type: str = typer.Option("ivf_pq", help="Index type: flat, ivf_pq, hnsw"),
):
    """Build FAISS index from embeddings."""
    import torch
    from claustrum.inference.index import FAISSIndex, IndexConfig

    console.print(f"Loading embeddings from {embeddings_path}")
    data = torch.load(embeddings_path)

    embeddings = data["embeddings"].numpy()
    metadata = data.get("metadata", None)

    console.print(f"Building {index_type} index for {len(embeddings)} embeddings")

    config = IndexConfig(
        index_type=index_type,
        embedding_dim=embeddings.shape[1],
    )
    index = FAISSIndex(config)
    index.build(embeddings, metadata)

    console.print(f"Saving index to {output_path}")
    index.save(output_path)

    console.print(f"Index built successfully with {index.ntotal} vectors")


@app.command()
def export(
    model_path: str = typer.Argument(..., help="Path to model"),
    output_path: str = typer.Argument(..., help="Output path for ONNX model"),
    quantize: bool = typer.Option(False, help="Apply INT8 quantization"),
):
    """Export model to ONNX format."""
    from claustrum.model import ClaustrumEncoder, ClaustrumConfig
    from claustrum.inference.export import export_to_onnx, quantize_model

    console.print(f"Loading model from {model_path}")
    config = ClaustrumConfig.from_pretrained(model_path)
    model = ClaustrumEncoder(config)

    import torch

    weights_path = Path(model_path) / "pytorch_model.bin"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    console.print(f"Exporting to ONNX: {output_path}")
    export_to_onnx(model, output_path)

    if quantize:
        quantized_path = output_path.replace(".onnx", "_int8.onnx")
        console.print(f"Quantizing to INT8: {quantized_path}")
        quantize_model(output_path, quantized_path)


@app.command()
def info():
    """Show CLAUSTRUM version and configuration."""
    from claustrum import __version__

    console.print(f"CLAUSTRUM v{__version__}")
    console.print()
    console.print("Cross-ISA Semantic Code Embedding System")
    console.print()
    console.print("Supported ISAs:")

    from claustrum.utils.types import ISA

    tier1 = [isa.value for isa in ISA if isa.tier == 1]
    tier2 = [isa.value for isa in ISA if isa.tier == 2]
    tier3 = [isa.value for isa in ISA if isa.tier == 3]
    tier4 = [isa.value for isa in ISA if isa.tier == 4]

    console.print(f"  Tier 1 (VEX): {', '.join(tier1)}")
    console.print(f"  Tier 2 (VEX/ESIL): {', '.join(tier2)}")
    console.print(f"  Tier 3 (ESIL): {', '.join(tier3)}")
    console.print(f"  Tier 4 (P-Code): {', '.join(tier4)}")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
