#!/usr/bin/env python3
"""Training script for CLAUSTRUM models.

Supports:
- Pretraining with MIM/CWP/DUP objectives
- Contrastive fine-tuning with cross-ISA pairs
- Trace-augmented pretraining
- Curriculum learning

Usage:
    # Pretraining
    python scripts/train.py pretrain --config configs/base.yaml --data-path data/pretrain
    
    # Contrastive fine-tuning
    python scripts/train.py finetune --config configs/base.yaml --data-path data/train
    
    # Full pipeline
    python scripts/train.py full --config configs/base.yaml
"""

import os
import sys
from pathlib import Path

import typer
import yaml
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(help="CLAUSTRUM Training CLI")
console = Console()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


@app.command()
def pretrain(
    config: str = typer.Option("configs/base.yaml", "--config", "-c", help="Config file path"),
    data_path: str = typer.Option(..., "--data-path", "-d", help="Path to pretraining data"),
    output_dir: str = typer.Option("output/pretrain", "--output", "-o", help="Output directory"),
    epochs: int = typer.Option(100, "--epochs", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    learning_rate: float = typer.Option(3e-5, "--lr", help="Learning rate"),
    use_traces: bool = typer.Option(False, "--traces", help="Enable trace prediction"),
    resume: str = typer.Option(None, "--resume", help="Resume from checkpoint"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    """Run pretraining with masked instruction modeling."""
    console.print("[bold blue]CLAUSTRUM Pretraining[/bold blue]")
    
    # Load config
    cfg = load_config(config)
    cfg.training.output_dir = output_dir
    cfg.training.num_epochs = epochs
    cfg.training.per_device_train_batch_size = batch_size
    cfg.training.learning_rate = learning_rate
    cfg.training.seed = seed
    
    console.print(f"Data path: {data_path}")
    console.print(f"Output dir: {output_dir}")
    console.print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    
    # Import modules
    import torch
    from claustrum.model.config import ClaustrumConfig
    from claustrum.model.pretraining import PretrainingModel
    from claustrum.tokenization import IRTokenizer
    from claustrum.data import create_pretraining_dataloader
    from claustrum.training.trainer import ClaustrumTrainer, TrainingConfig
    
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create model
    console.print("\n[bold]Creating model...[/bold]")
    model_config = ClaustrumConfig(
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        hidden_dropout_prob=cfg.model.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.model.attention_probs_dropout_prob,
        max_position_embeddings=cfg.model.max_position_embeddings,
        embedding_size=cfg.model.embedding_size,
    )
    
    if use_traces:
        from claustrum.tracing.predictor import TraceAugmentedPretraining
        model = TraceAugmentedPretraining(model_config)
        console.print("Using trace-augmented pretraining")
    else:
        model = PretrainingModel(model_config)
        
    param_count = sum(p.numel() for p in model.parameters())
    console.print(f"Model parameters: {param_count:,}")
    
    # Create tokenizer
    tokenizer = IRTokenizer()
    
    # Create dataloaders
    console.print("\n[bold]Loading data...[/bold]")
    train_dataloader = create_pretraining_dataloader(
        data_path=data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=cfg.data.max_sequence_length,
        mlm_probability=cfg.pretraining.mlm_probability,
        num_workers=cfg.training.dataloader_num_workers,
    )
    console.print(f"Training batches: {len(train_dataloader)}")
    
    # Create trainer
    training_config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        seed=seed,
    )
    
    trainer = ClaustrumTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
    )
    
    # Resume from checkpoint
    if resume:
        console.print(f"Resuming from {resume}")
        trainer.load_checkpoint(resume)
    
    # Train
    console.print("\n[bold green]Starting pretraining...[/bold green]")
    metrics = trainer.train()
    
    console.print("\n[bold green]Pretraining complete![/bold green]")
    console.print(f"Final loss: {metrics['train_loss'][-1]:.4f}")
    console.print(f"Checkpoints saved to: {output_dir}")


@app.command()
def finetune(
    config: str = typer.Option("configs/base.yaml", "--config", "-c", help="Config file path"),
    data_path: str = typer.Option(..., "--data-path", "-d", help="Path to training data"),
    eval_path: str = typer.Option(None, "--eval-path", help="Path to evaluation data"),
    output_dir: str = typer.Option("output/finetune", "--output", "-o", help="Output directory"),
    pretrained: str = typer.Option(None, "--pretrained", "-p", help="Pretrained model path"),
    epochs: int = typer.Option(50, "--epochs", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    learning_rate: float = typer.Option(1e-5, "--lr", help="Learning rate"),
    temperature: float = typer.Option(0.07, "--temperature", "-t", help="Contrastive temperature"),
    use_curriculum: bool = typer.Option(True, "--curriculum", help="Use curriculum learning"),
    use_hard_negatives: bool = typer.Option(True, "--hard-negatives", help="Use hard negative mining"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    """Fine-tune with contrastive learning on cross-ISA pairs."""
    console.print("[bold blue]CLAUSTRUM Contrastive Fine-tuning[/bold blue]")
    
    # Load config
    cfg = load_config(config)
    cfg.training.output_dir = output_dir
    cfg.training.num_epochs = epochs
    cfg.training.per_device_train_batch_size = batch_size
    cfg.training.learning_rate = learning_rate
    cfg.training.temperature = temperature
    cfg.training.use_curriculum = use_curriculum
    cfg.training.use_hard_negatives = use_hard_negatives
    cfg.training.seed = seed
    
    console.print(f"Data path: {data_path}")
    console.print(f"Output dir: {output_dir}")
    console.print(f"Temperature: {temperature}")
    
    # Import modules
    import torch
    from claustrum.model.config import ClaustrumConfig
    from claustrum.model.encoder import ClaustrumEncoder
    from claustrum.tokenization import IRTokenizer
    from claustrum.data import create_train_dataloader, create_eval_dataloader
    from claustrum.training.trainer import ClaustrumTrainer, TrainingConfig
    from claustrum.evaluation.metrics import compute_retrieval_metrics
    
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create model
    console.print("\n[bold]Creating model...[/bold]")
    model_config = ClaustrumConfig(
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        embedding_size=cfg.model.embedding_size,
        use_cfg_gnn=cfg.model.use_cfg_gnn,
        gnn_hidden_size=cfg.model.gnn_hidden_size,
        gnn_num_layers=cfg.model.gnn_num_layers,
        pooling_type=cfg.model.pooling_type,
    )
    
    model = ClaustrumEncoder(model_config)
    
    # Load pretrained weights
    if pretrained:
        console.print(f"Loading pretrained model from {pretrained}")
        state_dict = torch.load(Path(pretrained) / "pytorch_model.bin", map_location="cpu")
        # Handle pretraining model wrapper
        encoder_state = {
            k.replace("encoder.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("encoder.")
        }
        if encoder_state:
            model.load_state_dict(encoder_state, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
            
    param_count = sum(p.numel() for p in model.parameters())
    console.print(f"Model parameters: {param_count:,}")
    
    # Create tokenizer
    tokenizer = IRTokenizer()
    
    # Create dataloaders
    console.print("\n[bold]Loading data...[/bold]")
    train_dataloader = create_train_dataloader(
        data_path=data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=cfg.data.max_sequence_length,
        num_workers=cfg.training.dataloader_num_workers,
    )
    console.print(f"Training batches: {len(train_dataloader)}")
    
    eval_dataloader = None
    if eval_path:
        eval_dataloader = create_eval_dataloader(
            data_path=eval_path,
            tokenizer=tokenizer,
            batch_size=cfg.training.per_device_eval_batch_size,
            max_length=cfg.data.max_sequence_length,
            num_workers=cfg.training.dataloader_num_workers,
        )
        console.print(f"Evaluation batches: {len(eval_dataloader)}")
    
    # Metrics function
    def compute_metrics(embeddings, labels):
        return compute_retrieval_metrics(
            embeddings.numpy(),
            labels.numpy(),
            ks=[1, 5, 10],
        )
    
    # Create trainer
    training_config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        use_curriculum=use_curriculum,
        use_hard_negatives=use_hard_negatives,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        eval_strategy=cfg.training.eval_strategy,
        seed=seed,
    )
    
    trainer = ClaustrumTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        compute_metrics=compute_metrics if eval_dataloader else None,
    )
    
    # Train
    console.print("\n[bold green]Starting fine-tuning...[/bold green]")
    metrics = trainer.train()
    
    console.print("\n[bold green]Fine-tuning complete![/bold green]")
    console.print(f"Final loss: {metrics['train_loss'][-1]:.4f}")
    if metrics.get("eval_metrics"):
        final_eval = metrics["eval_metrics"][-1]
        console.print(f"Final Recall@1: {final_eval.get('recall@1', 0):.4f}")
        console.print(f"Final MRR: {final_eval.get('mrr', 0):.4f}")
    console.print(f"Checkpoints saved to: {output_dir}")


@app.command()
def evaluate(
    config: str = typer.Option("configs/base.yaml", "--config", "-c", help="Config file path"),
    model_path: str = typer.Option(..., "--model", "-m", help="Path to model checkpoint"),
    data_path: str = typer.Option(..., "--data-path", "-d", help="Path to evaluation data"),
    benchmark: str = typer.Option(None, "--benchmark", help="Benchmark name (poj104, binkit, cisco)"),
    pool_size: int = typer.Option(10000, "--pool-size", help="Retrieval pool size"),
    output_file: str = typer.Option(None, "--output", "-o", help="Output results file"),
):
    """Evaluate a trained model."""
    console.print("[bold blue]CLAUSTRUM Evaluation[/bold blue]")
    
    import json
    import torch
    import numpy as np
    
    from claustrum.model.config import ClaustrumConfig
    from claustrum.model.encoder import ClaustrumEncoder
    from claustrum.tokenization import IRTokenizer
    from claustrum.evaluation.metrics import (
        compute_retrieval_metrics,
        compute_clustering_metrics,
    )
    
    # Load config
    cfg = load_config(config)
    
    # Create model
    console.print(f"\n[bold]Loading model from {model_path}...[/bold]")
    model_config = ClaustrumConfig(
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        embedding_size=cfg.model.embedding_size,
    )
    
    model = ClaustrumEncoder(model_config)
    model.load_state_dict(torch.load(Path(model_path) / "pytorch_model.bin", map_location="cpu"))
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load data
    console.print(f"\n[bold]Loading data from {data_path}...[/bold]")
    tokenizer = IRTokenizer()
    
    if benchmark:
        from claustrum.data import create_benchmark_dataloader
        dataloader, dataset = create_benchmark_dataloader(
            benchmark_name=benchmark,
            data_path=data_path,
            tokenizer=tokenizer,
            pool_size=pool_size,
        )
        ground_truth = dataset.get_ground_truth()
    else:
        from claustrum.data import create_eval_dataloader, BinaryFunctionDataset
        dataloader = create_eval_dataloader(
            data_path=data_path,
            tokenizer=tokenizer,
            batch_size=64,
        )
        dataset = BinaryFunctionDataset(data_path)
        # Generate ground truth from source_ids
        source_to_ids = {}
        for sample in dataset.samples:
            if sample.source_id not in source_to_ids:
                source_to_ids[sample.source_id] = []
            source_to_ids[sample.source_id].append(sample.function_id)
        ground_truth = {
            sample.function_id: [
                fid for fid in source_to_ids[sample.source_id]
                if fid != sample.function_id
            ]
            for sample in dataset.samples
        }
    
    # Generate embeddings
    console.print("\n[bold]Generating embeddings...[/bold]")
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            all_embeddings.append(outputs["pooler_output"].cpu())
            if "source_labels" in batch:
                all_labels.append(batch["source_labels"].cpu())
    
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    console.print(f"Generated {len(embeddings)} embeddings")
    
    # Compute metrics
    console.print("\n[bold]Computing metrics...[/bold]")
    
    if all_labels:
        labels = torch.cat(all_labels, dim=0).numpy()
        retrieval_metrics = compute_retrieval_metrics(embeddings, labels, ks=[1, 5, 10, 50])
        clustering_metrics = compute_clustering_metrics(embeddings, labels)
    else:
        # Compute from ground truth
        retrieval_metrics = {"note": "Ground truth evaluation not implemented yet"}
        clustering_metrics = {}
    
    results = {
        "model_path": model_path,
        "data_path": data_path,
        "pool_size": len(embeddings),
        "retrieval": retrieval_metrics,
        "clustering": clustering_metrics,
    }
    
    # Print results
    console.print("\n[bold green]Results:[/bold green]")
    console.print(f"Pool size: {len(embeddings)}")
    
    if "recall@1" in retrieval_metrics:
        console.print(f"\nRetrieval Metrics:")
        console.print(f"  Recall@1:  {retrieval_metrics.get('recall@1', 0):.4f}")
        console.print(f"  Recall@5:  {retrieval_metrics.get('recall@5', 0):.4f}")
        console.print(f"  Recall@10: {retrieval_metrics.get('recall@10', 0):.4f}")
        console.print(f"  MRR:       {retrieval_metrics.get('mrr', 0):.4f}")
        console.print(f"  mAP:       {retrieval_metrics.get('map', 0):.4f}")
        
    if "ari" in clustering_metrics:
        console.print(f"\nClustering Metrics:")
        console.print(f"  ARI:        {clustering_metrics.get('ari', 0):.4f}")
        console.print(f"  NMI:        {clustering_metrics.get('nmi', 0):.4f}")
        console.print(f"  Silhouette: {clustering_metrics.get('silhouette', 0):.4f}")
    
    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\nResults saved to: {output_file}")


@app.command()
def export(
    model_path: str = typer.Option(..., "--model", "-m", help="Path to model checkpoint"),
    output_path: str = typer.Option("model.onnx", "--output", "-o", help="Output ONNX path"),
    config: str = typer.Option("configs/base.yaml", "--config", "-c", help="Config file path"),
    quantize: bool = typer.Option(False, "--quantize", "-q", help="Apply INT8 quantization"),
):
    """Export model to ONNX format."""
    console.print("[bold blue]CLAUSTRUM Model Export[/bold blue]")
    
    import torch
    
    from claustrum.model.config import ClaustrumConfig
    from claustrum.model.encoder import ClaustrumEncoder
    from claustrum.inference.export import ONNXExporter
    
    # Load config
    cfg = load_config(config)
    
    # Create and load model
    console.print(f"\n[bold]Loading model from {model_path}...[/bold]")
    model_config = ClaustrumConfig(
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        embedding_size=cfg.model.embedding_size,
    )
    
    model = ClaustrumEncoder(model_config)
    model.load_state_dict(torch.load(Path(model_path) / "pytorch_model.bin", map_location="cpu"))
    model.eval()
    
    # Export
    console.print(f"\n[bold]Exporting to {output_path}...[/bold]")
    exporter = ONNXExporter(model, model_config)
    exporter.export(
        output_path=output_path,
        quantize=quantize,
    )
    
    console.print(f"\n[bold green]Export complete![/bold green]")
    console.print(f"ONNX model saved to: {output_path}")
    if quantize:
        console.print("Applied INT8 quantization")


@app.command()
def full(
    config: str = typer.Option("configs/base.yaml", "--config", "-c", help="Config file path"),
    pretrain_data: str = typer.Option(..., "--pretrain-data", help="Pretraining data path"),
    train_data: str = typer.Option(..., "--train-data", help="Training data path"),
    eval_data: str = typer.Option(None, "--eval-data", help="Evaluation data path"),
    output_dir: str = typer.Option("output", "--output", "-o", help="Output directory"),
    pretrain_epochs: int = typer.Option(100, "--pretrain-epochs", help="Pretraining epochs"),
    finetune_epochs: int = typer.Option(50, "--finetune-epochs", help="Fine-tuning epochs"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    """Run full training pipeline: pretraining + fine-tuning."""
    console.print("[bold blue]CLAUSTRUM Full Training Pipeline[/bold blue]")
    
    pretrain_output = f"{output_dir}/pretrain"
    finetune_output = f"{output_dir}/finetune"
    
    # Phase 1: Pretraining
    console.print("\n[bold]Phase 1: Pretraining[/bold]")
    pretrain(
        config=config,
        data_path=pretrain_data,
        output_dir=pretrain_output,
        epochs=pretrain_epochs,
        seed=seed,
    )
    
    # Phase 2: Fine-tuning
    console.print("\n[bold]Phase 2: Contrastive Fine-tuning[/bold]")
    finetune(
        config=config,
        data_path=train_data,
        eval_path=eval_data,
        output_dir=finetune_output,
        pretrained=f"{pretrain_output}/checkpoint-best",
        epochs=finetune_epochs,
        seed=seed,
    )
    
    console.print("\n[bold green]Full pipeline complete![/bold green]")
    console.print(f"Pretrained model: {pretrain_output}")
    console.print(f"Fine-tuned model: {finetune_output}")


if __name__ == "__main__":
    app()
