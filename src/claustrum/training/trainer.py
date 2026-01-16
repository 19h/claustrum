"""Training loop for CLAUSTRUM models.

Provides comprehensive training infrastructure supporting:
- Pretraining (MIM, CWP, DUP) with optional trace prediction
- Contrastive fine-tuning with Multi-Positive InfoNCE
- Curriculum learning with ISA-aware batch sampling
- Progressive hard negative mining
- Distributed training via Hugging Face Accelerate
- Mixed precision training (FP16/BF16)
- Model EMA (Exponential Moving Average)
- Gradient checkpointing for memory efficiency
- Comprehensive logging (wandb, tensorboard, console)
- Early stopping with patience
- Rotating checkpoint management
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Callable, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts,
)
from tqdm.auto import tqdm

from claustrum.model.config import ClaustrumConfig
from claustrum.model.encoder import ClaustrumEncoder
from claustrum.model.pretraining import PretrainingModel
from claustrum.training.losses import MultiPositiveInfoNCE, SupConLoss
from claustrum.training.curriculum import ISACurriculum, CurriculumScheduler, CurriculumStage
from claustrum.training.negative_mining import ProgressiveHardNegativeMiner
from claustrum.utils.logging import get_logger

logger = get_logger(__name__)


class SchedulerType(str, Enum):
    """Supported learning rate scheduler types."""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    ONE_CYCLE = "one_cycle"


class SaveStrategy(str, Enum):
    """Checkpoint saving strategies."""
    NO = "no"
    EPOCH = "epoch"
    STEPS = "steps"
    BEST = "best"


class EvalStrategy(str, Enum):
    """Evaluation strategies."""
    NO = "no"
    EPOCH = "epoch"
    STEPS = "steps"


@dataclass
class TrainingConfig:
    """Comprehensive configuration for CLAUSTRUM training.
    
    Supports both pretraining and contrastive fine-tuning configurations.
    """
    
    # === Output Configuration ===
    output_dir: str = "output"
    overwrite_output_dir: bool = False
    run_name: Optional[str] = None
    
    # === Training Duration ===
    num_epochs: int = 100
    max_steps: int = -1  # -1 means use num_epochs
    
    # === Batch Configuration ===
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    
    # === Optimizer Configuration ===
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # === Learning Rate Schedule ===
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0  # Overrides warmup_ratio if > 0
    num_cycles: float = 0.5  # For cosine_with_restarts
    lr_end: float = 0.0  # Final LR for polynomial decay
    
    # === Contrastive Learning ===
    temperature: float = 0.07
    use_hard_negatives: bool = True
    hard_negative_weight: float = 1.0
    
    # === Curriculum Learning ===
    use_curriculum: bool = True
    curriculum_epochs_per_stage: int = 10
    curriculum_metric_threshold: float = 0.5  # Advance when metric exceeds this
    
    # === Hard Negative Mining ===
    mining_start_epoch: int = 5
    mining_initial_temperature: float = 0.0
    mining_final_temperature: float = 2.0
    mining_warmup_epochs: int = 10
    memory_bank_size: int = 65536
    
    # === Checkpointing ===
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 3
    save_on_each_node: bool = False
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "recall@1"
    greater_is_better: bool = True
    
    # === Evaluation ===
    eval_strategy: str = "epoch"
    eval_steps: int = 500
    eval_delay: int = 0  # Don't evaluate for first N epochs
    
    # === Logging ===
    logging_dir: Optional[str] = None
    logging_strategy: str = "steps"
    logging_steps: int = 100
    logging_first_step: bool = True
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # === Distributed Training ===
    local_rank: int = -1
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    
    # === Mixed Precision ===
    fp16: bool = False
    bf16: bool = False
    fp16_opt_level: str = "O1"
    fp16_full_eval: bool = False
    
    # === Memory Optimization ===
    gradient_checkpointing: bool = False
    optim_memory_efficient: bool = False
    
    # === Model EMA ===
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_update_after_step: int = 100
    ema_update_every: int = 10
    
    # === Early Stopping ===
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0
    
    # === Reproducibility ===
    seed: int = 42
    data_seed: Optional[int] = None
    full_determinism: bool = False
    
    # === Debug ===
    debug: bool = False
    dataloader_prefetch_factor: int = 2
    
    def __post_init__(self):
        """Validate and process configuration."""
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
            
        if self.data_seed is None:
            self.data_seed = self.seed
            
        if self.run_name is None:
            self.run_name = os.path.basename(self.output_dir)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save_to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            config_dict = json.load(f)
        return cls(**config_dict)


@dataclass
class TrainerState:
    """Mutable state of the trainer during training."""
    
    epoch: int = 0
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    
    # Logging
    log_history: List[Dict[str, float]] = field(default_factory=list)
    
    # Best model tracking
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    
    # Early stopping
    early_stopping_patience_counter: int = 0
    
    # Curriculum
    curriculum_stage: int = 0
    
    # Training state
    is_world_process_zero: bool = True
    is_local_process_zero: bool = True
    is_hyper_param_search: bool = False
    trial_name: Optional[str] = None
    trial_params: Optional[Dict[str, Any]] = None
    
    def save_to_json(self, path: str) -> None:
        """Save state to JSON."""
        state_dict = asdict(self)
        with open(path, "w") as f:
            json.dump(state_dict, f, indent=2)
    
    @classmethod
    def load_from_json(cls, path: str) -> "TrainerState":
        """Load state from JSON."""
        with open(path) as f:
            state_dict = json.load(f)
        return cls(**state_dict)


class TrainerCallback:
    """Base class for trainer callbacks."""
    
    def on_init_end(self, config: TrainingConfig, state: TrainerState, **kwargs):
        """Called at the end of trainer initialization."""
        pass
    
    def on_train_begin(self, config: TrainingConfig, state: TrainerState, **kwargs):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, config: TrainingConfig, state: TrainerState, **kwargs):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, config: TrainingConfig, state: TrainerState, **kwargs):
        """Called at the beginning of an epoch."""
        pass
    
    def on_epoch_end(self, config: TrainingConfig, state: TrainerState, **kwargs):
        """Called at the end of an epoch."""
        pass
    
    def on_step_begin(self, config: TrainingConfig, state: TrainerState, **kwargs):
        """Called at the beginning of a training step."""
        pass
    
    def on_step_end(self, config: TrainingConfig, state: TrainerState, **kwargs):
        """Called at the end of a training step."""
        pass
    
    def on_evaluate(self, config: TrainingConfig, state: TrainerState, metrics: Dict, **kwargs):
        """Called after evaluation."""
        pass
    
    def on_save(self, config: TrainingConfig, state: TrainerState, **kwargs):
        """Called when saving a checkpoint."""
        pass
    
    def on_log(self, config: TrainingConfig, state: TrainerState, logs: Dict, **kwargs):
        """Called when logging."""
        pass


class EarlyStoppingCallback(TrainerCallback):
    """Callback for early stopping based on evaluation metric."""
    
    def __init__(self, patience: int = 10, threshold: float = 0.0):
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.patience_counter = 0
        self.should_stop = False
    
    def on_evaluate(self, config: TrainingConfig, state: TrainerState, metrics: Dict, **kwargs):
        metric_value = metrics.get(config.metric_for_best_model)
        if metric_value is None:
            return
        
        if self.best_metric is None:
            self.best_metric = metric_value
            return
        
        if config.greater_is_better:
            improved = metric_value > self.best_metric + self.threshold
        else:
            improved = metric_value < self.best_metric - self.threshold
        
        if improved:
            self.best_metric = metric_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered after {self.patience} evaluations without improvement")


class ModelEMA:
    """Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of the model with EMA-updated weights,
    which often provides better generalization than the final weights.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 100,
        update_every: int = 10,
    ):
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.step = 0
        
        # Create shadow model
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters."""
        self.step += 1
        
        if self.step < self.update_after_step:
            return
        
        if self.step % self.update_every != 0:
            return
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay
                    )
    
    def apply_shadow(self, model: nn.Module) -> None:
        """Apply EMA weights to model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module) -> None:
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self) -> Dict:
        """Get state dict for checkpointing."""
        return {
            "shadow": self.shadow,
            "step": self.step,
            "decay": self.decay,
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state dict."""
        self.shadow = state_dict["shadow"]
        self.step = state_dict["step"]
        self.decay = state_dict.get("decay", self.decay)


class CurriculumBatchSampler(Sampler):
    """Batch sampler that respects curriculum learning stages.
    
    Samples batches according to the current curriculum stage,
    preferring ISA pairs appropriate for the stage.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        curriculum: ISACurriculum,
        isa_labels: Optional[List[str]] = None,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.curriculum = curriculum
        self.isa_labels = isa_labels or []
        self.drop_last = drop_last
        
        # Build ISA index
        self.isa_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, isa in enumerate(self.isa_labels):
            self.isa_to_indices[isa].append(idx)
    
    def __iter__(self):
        """Generate batches according to curriculum stage."""
        stage = self.curriculum.current_stage
        allowed_isas = self.curriculum.get_allowed_isa_pairs()
        
        # Get indices for allowed ISAs
        if allowed_isas:
            allowed_indices = []
            for isa in allowed_isas:
                allowed_indices.extend(self.isa_to_indices.get(isa, []))
        else:
            allowed_indices = list(range(len(self.dataset)))
        
        # Shuffle
        np.random.shuffle(allowed_indices)
        
        # Generate batches
        batch = []
        for idx in allowed_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class ClaustrumTrainer:
    """Comprehensive trainer for CLAUSTRUM models.
    
    Supports:
    - Pretraining with MIM, CWP, DUP objectives
    - Contrastive fine-tuning with Multi-Positive InfoNCE
    - Curriculum learning with ISA-aware sampling
    - Progressive hard negative mining
    - Distributed training via Accelerate
    - Mixed precision (FP16/BF16)
    - Model EMA
    - Gradient checkpointing
    - Comprehensive checkpointing and logging
    - Early stopping
    - Callbacks system
    
    Args:
        model: The model to train
        config: Training configuration
        train_dataloader: Training data loader
        eval_dataloader: Optional evaluation data loader
        compute_metrics: Optional function to compute evaluation metrics
        callbacks: Optional list of callbacks
        optimizers: Optional tuple of (optimizer, scheduler)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Optional[tuple[Optimizer, LRScheduler]] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks or []
        
        # State
        self.state = TrainerState()
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self._setup_output_dir()
        
        # Setup device and distributed training
        self._setup_distributed()
        
        # Setup mixed precision
        self._setup_mixed_precision()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup gradient checkpointing
        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Setup optimizer and scheduler
        if optimizers is not None:
            self.optimizer, self.scheduler = optimizers
        else:
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
        
        # Setup loss functions
        self._setup_losses()
        
        # Setup curriculum learning
        self._setup_curriculum()
        
        # Setup hard negative mining
        self._setup_negative_mining()
        
        # Setup model EMA
        self._setup_ema()
        
        # Setup early stopping
        self._setup_early_stopping()
        
        # Setup logging
        self._setup_logging()
        
        # Calculate training steps
        self._calculate_training_steps()
        
        # Call init callbacks
        self._call_callbacks("on_init_end")
    
    def _setup_output_dir(self) -> None:
        """Setup output directory structure."""
        if self.output_dir.exists() and not self.config.overwrite_output_dir:
            existing_checkpoints = list(self.output_dir.glob("checkpoint-*"))
            if existing_checkpoints:
                logger.warning(
                    f"Output directory {self.output_dir} already exists with checkpoints. "
                    "Set overwrite_output_dir=True to overwrite."
                )
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        Path(self.config.logging_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_distributed(self) -> None:
        """Setup distributed training with Accelerate."""
        self.accelerator = None
        self.is_distributed = False
        
        try:
            from accelerate import Accelerator
            from accelerate.utils import DistributedType
            
            mixed_precision = None
            if self.config.fp16:
                mixed_precision = "fp16"
            elif self.config.bf16:
                mixed_precision = "bf16"
            
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision=mixed_precision,
                log_with=self.config.report_to if self.config.report_to else None,
                project_dir=self.config.logging_dir,
            )
            
            self.device = self.accelerator.device
            self.is_distributed = self.accelerator.distributed_type != DistributedType.NO
            self.state.is_world_process_zero = self.accelerator.is_main_process
            self.state.is_local_process_zero = self.accelerator.is_local_main_process
            
            logger.info(f"Using Accelerate with device: {self.device}")
            if self.is_distributed:
                logger.info(f"Distributed training with {self.accelerator.num_processes} processes")
                
        except ImportError:
            logger.info("Accelerate not available, using single-device training")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
    
    def _setup_mixed_precision(self) -> None:
        """Setup mixed precision training."""
        self.scaler = None
        
        if self.config.fp16 and self.accelerator is None:
            # Manual FP16 with GradScaler when not using Accelerate
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Using FP16 mixed precision with GradScaler")
    
    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "gradient_checkpointing_enable"):
            self.model.encoder.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on encoder")
    
    def _create_optimizer(self) -> AdamW:
        """Create optimizer with proper parameter groups."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        no_decay_names = ["bias", "layer_norm", "LayerNorm", "layernorm", "ln_"]
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if any(nd in name for nd in no_decay_names):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[LRScheduler]:
        """Create learning rate scheduler."""
        if self.train_dataloader is None:
            return None
        
        num_training_steps = self.state.max_steps
        num_warmup_steps = self._get_warmup_steps(num_training_steps)
        
        scheduler_type = self.config.lr_scheduler_type.lower()
        
        if scheduler_type == "linear":
            return self._create_linear_schedule(num_training_steps, num_warmup_steps)
        elif scheduler_type == "cosine":
            return self._create_cosine_schedule(num_training_steps, num_warmup_steps)
        elif scheduler_type == "cosine_with_restarts":
            return self._create_cosine_with_restarts_schedule(num_training_steps, num_warmup_steps)
        elif scheduler_type == "one_cycle":
            return self._create_one_cycle_schedule(num_training_steps)
        elif scheduler_type == "constant":
            return None
        elif scheduler_type == "constant_with_warmup":
            return self._create_constant_with_warmup_schedule(num_warmup_steps)
        else:
            return self._create_cosine_schedule(num_training_steps, num_warmup_steps)
    
    def _get_warmup_steps(self, num_training_steps: int) -> int:
        """Calculate warmup steps."""
        if self.config.warmup_steps > 0:
            return self.config.warmup_steps
        return int(num_training_steps * self.config.warmup_ratio)
    
    def _create_linear_schedule(self, num_training_steps: int, num_warmup_steps: int) -> LRScheduler:
        """Create linear schedule with warmup."""
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_cosine_schedule(self, num_training_steps: int, num_warmup_steps: int) -> LRScheduler:
        """Create cosine schedule with warmup."""
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=self.config.learning_rate * 0.01,
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps],
        )
    
    def _create_cosine_with_restarts_schedule(self, num_training_steps: int, num_warmup_steps: int) -> LRScheduler:
        """Create cosine schedule with warm restarts."""
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )
        
        # T_0 is the number of iterations for the first restart
        T_0 = (num_training_steps - num_warmup_steps) // max(1, int(1 / self.config.num_cycles))
        
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(1, T_0),
            T_mult=2,
            eta_min=self.config.learning_rate * 0.01,
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps],
        )
    
    def _create_one_cycle_schedule(self, num_training_steps: int) -> LRScheduler:
        """Create one cycle schedule."""
        return OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=num_training_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )
    
    def _create_constant_with_warmup_schedule(self, num_warmup_steps: int) -> LRScheduler:
        """Create constant schedule with warmup."""
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _setup_losses(self) -> None:
        """Setup loss functions."""
        self.contrastive_loss = MultiPositiveInfoNCE(
            temperature=self.config.temperature,
        )
        
        # Alternative supervised contrastive loss
        self.supcon_loss = SupConLoss(
            temperature=self.config.temperature,
        )
    
    def _setup_curriculum(self) -> None:
        """Setup curriculum learning."""
        self.curriculum = None
        self.curriculum_scheduler = None
        
        if self.config.use_curriculum:
            self.curriculum = ISACurriculum(
                epochs_per_stage=self.config.curriculum_epochs_per_stage,
            )
            self.curriculum_scheduler = CurriculumScheduler(
                curriculum=self.curriculum,
                metric_threshold=self.config.curriculum_metric_threshold,
            )
            logger.info("Curriculum learning enabled")
    
    def _setup_negative_mining(self) -> None:
        """Setup hard negative mining."""
        self.negative_miner = None
        
        if self.config.use_hard_negatives:
            model_config = getattr(self.model, "config", None)
            if model_config is None:
                model_config = ClaustrumConfig()
            
            self.negative_miner = ProgressiveHardNegativeMiner(
                embedding_dim=model_config.embedding_size,
                memory_bank_size=self.config.memory_bank_size,
                initial_temperature=self.config.mining_initial_temperature,
                final_temperature=self.config.mining_final_temperature,
                warmup_epochs=self.config.mining_warmup_epochs,
            )
            logger.info(f"Hard negative mining enabled with bank size {self.config.memory_bank_size}")
    
    def _setup_ema(self) -> None:
        """Setup model EMA."""
        self.ema = None
        
        if self.config.use_ema:
            self.ema = ModelEMA(
                model=self.model,
                decay=self.config.ema_decay,
                update_after_step=self.config.ema_update_after_step,
                update_every=self.config.ema_update_every,
            )
            logger.info(f"Model EMA enabled with decay {self.config.ema_decay}")
    
    def _setup_early_stopping(self) -> None:
        """Setup early stopping."""
        self.early_stopping_callback = None
        
        if self.config.early_stopping:
            self.early_stopping_callback = EarlyStoppingCallback(
                patience=self.config.early_stopping_patience,
                threshold=self.config.early_stopping_threshold,
            )
            self.callbacks.append(self.early_stopping_callback)
            logger.info(f"Early stopping enabled with patience {self.config.early_stopping_patience}")
    
    def _setup_logging(self) -> None:
        """Setup logging backends."""
        self.tb_writer = None
        self.wandb_run = None
        
        if self.state.is_world_process_zero:
            # TensorBoard
            if "tensorboard" in self.config.report_to:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self.tb_writer = SummaryWriter(log_dir=self.config.logging_dir)
                    logger.info(f"TensorBoard logging to {self.config.logging_dir}")
                except ImportError:
                    logger.warning("TensorBoard not available")
            
            # Weights & Biases
            if "wandb" in self.config.report_to:
                try:
                    import wandb
                    self.wandb_run = wandb.init(
                        project="claustrum",
                        name=self.config.run_name,
                        config=self.config.to_dict(),
                        dir=self.config.logging_dir,
                    )
                    logger.info("Weights & Biases logging enabled")
                except ImportError:
                    logger.warning("wandb not available")
    
    def _calculate_training_steps(self) -> None:
        """Calculate total training steps."""
        if self.train_dataloader is None:
            self.state.max_steps = 0
            return
        
        num_update_steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        num_update_steps_per_epoch = max(1, num_update_steps_per_epoch)
        
        if self.config.max_steps > 0:
            self.state.max_steps = self.config.max_steps
            self.state.num_train_epochs = math.ceil(self.config.max_steps / num_update_steps_per_epoch)
        else:
            self.state.max_steps = num_update_steps_per_epoch * self.config.num_epochs
            self.state.num_train_epochs = self.config.num_epochs
    
    def _call_callbacks(self, event: str, **kwargs) -> None:
        """Call all callbacks for an event."""
        for callback in self.callbacks:
            getattr(callback, event)(self.config, self.state, **kwargs)
    
    def train(self) -> Dict[str, Any]:
        """Run the full training loop.
        
        Returns:
            Dictionary with training metrics and history
        """
        if self.train_dataloader is None:
            raise ValueError("train_dataloader is required for training")
        
        # Set seed for reproducibility
        self._set_seed()
        
        # Prepare model and dataloaders with Accelerate
        if self.accelerator is not None:
            self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader
            )
            if self.eval_dataloader is not None:
                self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
            if self.scheduler is not None:
                self.scheduler = self.accelerator.prepare(self.scheduler)
        
        # Log training info
        self._log_training_info()
        
        # Call train begin callbacks
        self._call_callbacks("on_train_begin")
        
        # Training metrics
        training_metrics = {
            "train_loss": [],
            "learning_rates": [],
            "eval_metrics": [],
            "curriculum_stages": [],
        }
        
        # Training loop
        train_iterator = tqdm(
            range(self.state.num_train_epochs),
            desc="Training",
            disable=not self.state.is_world_process_zero,
        )
        
        for epoch in train_iterator:
            self.state.epoch = epoch
            self._call_callbacks("on_epoch_begin")
            
            # Update curriculum
            if self.curriculum is not None:
                self.curriculum.update_epoch(epoch)
                training_metrics["curriculum_stages"].append(self.curriculum.current_stage.value)
                if self.state.is_world_process_zero:
                    logger.info(f"Epoch {epoch}: Curriculum stage = {self.curriculum.current_stage.name}")
            
            # Update hard negative mining temperature
            if self.negative_miner is not None and epoch >= self.config.mining_start_epoch:
                self.negative_miner.update_temperature(epoch - self.config.mining_start_epoch)
            
            # Train epoch
            epoch_metrics = self._train_epoch(epoch)
            training_metrics["train_loss"].append(epoch_metrics["loss"])
            training_metrics["learning_rates"].append(epoch_metrics["learning_rate"])
            
            # Evaluate
            if self._should_evaluate(epoch):
                eval_metrics = self.evaluate()
                training_metrics["eval_metrics"].append(eval_metrics)
                
                # Update curriculum based on metrics
                if self.curriculum_scheduler is not None:
                    metric_value = eval_metrics.get(self.config.metric_for_best_model, 0.0)
                    self.curriculum_scheduler.step(epoch, metric_value)
                
                # Update best model
                self._update_best_model(eval_metrics)
                
                # Call evaluate callbacks (includes early stopping check)
                self._call_callbacks("on_evaluate", metrics=eval_metrics)
            
            # Save checkpoint
            if self._should_save(epoch):
                self._save_checkpoint(f"epoch-{epoch}")
            
            # Call epoch end callbacks
            self._call_callbacks("on_epoch_end")
            
            # Check early stopping
            if self.early_stopping_callback and self.early_stopping_callback.should_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Update progress bar
            train_iterator.set_postfix({
                "loss": f"{epoch_metrics['loss']:.4f}",
                "lr": f"{epoch_metrics['learning_rate']:.2e}",
            })
        
        # Call train end callbacks
        self._call_callbacks("on_train_end")
        
        # Load best model if configured
        if self.config.load_best_model_at_end and self.state.best_model_checkpoint:
            self._load_checkpoint(self.state.best_model_checkpoint)
            logger.info(f"Loaded best model from {self.state.best_model_checkpoint}")
        
        # Final save
        self._save_checkpoint("final")
        
        # Close logging
        self._close_logging()
        
        return training_metrics
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        # Progress bar for batches
        batch_iterator = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            leave=False,
            disable=not self.state.is_world_process_zero,
        )
        
        for batch_idx, batch in enumerate(batch_iterator):
            self._call_callbacks("on_step_begin")
            
            # Move batch to device (if not using Accelerate)
            if self.accelerator is None:
                batch = self._move_batch_to_device(batch)
            
            # Forward pass
            loss = self._training_step(batch)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.accelerator is not None:
                self.accelerator.backward(loss)
            elif self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Optimizer step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    if self.accelerator is not None:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                    elif self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.ema is not None:
                    self.ema.update(self.model)
                
                # Update global step
                self.state.global_step += 1
                
                # Logging
                if self._should_log():
                    self._log_metrics({
                        "train/loss": total_loss / num_batches,
                        "train/learning_rate": self._get_current_lr(),
                        "train/epoch": epoch,
                        "train/global_step": self.state.global_step,
                    })
                
                # Step-based checkpointing
                if self._should_save_steps():
                    self._save_checkpoint(f"step-{self.state.global_step}")
                
                # Step-based evaluation
                if self._should_evaluate_steps():
                    eval_metrics = self.evaluate()
                    self._update_best_model(eval_metrics)
                    self._call_callbacks("on_evaluate", metrics=eval_metrics)
            
            self._call_callbacks("on_step_end")
            
            # Update progress bar
            batch_iterator.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
            })
            
            # Check max steps
            if self.config.max_steps > 0 and self.state.global_step >= self.config.max_steps:
                break
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            "loss": avg_loss,
            "learning_rate": self._get_current_lr(),
            "epoch_time": epoch_time,
            "num_batches": num_batches,
        }
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute a single training step.
        
        Args:
            batch: Batch dictionary
            
        Returns:
            Loss tensor
        """
        # Determine if we're doing pretraining or contrastive fine-tuning
        is_pretraining = isinstance(self.model, PretrainingModel) or (
            hasattr(self.model, "module") and isinstance(self.model.module, PretrainingModel)
        )
        
        if is_pretraining:
            return self._pretraining_step(batch)
        else:
            return self._contrastive_step(batch)
    
    def _pretraining_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pretraining step with MIM/CWP/DUP objectives."""
        # Handle wrapped models (DDP, etc.)
        model = self.model.module if hasattr(self.model, "module") else self.model
        
        # Mixed precision context
        with self._autocast_context():
            outputs = model(**batch)
        
        loss = outputs.get("loss", torch.tensor(0.0, device=self.device))
        return loss
    
    def _contrastive_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Contrastive fine-tuning step."""
        # Handle wrapped models
        model = self.model.module if hasattr(self.model, "module") else self.model
        
        # Mixed precision context
        with self._autocast_context():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        
        embeddings = outputs["pooler_output"]
        labels = batch.get("source_labels")
        
        if labels is None:
            return torch.tensor(0.0, device=self.device)
        
        # Compute contrastive loss
        loss = self.contrastive_loss(embeddings, labels)
        
        # Hard negative mining
        if self.negative_miner is not None and self.state.epoch >= self.config.mining_start_epoch:
            # Update memory bank
            self.negative_miner.update_memory(embeddings.detach(), labels)
            
            # Get hard negatives
            hard_neg_embeddings, hard_neg_labels = self.negative_miner.sample_hard_negatives(
                embeddings, labels, num_negatives=embeddings.size(0) // 2
            )
            
            if hard_neg_embeddings is not None:
                # Compute loss with hard negatives
                combined_embeddings = torch.cat([embeddings, hard_neg_embeddings], dim=0)
                combined_labels = torch.cat([labels, hard_neg_labels], dim=0)
                
                hard_neg_loss = self.contrastive_loss(combined_embeddings, combined_labels)
                loss = loss + self.config.hard_negative_weight * hard_neg_loss
        
        return loss
    
    def _autocast_context(self):
        """Get autocast context for mixed precision."""
        if self.accelerator is not None:
            return self.accelerator.autocast()
        elif self.config.fp16:
            return torch.cuda.amp.autocast()
        elif self.config.bf16:
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            return torch.cuda.amp.autocast(enabled=False)
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        
        # Apply EMA weights for evaluation
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        
        all_embeddings = []
        all_labels = []
        all_isa_labels = []
        
        eval_iterator = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            leave=False,
            disable=not self.state.is_world_process_zero,
        )
        
        for batch in eval_iterator:
            if self.accelerator is None:
                batch = self._move_batch_to_device(batch)
            
            # Handle wrapped models
            model = self.model.module if hasattr(self.model, "module") else self.model
            
            with self._autocast_context():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
            
            embeddings = outputs["pooler_output"]
            
            # Gather from all processes if distributed
            if self.accelerator is not None:
                embeddings = self.accelerator.gather(embeddings)
            
            all_embeddings.append(embeddings.cpu())
            
            if "source_labels" in batch:
                labels = batch["source_labels"]
                if self.accelerator is not None:
                    labels = self.accelerator.gather(labels)
                all_labels.append(labels.cpu())
            
            if "isas" in batch:
                all_isa_labels.extend(batch["isas"])
        
        # Restore original weights
        if self.ema is not None:
            self.ema.restore(self.model)
        
        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)
        
        metrics = {}
        
        if all_labels:
            labels = torch.cat(all_labels, dim=0)
            
            if self.compute_metrics is not None:
                metrics = self.compute_metrics(embeddings, labels)
        
        # Log metrics
        if metrics and self.state.is_world_process_zero:
            self._log_metrics({f"eval/{k}": v for k, v in metrics.items()})
            logger.info(f"Evaluation metrics: {metrics}")
        
        self.model.train()
        
        return metrics
    
    def _update_best_model(self, metrics: Dict[str, float]) -> None:
        """Update best model tracking."""
        metric_value = metrics.get(self.config.metric_for_best_model)
        
        if metric_value is None:
            return
        
        is_better = False
        if self.state.best_metric is None:
            is_better = True
        elif self.config.greater_is_better:
            is_better = metric_value > self.state.best_metric
        else:
            is_better = metric_value < self.state.best_metric
        
        if is_better:
            self.state.best_metric = metric_value
            checkpoint_name = f"best-{self.config.metric_for_best_model}-{metric_value:.4f}"
            self._save_checkpoint(checkpoint_name)
            self.state.best_model_checkpoint = str(self.output_dir / "checkpoints" / checkpoint_name)
            logger.info(f"New best model: {self.config.metric_for_best_model} = {metric_value:.4f}")
    
    def _should_evaluate(self, epoch: int) -> bool:
        """Check if should evaluate at this epoch."""
        if self.eval_dataloader is None:
            return False
        if epoch < self.config.eval_delay:
            return False
        if self.config.eval_strategy == "no":
            return False
        if self.config.eval_strategy == "epoch":
            return True
        return False
    
    def _should_evaluate_steps(self) -> bool:
        """Check if should evaluate at this step."""
        if self.eval_dataloader is None:
            return False
        if self.config.eval_strategy != "steps":
            return False
        return self.state.global_step % self.config.eval_steps == 0
    
    def _should_save(self, epoch: int) -> bool:
        """Check if should save at this epoch."""
        if self.config.save_strategy == "no":
            return False
        if self.config.save_strategy == "epoch":
            return True
        return False
    
    def _should_save_steps(self) -> bool:
        """Check if should save at this step."""
        if self.config.save_strategy != "steps":
            return False
        return self.state.global_step % self.config.save_steps == 0
    
    def _should_log(self) -> bool:
        """Check if should log at this step."""
        if self.config.logging_strategy == "no":
            return False
        if self.config.logging_strategy == "epoch":
            return False
        return self.state.global_step % self.config.logging_steps == 0
    
    def _save_checkpoint(self, name: str) -> None:
        """Save a checkpoint.
        
        Args:
            name: Checkpoint name
        """
        if not self.state.is_world_process_zero:
            return
        
        checkpoint_dir = self.output_dir / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        if hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(str(checkpoint_dir))
        else:
            torch.save(model_to_save.state_dict(), checkpoint_dir / "pytorch_model.bin")
        
        # Save model config
        if hasattr(model_to_save, "config"):
            model_to_save.config.save_to_json(str(checkpoint_dir / "config.json"))
        
        # Save optimizer and scheduler
        training_state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }
        torch.save(training_state, checkpoint_dir / "optimizer.bin")
        
        # Save trainer state
        self.state.save_to_json(str(checkpoint_dir / "trainer_state.json"))
        
        # Save training config
        self.config.save_to_json(str(checkpoint_dir / "training_config.json"))
        
        # Save EMA
        if self.ema is not None:
            torch.save(self.ema.state_dict(), checkpoint_dir / "ema.bin")
        
        # Save negative miner
        if self.negative_miner is not None:
            torch.save(self.negative_miner.state_dict(), checkpoint_dir / "negative_miner.bin")
        
        # Call save callbacks
        self._call_callbacks("on_save")
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Rotate checkpoints
        self._rotate_checkpoints()
    
    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints to stay within limit."""
        if self.config.save_total_limit is None or self.config.save_total_limit <= 0:
            return
        
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoints = sorted(
            checkpoint_dir.glob("*"),
            key=lambda x: x.stat().st_mtime,
        )
        
        # Keep best and final checkpoints
        keep_patterns = ["best-", "final"]
        checkpoints_to_consider = [
            cp for cp in checkpoints
            if not any(pattern in cp.name for pattern in keep_patterns)
        ]
        
        # Remove old checkpoints
        while len(checkpoints_to_consider) > self.config.save_total_limit:
            checkpoint_to_remove = checkpoints_to_consider.pop(0)
            logger.info(f"Removing old checkpoint: {checkpoint_to_remove}")
            shutil.rmtree(checkpoint_to_remove)
    
    def _load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load a checkpoint.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_dir)
        
        # Load model
        model_path = checkpoint_path / "pytorch_model.bin"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            model_to_load = self.model.module if hasattr(self.model, "module") else self.model
            model_to_load.load_state_dict(state_dict)
        
        # Load optimizer and scheduler
        optimizer_path = checkpoint_path / "optimizer.bin"
        if optimizer_path.exists():
            training_state = torch.load(optimizer_path, map_location=self.device)
            self.optimizer.load_state_dict(training_state["optimizer"])
            if self.scheduler and training_state.get("scheduler"):
                self.scheduler.load_state_dict(training_state["scheduler"])
            if self.scaler and training_state.get("scaler"):
                self.scaler.load_state_dict(training_state["scaler"])
        
        # Load trainer state
        state_path = checkpoint_path / "trainer_state.json"
        if state_path.exists():
            self.state = TrainerState.load_from_json(str(state_path))
        
        # Load EMA
        ema_path = checkpoint_path / "ema.bin"
        if ema_path.exists() and self.ema is not None:
            self.ema.load_state_dict(torch.load(ema_path, map_location=self.device))
        
        # Load negative miner
        miner_path = checkpoint_path / "negative_miner.bin"
        if miner_path.exists() and self.negative_miner is not None:
            self.negative_miner.load_state_dict(torch.load(miner_path, map_location=self.device))
        
        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Public method to load a checkpoint.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
        """
        self._load_checkpoint(checkpoint_dir)
    
    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()[0]
        return self.optimizer.param_groups[0]["lr"]
    
    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to all backends."""
        if not self.state.is_world_process_zero:
            return
        
        # Add to history
        self.state.log_history.append({
            **metrics,
            "step": self.state.global_step,
            "epoch": self.state.epoch,
        })
        
        # TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, self.state.global_step)
        
        # Weights & Biases
        if self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=self.state.global_step)
        
        # Call log callbacks
        self._call_callbacks("on_log", logs=metrics)
    
    def _log_training_info(self) -> None:
        """Log training configuration."""
        if not self.state.is_world_process_zero:
            return
        
        logger.info("***** Running training *****")
        logger.info(f"  Num epochs = {self.state.num_train_epochs}")
        logger.info(f"  Batch size = {self.config.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.state.max_steps}")
        logger.info(f"  Learning rate = {self.config.learning_rate}")
        logger.info(f"  Scheduler = {self.config.lr_scheduler_type}")
        
        if self.is_distributed:
            logger.info(f"  Distributed training with {self.accelerator.num_processes} processes")
        
        if self.config.fp16:
            logger.info("  Using FP16 mixed precision")
        elif self.config.bf16:
            logger.info("  Using BF16 mixed precision")
        
        if self.config.gradient_checkpointing:
            logger.info("  Using gradient checkpointing")
        
        if self.config.use_curriculum:
            logger.info("  Using curriculum learning")
        
        if self.config.use_hard_negatives:
            logger.info("  Using hard negative mining")
        
        if self.config.use_ema:
            logger.info(f"  Using model EMA with decay {self.config.ema_decay}")
    
    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        import random
        
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if self.config.full_determinism:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _close_logging(self) -> None:
        """Close logging backends."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
    
    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return self.train_dataloader
    
    def get_eval_dataloader(self) -> Optional[DataLoader]:
        """Get evaluation dataloader."""
        return self.eval_dataloader
    
    def get_model(self) -> nn.Module:
        """Get the model (unwrapped from DDP if needed)."""
        return self.model.module if hasattr(self.model, "module") else self.model
