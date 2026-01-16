"""Training module for CLAUSTRUM.

Provides training infrastructure for:
- Self-supervised pretraining (MIM, CWP, DUP)
- Contrastive fine-tuning with Multi-Positive InfoNCE loss
- Curriculum learning for ISA pairs
- Hard negative mining
"""

from claustrum.training.losses import (
    MultiPositiveInfoNCE,
    ContrastiveLoss,
    compute_similarity_matrix,
)
from claustrum.training.curriculum import (
    ISACurriculum,
    CurriculumScheduler,
)
from claustrum.training.negative_mining import (
    HardNegativeMiner,
    ProgressiveHardNegativeMiner,
)
from claustrum.training.trainer import (
    ClaustrumTrainer,
    TrainingConfig,
)

__all__ = [
    "MultiPositiveInfoNCE",
    "ContrastiveLoss",
    "compute_similarity_matrix",
    "ISACurriculum",
    "CurriculumScheduler",
    "HardNegativeMiner",
    "ProgressiveHardNegativeMiner",
    "ClaustrumTrainer",
    "TrainingConfig",
]
