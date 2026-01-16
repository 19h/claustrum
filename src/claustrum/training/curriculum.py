"""Curriculum learning for ISA pairs.

Implements the curriculum strategy from the plan:
1. Start with architecturally similar pairs (ARM32 <-> ARM64)
2. Progress to RISC <-> RISC pairs (ARM <-> MIPS)
3. Then CISC <-> RISC (x86 <-> ARM)
4. Finally register <-> stack machines (all <-> JVM, WASM)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from claustrum.utils.types import ISA, get_isa_similarity_score


class CurriculumStage(Enum):
    """Curriculum learning stages."""

    SAME_FAMILY = auto()  # ARM32 <-> ARM64, x86 <-> x64
    SIMILAR_ISA = auto()  # RISC <-> RISC (ARM <-> MIPS)
    CROSS_PARADIGM = auto()  # CISC <-> RISC (x86 <-> ARM)
    STACK_MACHINES = auto()  # Register <-> Stack (all <-> JVM/WASM)
    ALL_PAIRS = auto()  # No restrictions


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    # Stage progression (cumulative epochs per stage)
    stage_epochs: Optional[dict[CurriculumStage, float]] = None

    # Minimum similarity threshold per stage
    stage_thresholds: Optional[dict[CurriculumStage, float]] = None

    # Whether to warm start with easier pairs
    warm_start: bool = True

    # Gradually reduce threshold
    smooth_transition: bool = True
    transition_epochs: int = 2

    def __post_init__(self):
        if self.stage_epochs is None:
            self.stage_epochs = {
                CurriculumStage.SAME_FAMILY: 10,
                CurriculumStage.SIMILAR_ISA: 20,
                CurriculumStage.CROSS_PARADIGM: 30,
                CurriculumStage.STACK_MACHINES: 40,
                CurriculumStage.ALL_PAIRS: float("inf"),
            }

        if self.stage_thresholds is None:
            self.stage_thresholds = {
                CurriculumStage.SAME_FAMILY: 0.7,  # Very similar
                CurriculumStage.SIMILAR_ISA: 0.5,  # Moderately similar
                CurriculumStage.CROSS_PARADIGM: 0.3,  # Less similar
                CurriculumStage.STACK_MACHINES: 0.1,  # Very different
                CurriculumStage.ALL_PAIRS: 0.0,  # All pairs
            }


class ISACurriculum:
    """Manages curriculum learning for ISA pairs.

    Determines which ISA pairs should be included in training
    at each stage based on their similarity.
    """

    # ISA pair difficulties
    SAME_FAMILY_PAIRS = [
        (ISA.ARM32, ISA.ARM64),
        (ISA.X86, ISA.X86_64),
        (ISA.MIPS32, ISA.MIPS64),
        (ISA.PPC32, ISA.PPC64),
        (ISA.RISCV32, ISA.RISCV64),
        (ISA.SPARC32, ISA.SPARC64),
    ]

    SIMILAR_ISA_PAIRS = [
        (ISA.ARM64, ISA.RISCV64),
        (ISA.ARM32, ISA.MIPS32),
        (ISA.RISCV64, ISA.MIPS64),
        (ISA.PPC64, ISA.MIPS64),
    ]

    CROSS_PARADIGM_PAIRS = [
        (ISA.X86_64, ISA.ARM64),
        (ISA.X86, ISA.ARM32),
        (ISA.X86_64, ISA.RISCV64),
        (ISA.X86_64, ISA.MIPS64),
    ]

    STACK_MACHINE_PAIRS = [
        (ISA.X86_64, ISA.JVM),
        (ISA.ARM64, ISA.JVM),
        (ISA.X86_64, ISA.WASM),
        (ISA.ARM64, ISA.WASM),
        (ISA.JVM, ISA.DALVIK),
    ]

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        epochs_per_stage: int = 10,
    ):
        self.config = config or CurriculumConfig()
        self.epochs_per_stage = epochs_per_stage
        self.current_epoch = 0
        self._current_stage = CurriculumStage.SAME_FAMILY
        
        # Ensure config dicts are initialized (they're set in __post_init__)
        assert self.config.stage_epochs is not None, "stage_epochs should be initialized"
        assert self.config.stage_thresholds is not None, "stage_thresholds should be initialized"
        
        # Override config stage epochs if epochs_per_stage provided
        if epochs_per_stage > 0 and config is None:
            cumulative = 0
            self.config.stage_epochs = {}
            for i, stage in enumerate(CurriculumStage):
                if stage == CurriculumStage.ALL_PAIRS:
                    self.config.stage_epochs[stage] = float("inf")
                else:
                    self.config.stage_epochs[stage] = cumulative + epochs_per_stage
                    cumulative += epochs_per_stage

    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage based on epoch."""
        return self._current_stage
    
    def get_allowed_isa_pairs(self) -> Optional[set[str]]:
        """Get set of allowed ISAs for current curriculum stage.
        
        Returns:
            Set of ISA names allowed at current stage, or None if all are allowed.
        """
        if self._current_stage == CurriculumStage.ALL_PAIRS:
            return None
            
        # Collect allowed ISAs based on stage
        allowed = set()
        
        if self._current_stage == CurriculumStage.SAME_FAMILY:
            # Only same-family pairs
            for isa1, isa2 in self.SAME_FAMILY_PAIRS:
                allowed.add(isa1.value if hasattr(isa1, 'value') else str(isa1))
                allowed.add(isa2.value if hasattr(isa2, 'value') else str(isa2))
                
        elif self._current_stage == CurriculumStage.SIMILAR_ISA:
            # Same family + similar ISAs
            for pairs in [self.SAME_FAMILY_PAIRS, self.SIMILAR_ISA_PAIRS]:
                for isa1, isa2 in pairs:
                    allowed.add(isa1.value if hasattr(isa1, 'value') else str(isa1))
                    allowed.add(isa2.value if hasattr(isa2, 'value') else str(isa2))
                    
        elif self._current_stage == CurriculumStage.CROSS_PARADIGM:
            # All register-based ISAs
            for pairs in [self.SAME_FAMILY_PAIRS, self.SIMILAR_ISA_PAIRS, self.CROSS_PARADIGM_PAIRS]:
                for isa1, isa2 in pairs:
                    allowed.add(isa1.value if hasattr(isa1, 'value') else str(isa1))
                    allowed.add(isa2.value if hasattr(isa2, 'value') else str(isa2))
                    
        elif self._current_stage == CurriculumStage.STACK_MACHINES:
            # All ISAs including stack machines
            return None  # Allow all at this point
            
        return allowed if allowed else None

    def update_epoch(self, epoch: int) -> None:
        """Update curriculum based on epoch number."""
        self.current_epoch = epoch

        stage_epochs = self.config.stage_epochs
        if stage_epochs is None:
            self._current_stage = CurriculumStage.ALL_PAIRS
            return

        cumulative = 0
        for stage in CurriculumStage:
            cumulative += stage_epochs.get(stage, 0)
            if epoch < cumulative:
                self._current_stage = stage
                return

        self._current_stage = CurriculumStage.ALL_PAIRS

    def get_similarity_threshold(self) -> float:
        """Get current similarity threshold for pair filtering."""
        stage_thresholds = self.config.stage_thresholds
        if stage_thresholds is None:
            return 0.0  # Default: allow all pairs
        
        base_threshold = stage_thresholds[self._current_stage]

        if not self.config.smooth_transition:
            return base_threshold

        # Smooth transition between stages
        # TODO: Implement gradual threshold decay
        return base_threshold

    def should_include_pair(self, isa1: ISA, isa2: ISA) -> bool:
        """Determine if an ISA pair should be included at current stage.

        Args:
            isa1: First ISA
            isa2: Second ISA

        Returns:
            True if pair should be included in training
        """
        if self._current_stage == CurriculumStage.ALL_PAIRS:
            return True

        similarity = get_isa_similarity_score(isa1, isa2)
        threshold = self.get_similarity_threshold()

        return similarity >= threshold

    def filter_batch(
        self,
        isa_labels: list[ISA],
        source_labels: list[int],
    ) -> list[bool]:
        """Filter a batch based on curriculum.

        Args:
            isa_labels: ISA for each sample
            source_labels: Source function label for each sample

        Returns:
            Boolean mask for samples to include
        """
        n = len(isa_labels)
        include = [True] * n

        if self._current_stage == CurriculumStage.ALL_PAIRS:
            return include

        # For each sample, check if it has valid positive pairs in batch
        for i in range(n):
            has_valid_positive = False
            for j in range(n):
                if i != j and source_labels[i] == source_labels[j]:
                    if self.should_include_pair(isa_labels[i], isa_labels[j]):
                        has_valid_positive = True
                        break
            include[i] = has_valid_positive

        return include

    def get_pair_weight(self, isa1: ISA, isa2: ISA) -> float:
        """Get curriculum-based weight for an ISA pair.

        Easier pairs get lower weight as training progresses.
        """
        similarity = get_isa_similarity_score(isa1, isa2)

        # In early stages, weight by similarity (easier pairs higher)
        # In later stages, inverse weight (harder pairs higher)
        if self._current_stage in (CurriculumStage.SAME_FAMILY, CurriculumStage.SIMILAR_ISA):
            return similarity
        else:
            return 1.0 - similarity * 0.5  # Still some weight for easy pairs


class CurriculumScheduler:
    """Scheduler for curriculum progression.

    Automatically advances curriculum based on training metrics.
    """

    def __init__(
        self,
        curriculum: ISACurriculum,
        metric_threshold: float = 0.6,
        patience: int = 5,
    ):
        self.curriculum = curriculum
        self.metric_threshold = metric_threshold
        self.patience = patience

        self._best_metric = 0.0
        self._epochs_without_improvement = 0

    def step(self, epoch: int, metric: Optional[float] = None) -> bool:
        """Update curriculum and check for stage advancement.

        Args:
            epoch: Current epoch
            metric: Optional validation metric (e.g., recall@1)

        Returns:
            True if stage advanced
        """
        old_stage = self.curriculum.current_stage
        self.curriculum.update_epoch(epoch)

        if metric is not None:
            if metric > self._best_metric:
                self._best_metric = metric
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += 1

            # Early advancement if metric is good
            if metric >= self.metric_threshold:
                self._epochs_without_improvement = 0

        return self.curriculum.current_stage != old_stage

    def should_advance(self) -> bool:
        """Check if curriculum should advance based on metrics."""
        return self._epochs_without_improvement >= self.patience
