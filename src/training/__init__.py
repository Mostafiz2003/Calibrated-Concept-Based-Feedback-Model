"""
Training infrastructure, losses, metrics, and checkpointing for CUB-200-2011.
"""

from src.training.checkpoint_manager import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.experiment_manager import ExperimentManager
from src.training.losses import MultiTaskRecognitionLoss
from src.training.metrics import (
    compute_concept_metrics,
    compute_recognition_metrics,
    compute_species_metrics,
)
from src.training.trainer import MultiTaskRecognitionTrainer

__all__ = [
    "CheckpointManager",
    "EarlyStopping",
    "ExperimentManager",
    "MultiTaskRecognitionLoss",
    "MultiTaskRecognitionTrainer",
    "compute_concept_metrics",
    "compute_species_metrics",
    "compute_recognition_metrics",
]
