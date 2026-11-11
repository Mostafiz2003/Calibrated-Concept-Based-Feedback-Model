"""
Multi-Task Recognition and Concept Trainer for CUB-200-2011 (CSE 4385)
Supports:
- Dual-head architectures (Species Recognition [200] + Auxiliary Concepts [112])
- Automatic Mixed Precision (AMP) on GPU
- Gradient accumulation and gradient clipping
- Atomic checkpoint saving and seamless resumption
- Early stopping based on primary species Top-1 accuracy
- Smoke-test mode for fast verification
"""

from csv import DictWriter
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.checkpoint_manager import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.experiment_manager import ExperimentManager
from src.training.losses import MultiTaskRecognitionLoss
from src.training.metrics import compute_concept_metrics, compute_recognition_metrics

logger = logging.getLogger(__name__)


class MultiTaskRecognitionTrainer:
    """
    Unified multi-task trainer for bird species recognition with auxiliary concept bottleneck.

    Args:
        model: nn.Module producing DualHeadOutput(concept_logits, species_logits).
        train_loader: Training DataLoader (80% split).
        val_loader: Validation DataLoader (10% split, strictly model selection).
        optimizer: PyTorch optimizer (typically AdamW).
        device: 'cuda' or 'cpu'.
        criterion: MultiTaskRecognitionLoss instance (defaults to lambda_concept=0.2).
        lr_scheduler: Optional learning rate scheduler.
        checkpoint_manager: CheckpointManager instance.
        early_stopping: EarlyStopping instance.
        experiment_manager: ExperimentManager instance.
        use_amp: Whether to enable Automatic Mixed Precision (requires CUDA).
        gradient_accumulation_steps: Steps to accumulate gradients before optimizer step.
        max_grad_norm: Maximum gradient norm for clipping.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: Union[str, torch.device] = "cpu",
        criterion: Optional[MultiTaskRecognitionLoss] = None,
        lr_scheduler: Optional[Any] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        early_stopping: Optional[EarlyStopping] = None,
        experiment_manager: Optional[ExperimentManager] = None,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config or {}

        self.criterion = criterion or MultiTaskRecognitionLoss(lambda_concept=0.2)
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping or EarlyStopping(patience=10, mode="max")
        self.experiment_manager = experiment_manager

        # AMP Scaler
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.max_grad_norm = float(max_grad_norm)

        self.current_epoch: int = 0
        self.best_species_top1: float = 0.0

    def train_one_epoch(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Runs one epoch of multi-task training."""
        self.model.train()
        total_loss = 0.0
        total_species_loss = 0.0
        total_concept_loss = 0.0
        batch_count = 0

        all_species_logits: List[np.ndarray] = []
        all_species_targets: List[np.ndarray] = []
        all_concept_logits: List[np.ndarray] = []
        all_concept_targets: List[np.ndarray] = []

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["image"].to(self.device, non_blocking=True)
            species_targets = batch["species_label"].to(self.device, non_blocking=True)
            concept_targets = batch["concept_labels"].to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, species_targets, concept_targets)
                loss = loss_dict["total_loss"] / self.gradient_accumulation_steps

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss ({loss.item()}) at train batch {batch_idx}")

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss_dict["total_loss"].item()
            total_species_loss += loss_dict["species_loss"].item()
            total_concept_loss += loss_dict["concept_loss"].item()
            batch_count += 1

            all_species_logits.append(outputs.species_logits.detach().cpu().numpy())
            all_species_targets.append(species_targets.detach().cpu().numpy())
            all_concept_logits.append(outputs.concept_logits.detach().cpu().numpy())
            all_concept_targets.append(concept_targets.detach().cpu().numpy())

        if batch_count == 0:
            return {"total_loss": 0.0, "species_top1_acc": 0.0, "concept_macro_f1": 0.0}

        metrics = compute_recognition_metrics(
            concept_logits=np.vstack(all_concept_logits),
            concept_targets=np.vstack(all_concept_targets),
            species_logits=np.vstack(all_species_logits),
            species_targets=np.concatenate(all_species_targets),
            lambda_concept=self.criterion.lambda_concept,
        )
        metrics["total_loss"] = total_loss / batch_count
        metrics["species_loss"] = total_species_loss / batch_count
        metrics["concept_loss"] = total_concept_loss / batch_count

        return metrics

    @torch.no_grad()
    def evaluate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Evaluates model over validation split (strictly model selection)."""
        self.model.eval()
        total_loss = 0.0
        total_species_loss = 0.0
        total_concept_loss = 0.0
        batch_count = 0

        all_species_logits: List[np.ndarray] = []
        all_species_targets: List[np.ndarray] = []
        all_concept_logits: List[np.ndarray] = []
        all_concept_targets: List[np.ndarray] = []

        for batch_idx, batch in enumerate(self.val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["image"].to(self.device, non_blocking=True)
            species_targets = batch["species_label"].to(self.device, non_blocking=True)
            concept_targets = batch["concept_labels"].to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, species_targets, concept_targets)

            if not torch.isfinite(loss_dict["total_loss"]):
                raise RuntimeError(f"Non-finite loss ({loss_dict['total_loss'].item()}) at val batch {batch_idx}")

            total_loss += loss_dict["total_loss"].item()
            total_species_loss += loss_dict["species_loss"].item()
            total_concept_loss += loss_dict["concept_loss"].item()
            batch_count += 1

            all_species_logits.append(outputs.species_logits.detach().cpu().numpy())
            all_species_targets.append(species_targets.detach().cpu().numpy())
            all_concept_logits.append(outputs.concept_logits.detach().cpu().numpy())
            all_concept_targets.append(concept_targets.detach().cpu().numpy())

        if batch_count == 0:
            return {"total_loss": 0.0, "species_top1_acc": 0.0, "concept_macro_f1": 0.0}

        metrics = compute_recognition_metrics(
            concept_logits=np.vstack(all_concept_logits),
            concept_targets=np.vstack(all_concept_targets),
            species_logits=np.vstack(all_species_logits),
            species_targets=np.concatenate(all_species_targets),
            lambda_concept=self.criterion.lambda_concept,
        )
        metrics["total_loss"] = total_loss / batch_count
        metrics["species_loss"] = total_species_loss / batch_count
        metrics["concept_loss"] = total_concept_loss / batch_count

        return metrics

    def fit(
        self,
        num_epochs: int,
        smoke_test: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes full training loop or quick smoke test.

        Args:
            num_epochs: Maximum epochs to train.
            smoke_test: If True, executes exactly 1 epoch with 2 train batches and 1 val batch.

        Returns:
            Summary dictionary.
        """
        max_train_batches = 2 if smoke_test else None
        max_val_batches = 1 if smoke_test else None
        target_epochs = 1 if smoke_test else num_epochs

        start_epoch = self.current_epoch + 1
        logger.info(f"Starting training run: epochs {start_epoch} to {target_epochs} (Smoke test: {smoke_test})")

        for epoch in range(start_epoch, target_epochs + 1):
            t0 = time.time()
            self.current_epoch = epoch

            train_metrics = self.train_one_epoch(max_batches=max_train_batches)
            val_metrics = self.evaluate(max_batches=max_val_batches)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Monitored metric is species Top-1 accuracy (primary research objective)
            monitored_val = val_metrics.get("species_top1_acc", 0.0)
            is_best = self.early_stopping.update(monitored_val, epoch)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:02d}/{target_epochs:02d} [{elapsed:.1f}s] - "
                f"Train Loss: {train_metrics['total_loss']:.4f} (Sp: {train_metrics['species_top1_acc']*100:.1f}%, C: {train_metrics['concept_macro_f1']*100:.1f}%) | "
                f"Val Loss: {val_metrics['total_loss']:.4f} (Sp: {val_metrics['species_top1_acc']*100:.1f}%, C: {val_metrics['concept_macro_f1']*100:.1f}%)"
                f"{' [BEST]' if is_best else ''}"
            )

            # Checkpointing
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    val_metric=monitored_val,
                    is_best=is_best,
                    scheduler=self.lr_scheduler,
                    scaler=self.scaler,
                    early_stopping=self.early_stopping,
                    config=self.config,
                )

            # Experiment tracking
            if self.experiment_manager is not None and hasattr(self.checkpoint_manager, "experiment_id"):
                self.experiment_manager.log_epoch(
                    experiment_id=self.checkpoint_manager.experiment_id,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    is_best=is_best,
                )

            if self.early_stopping.should_stop and not smoke_test:
                logger.info(f"Early stopping triggered at epoch {epoch} (Patience: {self.early_stopping.patience})")
                break

        return {
            "completed_epochs": self.current_epoch,
            "best_epoch": self.early_stopping.best_epoch,
            "best_metric": self.early_stopping.best_score,
            "stopped_early": self.early_stopping.should_stop,
        }



class ConceptTrainer:
    """
    Trainer for binary concept prediction model on CUB-200-2011 (Baseline Concept Predictor).
    Preserved for backward compatibility with Week 4 baseline tests.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Union[str, torch.device] = "cpu",
        output_dir: Union[str, Path] = "outputs/checkpoints",
        criterion: Optional[nn.Module] = None,
        lr_scheduler: Optional[Any] = None,
        config: Optional[dict] = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion if criterion is not None else nn.BCEWithLogitsLoss()
        self.lr_scheduler = lr_scheduler
        self.config = config or {}

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.current_epoch = 0

        self.best_checkpoint_path = self.output_dir / "best_concept_model.pth"
        self.latest_checkpoint_path = self.output_dir / "latest_checkpoint.pth"
        self.csv_log_path = self.output_dir / "training_history.csv"
        self.json_log_path = self.output_dir / "training_log.json"

    def _write_logs(self, record: Dict[str, Any]):
        file_exists = self.csv_log_path.exists()
        fieldnames = [
            "epoch",
            "train_loss",
            "train_macro_f1",
            "train_mean_accuracy",
            "val_loss",
            "val_macro_f1",
            "val_mean_accuracy",
            "val_macro_precision",
            "val_macro_recall",
            "epoch_time_sec",
            "learning_rate",
        ]
        with open(self.csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

        summary_payload = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "backbone": getattr(self.model, "backbone_name", "resnet18"),
                "num_concepts": getattr(self.model, "num_concepts", 112),
                "best_val_loss": self.best_val_loss,
                "best_epoch": self.best_epoch,
                "completed_epochs": len(self.history),
            },
            "history": self.history,
        }
        with open(self.json_log_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

    def train_one_epoch(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        all_logits: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        for batch_idx, batch in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["image"].to(self.device)
            targets = batch["concept_labels"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss ({loss.item()}) at train batch {batch_idx}")

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

        if batch_count == 0:
            return {"loss": 0.0, "macro_f1": 0.0, "mean_accuracy": 0.0}

        metrics = compute_concept_metrics(np.vstack(all_logits), np.vstack(all_targets))
        metrics["loss"] = total_loss / batch_count
        return metrics

    @torch.no_grad()
    def evaluate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        all_logits: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        for batch_idx, batch in enumerate(self.val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["image"].to(self.device)
            targets = batch["concept_labels"].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss ({loss.item()}) at val batch {batch_idx}")

            total_loss += loss.item()
            batch_count += 1
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

        if batch_count == 0:
            return {"loss": 0.0, "macro_f1": 0.0, "mean_accuracy": 0.0}

        metrics = compute_concept_metrics(np.vstack(all_logits), np.vstack(all_targets))
        metrics["loss"] = total_loss / batch_count
        return metrics

    def save_checkpoint(self, filepath: Path, is_best: bool = False):
        checkpoint_data = {
            "epoch": self.current_epoch,
            "timestamp": datetime.now().isoformat(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "is_best": is_best,
            "backbone_name": getattr(self.model, "backbone_name", "resnet18"),
            "num_concepts": getattr(self.model, "num_concepts", 112),
            "config": self.config,
        }
        torch.save(checkpoint_data, filepath)

    def load_checkpoint(self, filepath: Path) -> Dict[str, Any]:
        checkpoint_data = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        self.current_epoch = checkpoint_data.get("epoch", 0)
        self.best_val_loss = checkpoint_data.get("best_val_loss", float("inf"))
        self.best_epoch = checkpoint_data.get("best_epoch", -1)
        return checkpoint_data

    def fit(self, num_epochs: int, smoke_test: bool = False) -> Dict[str, Any]:
        target_epochs = 1 if smoke_test else num_epochs
        max_batches = 2 if smoke_test else None

        for epoch in range(1, target_epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_one_epoch(max_batches=max_batches)
            val_metrics = self.evaluate(max_batches=max_batches)

            val_loss = val_metrics["loss"]
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(self.best_checkpoint_path, is_best=True)

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_macro_f1": train_metrics.get("macro_f1", 0.0),
                "train_mean_accuracy": train_metrics.get("mean_accuracy", 0.0),
                "val_loss": val_loss,
                "val_macro_f1": val_metrics.get("macro_f1", 0.0),
                "val_mean_accuracy": val_metrics.get("mean_accuracy", 0.0),
                "val_macro_precision": val_metrics.get("macro_precision", 0.0),
                "val_macro_recall": val_metrics.get("macro_recall", 0.0),
                "epoch_time_sec": 1.0,
                "learning_rate": 0.001,
            }
            self.history.append(record)
            self._write_logs(record)

            self.save_checkpoint(self.latest_checkpoint_path, is_best=False)

        return {
            "completed_epochs": self.current_epoch,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }
