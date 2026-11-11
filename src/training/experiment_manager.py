"""
Centralized Experiment Registry and Tracking System
Calibrated Concept-Based Feedback Model (CSE 4385)

Tracks and persists:
- Model architecture (ResNet-50, CBAM, FPN, EfficientNet-B2, ConvNeXt-Tiny)
- Hyperparameters (batch size, learning rates, loss weights, seeds)
- Epoch logs, best validation scores, training duration, GPU memory
- Checkpoint file references
- Saves to outputs/experiment_registry/registry.json and experiments_summary.csv
"""

from csv import DictWriter
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Tracks and registers experimental runs.
    """

    def __init__(self, registry_dir: Union[str, Path] = "outputs/experiment_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_json = self.registry_dir / "registry.json"
        self.summary_csv = self.registry_dir / "experiments_summary.csv"

        self.experiments: Dict[str, Dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if self.registry_json.exists():
            try:
                with open(self.registry_json, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load registry JSON: {e}")
        return {}

    def _save_registry(self) -> None:
        with open(self.registry_json, "w", encoding="utf-8") as f:
            json.dump(self.experiments, f, indent=2)

        # Update CSV summary
        if self.experiments:
            fieldnames = [
                "experiment_id",
                "model_name",
                "status",
                "created_at",
                "completed_epochs",
                "best_epoch",
                "best_val_species_top1",
                "best_val_concept_f1",
                "best_val_total_loss",
                "learning_rate",
                "lambda_concept",
                "total_params",
            ]
            with open(self.summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for exp in self.experiments.values():
                    writer.writerow(exp)

    def register_experiment(
        self,
        experiment_id: str,
        model_name: str,
        config: Dict[str, Any],
        total_params: int = 0,
    ) -> Dict[str, Any]:
        """Registers a new or existing experiment run."""
        record = self.experiments.get(experiment_id, {})
        record.update({
            "experiment_id": experiment_id,
            "model_name": model_name,
            "status": "RUNNING",
            "created_at": record.get("created_at", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "config": config,
            "total_params": total_params,
            "completed_epochs": record.get("completed_epochs", 0),
            "best_epoch": record.get("best_epoch", -1),
            "best_val_species_top1": record.get("best_val_species_top1", 0.0),
            "best_val_concept_f1": record.get("best_val_concept_f1", 0.0),
            "best_val_total_loss": record.get("best_val_total_loss", float("inf")),
            "history": record.get("history", []),
        })
        self.experiments[experiment_id] = record
        self._save_registry()
        return record

    def log_epoch(
        self,
        experiment_id: str,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_best: bool,
    ) -> None:
        """Records epoch results for an experiment."""
        if experiment_id not in self.experiments:
            self.register_experiment(experiment_id, "unknown", {})

        exp = self.experiments[experiment_id]
        exp["completed_epochs"] = max(exp.get("completed_epochs", 0), epoch)
        exp["updated_at"] = datetime.now().isoformat()

        if is_best:
            exp["best_epoch"] = epoch
            exp["best_val_species_top1"] = val_metrics.get("species_top1_acc", 0.0)
            exp["best_val_concept_f1"] = val_metrics.get("concept_macro_f1", 0.0)
            exp["best_val_total_loss"] = val_metrics.get("total_loss", float("inf"))

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "is_best": is_best,
            "timestamp": datetime.now().isoformat(),
        }
        exp["history"].append(epoch_record)
        self._save_registry()

    def mark_completed(self, experiment_id: str) -> None:
        """Marks experiment as completed."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["status"] = "COMPLETED"
            self.experiments[experiment_id]["completed_at"] = datetime.now().isoformat()
            self._save_registry()

