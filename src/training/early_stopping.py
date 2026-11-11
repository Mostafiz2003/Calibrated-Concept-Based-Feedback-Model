"""
Early Stopping Handler for Model Training
Calibrated Concept-Based Feedback Model (CSE 4385)

Monitors validation metrics, manages patience, and signals training termination
when performance ceases to improve. Supports full serialization for checkpoint recovery.
"""

from typing import Any, Dict


class EarlyStopping:
    """
    Early stopping monitor.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change in monitored metric to qualify as an improvement.
        mode: 'max' (for accuracy/F1) or 'min' (for loss).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
    ):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode.lower()
        if self.mode not in ("max", "min"):
            raise ValueError(f"Mode must be 'max' or 'min', got '{mode}'")

        self.counter: int = 0
        self.best_score: float = -float("inf") if self.mode == "max" else float("inf")
        self.best_epoch: int = -1
        self.should_stop: bool = False

    def update(self, current_metric: float, epoch: int) -> bool:
        """
        Updates early stopping state with the current epoch's metric.

        Args:
            current_metric: Evaluation score (e.g. species Top-1 accuracy or val loss).
            epoch: Current epoch index.

        Returns:
            is_best: True if this epoch achieved a new best metric.
        """
        if self.mode == "max":
            improved = current_metric > (self.best_score + self.min_delta)
        else:
            improved = current_metric < (self.best_score - self.min_delta)

        if improved:
            self.best_score = float(current_metric)
            self.best_epoch = int(epoch)
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def state_dict(self) -> Dict[str, Any]:
        """Returns state for checkpointing."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "counter": self.counter,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restores state from checkpoint."""
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.mode = state.get("mode", self.mode)
        self.counter = state.get("counter", 0)
        self.best_score = state.get("best_score", self.best_score)
        self.best_epoch = state.get("best_epoch", -1)
        self.should_stop = state.get("should_stop", False)

