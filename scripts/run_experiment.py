#!/usr/bin/env python3
"""
Multi-Task Bird Recognition Experiment Runner
Calibrated Concept-Based Feedback Model (CSE 4385)

Usage Examples:
  # Fast 2-batch smoke test:
  python scripts/run_experiment.py --model resnet50 --smoke-test

  # Resume interrupted experiment:
  python scripts/run_experiment.py --model resnet50 --resume

  # Full training run (requires explicit user authorization):
  python scripts/run_experiment.py --model resnet50_cbam_fpn --epochs 40 --batch-size 32
"""

import argparse
import logging
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.optim as optim
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import create_cub_dataloaders
from src.models.factory import SUPPORTED_ARCHITECTURES, build_recognition_model
from src.training.checkpoint_manager import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.experiment_manager import ExperimentManager
from src.training.losses import MultiTaskRecognitionLoss
from src.training.trainer import MultiTaskRecognitionTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_config(model_name: str) -> dict:
    """Loads model-specific config if available, falls back to default."""
    cfg_path = PROJECT_ROOT / f"configs/models/{model_name}.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run Bird Species Recognition Experiment")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"Model architecture: {list(SUPPORTED_ARCHITECTURES.keys())}",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Epoch budget")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Head learning rate")
    parser.add_argument("--backbone-lr", type=float, default=None, help="Backbone learning rate")
    parser.add_argument("--lambda-concept", type=float, default=0.2, help="Auxiliary concept loss weight")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints", help="Directory for checkpoints")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-batch smoke test only (no full training)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest valid checkpoint")
    parser.add_argument("--use-amp", action="store_true", help="Enable Automatic Mixed Precision on GPU")

    args = parser.parse_args()

    # 1. Resolve Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # 2. Load Configs
    model_cfg = load_model_config(args.model)
    default_cfg = load_config()

    train_cfg = model_cfg.get("training", {})
    epochs = args.epochs or train_cfg.get("num_epochs", 40)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    head_lr = args.lr or train_cfg.get("head_lr", 0.001)
    backbone_lr = args.backbone_lr or train_cfg.get("backbone_lr", 0.0001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    lambda_c = args.lambda_concept if args.lambda_concept is not None else train_cfg.get("lambda_concept", 0.2)

    set_seed(42)

    experiment_id = f"{args.model}_cub200"
    checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir / args.model

    print("=" * 75)
    print("      MULTI-TASK BIRD RECOGNITION EXPERIMENT RUNNER                    ")
    print("=" * 75)
    print(f"Model Architecture:   {args.model} (Exp: {model_cfg.get('model', {}).get('experiment_id', 'N/A')})")
    print(f"Device:               {device}")
    print(f"Batch Size:           {batch_size}")
    print(f"Backbone LR / Head LR:{backbone_lr} / {head_lr}")
    print(f"Multi-task Lambda:    L_total = L_species + {lambda_c} * L_concept")
    print(f"Epoch Budget:         {epochs} (Smoke test: {args.smoke_test})")
    print(f"Resume Mode:          {args.resume}")
    print(f"Checkpoint Dir:       {checkpoint_dir}")
    print("=" * 75)

    # 3. Instantiate Model
    print(f"\n[1/5] Initializing {args.model} architecture...")
    model = build_recognition_model(
        model_name=args.model,
        num_classes=200,
        num_concepts=112,
        pretrained=True,
    )
    if hasattr(model, "get_parameter_counts"):
        pcounts = model.get_parameter_counts()
        print(f"      Total parameters:     {pcounts['total']:,}")
        print(f"      Trainable parameters: {pcounts['trainable']:,}")

    # 4. Optimizer with Differential Learning Rates
    print("\n[2/5] Configuring AdamW optimizer with differential learning rates...")
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if "species_head" in name or "concept_head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=weight_decay)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Checkpoint & Experiment Managers
    ckpt_mgr = CheckpointManager(checkpoint_dir, experiment_id)
    exp_mgr = ExperimentManager(PROJECT_ROOT / "outputs/experiment_registry")
    early_stopping = EarlyStopping(patience=10, mode="max")

    if args.resume and ckpt_mgr.has_checkpoint():
        print("\n[3/5] Resuming from existing checkpoint...")
        ckpt_mgr.load_latest_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            early_stopping=early_stopping,
            device=device,
        )
    else:
        print("\n[3/5] Initializing clean experiment registration...")
        exp_mgr.register_experiment(
            experiment_id=experiment_id,
            model_name=args.model,
            config=model_cfg,
            total_params=sum(p.numel() for p in model.parameters()),
        )

    # 6. DataLoaders (Train and Val ONLY)
    print("\n[4/5] Loading train and validation DataLoaders...")
    loaders = create_cub_dataloaders(
        config=default_cfg,
        batch_size=batch_size,
        splits=("train", "val"),  # Strictly isolates calib and test!
        allow_test_split=False,
    )

    # 7. Multi-Task Trainer
    print("\n[5/5] Instantiating MultiTaskRecognitionTrainer...")
    criterion = MultiTaskRecognitionLoss(lambda_concept=lambda_c)
    trainer = MultiTaskRecognitionTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer,
        device=device,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        checkpoint_manager=ckpt_mgr,
        early_stopping=early_stopping,
        experiment_manager=exp_mgr,
        use_amp=args.use_amp,
        config=model_cfg,
    )

    if args.smoke_test:
        print("\n--> EXECUTING SMOKE TEST ONLY (1 epoch, 2 train batches, 1 val batch)...")
        results = trainer.fit(num_epochs=1, smoke_test=True)
        print("\n" + "=" * 75)
        print("                 SMOKE TEST COMPLETED SUCCESSFULLY                     ")
        print("=" * 75)
        print(f"Completed batches without error. Best Val Metric: {results['best_metric']:.4f}")
    else:
        print(f"\n--> STARTING FULL TRAINING RUN ({epochs} epochs)...")
        results = trainer.fit(num_epochs=epochs, smoke_test=False)
        exp_mgr.mark_completed(experiment_id)
        print("\n" + "=" * 75)
        print("                 EXPERIMENT COMPLETED SUCCESSFULLY                     ")
        print("=" * 75)


if __name__ == "__main__":
    main()

