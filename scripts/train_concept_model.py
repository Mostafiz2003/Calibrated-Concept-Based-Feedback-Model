#!/usr/bin/env python3
"""
Concept Predictor Training Script for CUB-200-2011 (CSE 4385)
Trains 112-concept predictor on official training partition with validation model selection.

Usage:
  # Smoke test (fast sanity check on 2 train batches, 1 val batch):
  python scripts/train_concept_model.py --smoke-test

  # Full training run:
  python scripts/train_concept_model.py --epochs 10 --batch-size 32 --lr 0.001
"""

import argparse
import logging
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import create_cub_dataloaders
from src.models.concept_model import build_concept_predictor
from src.training.trainer import ConceptTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Sets random seeds across Python, NumPy, and PyTorch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train 112-Concept Predictor on CUB-200-2011")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone (resnet18, resnet34, resnet50)")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--output-dir", type=str, default="outputs/concept_model", help="Directory for checkpoints")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 2-batch smoke test")

    args = parser.parse_args()

    # 1. Load Configuration
    config = load_config(args.config)
    seed = int(config.get("reproducibility", {}).get("seed", 42))
    set_seed(seed)

    # Resolve Hyperparameters
    epochs = args.epochs if args.epochs is not None else int(config.get("training", {}).get("num_epochs", 10))
    batch_size = args.batch_size if args.batch_size is not None else int(config.get("training", {}).get("batch_size", 32))
    lr = args.lr if args.lr is not None else float(config.get("training", {}).get("learning_rate", 0.001))
    weight_decay = float(config.get("training", {}).get("weight_decay", 0.0001))
    backbone = args.backbone or config.get("model", {}).get("fallback_backbone", "resnet18")

    # Resolve Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("      CALIBRATED CONCEPT-BASED FEEDBACK MODEL: TRAINING PIPELINE      ")
    print("=" * 70)
    print(f"Device:           {device}")
    print(f"Backbone:         {backbone} (Pretrained: True)")
    print(f"Batch Size:       {batch_size}")
    print(f"Learning Rate:    {lr}")
    print(f"Weight Decay:     {weight_decay}")
    print(f"Epochs Budget:    {epochs if not args.smoke_test else 'SMOKE-TEST (1 epoch, 2 batches)'}")
    print(f"Seed:             {seed}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 70)

    # 2. Initialize DataLoaders (Train and Val ONLY; Test is strictly isolated!)
    print("\n[1/4] Loading CUB-200-2011 DataLoaders (Train: 4,794, Val: 600)...")
    loaders = create_cub_dataloaders(
        config=config,
        batch_size=batch_size,
        splits=("train", "val"),  # Calib & Test are excluded from model training
        allow_test_split=False,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(f"      Train loader: {len(train_loader)} batches ({len(train_loader.dataset)} images)")
    print(f"      Val loader:   {len(val_loader)} batches ({len(val_loader.dataset)} images)")

    # 3. Build Concept Predictor
    print(f"\n[2/4] Initializing ConceptPredictor with backbone '{backbone}'...")
    model = build_concept_predictor(
        config=config,
        backbone_name=backbone,
        num_concepts=112,
        pretrained=True,
    )
    print(f"      Feature dimension: {model.feature_dim}")
    print(f"      Concept targets:   {model.num_concepts}")
    print(f"      Pretrained source: {model.pretrained_source}")

    # 4. Setup Optimizer & Loss Criterion
    # Fine-tune backbone with lower learning rate or joint training
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 5. Initialize Trainer
    trainer = ConceptTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        criterion=criterion,
        config=config,
    )

    if args.smoke_test:
        print("\n[3/4] Executing Smoke Test (2 train batches, 1 val batch)...")
        results = trainer.fit(
            num_epochs=1,
            max_train_batches=2,
            max_val_batches=1,
        )
        print("\n[4/4] Smoke test completed successfully!")
        print(f"      Checkpoint saved to: {results['best_checkpoint_path']}")
        return results

    # Full Training Run
    print(f"\n[3/4] Starting training for {epochs} epochs...")
    results = trainer.fit(num_epochs=epochs)

    print("\n[4/4] Training Complete!")
    print(f"      Completed Epochs:    {results['completed_epochs']}")
    print(f"      Best Validation Loss:{results['best_val_loss']:.4f} (Epoch {results['best_epoch']})")
    print(f"      Best Checkpoint:     {results['best_checkpoint_path']}")
    print(f"      Latest Checkpoint:   {results['latest_checkpoint_path']}")
    return results


if __name__ == "__main__":
    main()

