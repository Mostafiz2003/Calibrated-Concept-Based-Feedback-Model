"""
Tests for Week 5 Concept Evaluation and Per-Concept Breakdown Analysis
"""

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.concept_analysis import (
    load_concept_metadata,
    collect_validation_predictions,
    compute_per_concept_breakdown,
    compute_aggregate_metrics,
    save_week5_analysis_artifacts,
)


def test_load_concept_metadata():
    mapping_file = Path("data/selected_concepts_112.json")
    if not mapping_file.exists():
        pytest.skip("data/selected_concepts_112.json not found.")

    metadata = load_concept_metadata(mapping_file)
    assert len(metadata) == 112

    for i, c in enumerate(metadata):
        assert c["concept_index"] == i
        assert "cub_attribute_id" in c
        assert "attribute_name" in c
        assert isinstance(c["attribute_name"], str)
        assert len(c["attribute_name"]) > 0


def test_compute_per_concept_breakdown_synthetic():
    # 4 samples, 2 concepts
    # Concept 0: y=[1, 1, 0, 0], p=[0.9, 0.4, 0.1, 0.8] -> pred=[1, 0, 0, 1]
    # TP=1, FP=1, FN=1, TN=1 -> Acc=0.5, Prec=0.5, Rec=0.5, F1=0.5, Support=2, Prev=0.5
    # Concept 1: y=[0, 0, 0, 0], p=[0.2, 0.1, 0.3, 0.4] -> pred=[0, 0, 0, 0]
    # TP=0, FP=0, FN=0, TN=4 -> Acc=1.0, Prec=0.0, Rec=0.0, F1=0.0, Support=0, Prev=0.0
    targets = np.array([[1, 0], [1, 0], [0, 0], [0, 0]], dtype=np.float32)
    probs = np.array([[0.9, 0.2], [0.4, 0.1], [0.1, 0.3], [0.8, 0.4]], dtype=np.float32)

    meta = [
        {"concept_index": 0, "cub_attribute_id": 10, "attribute_name": "attr_a"},
        {"concept_index": 1, "cub_attribute_id": 20, "attribute_name": "attr_b"},
    ]

    breakdown = compute_per_concept_breakdown(targets, probs, threshold=0.5, concept_metadata=meta)
    assert len(breakdown) == 2

    # Concept 0
    c0 = breakdown[0]
    assert c0["support"] == 2
    assert c0["positive_prevalence"] == 0.5
    assert c0["true_positives"] == 1
    assert c0["false_positives"] == 1
    assert c0["false_negatives"] == 1
    assert c0["true_negatives"] == 1
    assert c0["accuracy"] == 0.5
    assert c0["precision"] == 0.5
    assert c0["recall"] == 0.5
    assert c0["f1_score"] == 0.5
    assert c0["attribute_name"] == "attr_a"

    # Concept 1 (zero support edge case)
    c1 = breakdown[1]
    assert c1["support"] == 0
    assert c1["positive_prevalence"] == 0.0
    assert c1["true_positives"] == 0
    assert c1["false_positives"] == 0
    assert c1["precision"] == 0.0
    assert c1["recall"] == 0.0
    assert c1["f1_score"] == 0.0
    assert c1["accuracy"] == 1.0


def test_compute_aggregate_metrics():
    # 2 samples, 2 concepts
    # Sample 0: true=[1, 0], pred=[1, 0] (exact match)
    # Sample 1: true=[0, 1], pred=[0, 0] (mismatch)
    targets = np.array([[1, 0], [0, 1]], dtype=np.float32)
    probs = np.array([[0.8, 0.2], [0.1, 0.4]], dtype=np.float32)

    agg = compute_aggregate_metrics(targets, probs, loss_val=0.45, threshold=0.5)

    assert agg["loss"] == 0.45
    assert agg["num_samples"] == 2
    assert agg["num_concepts"] == 2
    assert agg["exact_match_ratio"] == 0.5
    assert agg["mean_concept_accuracy"] == 0.75  # 3 correct out of 4 entries

    # Check that micro and macro metrics exist
    assert "macro_f1" in agg
    assert "macro_precision" in agg
    assert "macro_recall" in agg
    assert "micro_f1" in agg
    assert "micro_precision" in agg
    assert "micro_recall" in agg
    assert "prevalence_summary" in agg


def test_collect_validation_predictions_synthetic():
    class DummyDataset(Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 224, 224),
                "concept_labels": torch.zeros(112),
                "species_label": 0,
            }

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 * 224 * 224, 112)

        def forward(self, x):
            flat = x.view(x.size(0), -1)
            return self.linear(flat)

    loader = DataLoader(DummyDataset(), batch_size=4)
    model = DummyModel()

    res = collect_validation_predictions(model, loader, device="cpu")
    assert res["logits"].shape == (8, 112)
    assert res["targets"].shape == (8, 112)
    assert res["probabilities"].shape == (8, 112)
    assert res["num_samples"] == 8
    assert np.all(res["probabilities"] >= 0.0)
    assert np.all(res["probabilities"] <= 1.0)


def test_save_week5_analysis_artifacts(tmp_path):
    per_concept = [
        {
            "concept_index": 0,
            "cub_attribute_id": 2,
            "attribute_name": "has_bill_shape::dagger",
            "support": 10,
            "positive_prevalence": 0.1,
            "predicted_positives": 5,
            "true_positives": 4,
            "false_positives": 1,
            "false_negatives": 6,
            "true_negatives": 89,
            "accuracy": 0.93,
            "precision": 0.8,
            "recall": 0.4,
            "f1_score": 0.5333,
        }
    ]
    aggregate = {
        "evaluation_split": "validation",
        "num_samples": 100,
        "num_concepts": 1,
        "threshold": 0.5,
        "loss": 0.35,
        "macro_f1": 0.5333,
        "macro_precision": 0.8,
        "macro_recall": 0.4,
        "micro_f1": 0.5333,
        "micro_precision": 0.8,
        "micro_recall": 0.4,
        "mean_concept_accuracy": 0.93,
        "exact_match_ratio": 0.93,
        "zero_support_count": 0,
        "zero_support_concepts": [],
        "zero_predicted_positives_count": 0,
        "zero_predicted_positives_concepts": [],
        "prevalence_summary": {
            "min_prevalence": 0.1,
            "max_prevalence": 0.1,
            "mean_prevalence": 0.1,
            "median_prevalence": 0.1,
        },
    }

    saved = save_week5_analysis_artifacts(per_concept, aggregate, tmp_path)
    assert saved["csv"].exists()
    assert saved["json"].exists()
    assert saved["markdown"].exists()

    with open(saved["json"], "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["macro_f1"] == 0.5333
