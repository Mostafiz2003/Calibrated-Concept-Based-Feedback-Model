"""
CUB-200-2011 Concept Audit and Selection Engine
Calibrated Concept-Based Feedback Model (CSE 4385)

Analyzes the 312 CUB-200-2011 attributes, computes occurrence statistics
across bird species classes under clearly defined counting rules, and audits
candidate filtering rules against the required 112-concept target.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.data.metadata_parser import CUBMetadata, DatasetNotFoundError, CorruptAnnotationError


@dataclass
class AttributeStat:
    """Statistics for an individual CUB visual attribute."""
    attribute_id: int
    attribute_name: str
    total_image_occurrences: int
    train_image_occurrences: int
    qualifying_classes_count: int
    qualifies: bool


@dataclass
class AttributeAuditReport:
    """Summary of the attribute audit across all attributes."""
    counting_rule: str
    min_classes_threshold: int
    total_attributes_audited: int
    qualifying_attribute_ids: List[int]
    attribute_stats: List[AttributeStat]
    target_concept_count: int = 112

    @property
    def qualifying_count(self) -> int:
        return len(self.qualifying_attribute_ids)

    @property
    def discrepancy_from_target(self) -> int:
        """Difference from the target 112 concepts. 0 means exact match."""
        return self.qualifying_count - self.target_concept_count

    def summary_table(self, top_n: int = 10) -> str:
        """Returns a formatted summary text of the audit."""
        lines = [
            f"Audit Rule:                  {self.counting_rule}",
            f"Species Threshold:           >= {self.min_classes_threshold} bird species",
            f"Total Attributes Audited:    {self.total_attributes_audited}",
            f"Qualifying Attributes:       {self.qualifying_count}",
            f"Target Concept Count:        {self.target_concept_count}",
            f"Discrepancy (Count - 112):   {self.discrepancy_from_target:+d}",
        ]
        if self.discrepancy_from_target == 0:
            lines.append("Status:                      EXACT MATCH (112 concepts)")
        else:
            lines.append(f"Status:                      MISMATCH ({abs(self.discrepancy_from_target)} attributes difference)")
        return "\n".join(lines)


class ConceptAuditor:
    """
    Audits CUB-200-2011 attributes and evaluates concept filtering rules.
    """

    def __init__(self, metadata: CUBMetadata):
        self.metadata = metadata

    def parse_image_attributes_stream(
        self,
        filepath: Optional[Path] = None,
        train_only: bool = True,
        min_certainty: int = 1,
    ) -> Dict[int, Dict[int, int]]:
        """
        Streams image_attribute_labels.txt and counts positive occurrences.

        Returns:
            attr_class_counts: {attr_id: {class_id: count_of_positive_images}}
        """
        if filepath is None:
            filepath = self.metadata.dataset_dir / "attributes" / "image_attribute_labels.txt"
            if not filepath.exists():
                filepath = self.metadata.dataset_dir / "image_attribute_labels.txt"

        if not filepath.exists():
            raise DatasetNotFoundError(f"image_attribute_labels.txt not found at {filepath}")

        # attr_id -> {class_id: count_of_images_with_attr}
        attr_class_counts: Dict[int, Dict[int, int]] = {
            attr_id: {} for attr_id in self.metadata.attributes.keys()
        }

        train_ids_set = set(self.metadata.train_image_ids) if train_only else None

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    raise CorruptAnnotationError(
                        f"Malformed record at line {line_no} in {filepath}: '{raw_line.strip()}'"
                    )
                try:
                    img_id = int(parts[0])
                    attr_id = int(parts[1])
                    is_present = int(parts[2])
                    certainty_id = int(parts[3]) if len(parts) >= 4 else 1
                except ValueError:
                    raise CorruptAnnotationError(
                        f"Invalid numeric values at line {line_no} in {filepath}: '{raw_line.strip()}'"
                    )

                if train_only and train_ids_set is not None and img_id not in train_ids_set:
                    continue

                if is_present == 1 and certainty_id >= min_certainty:
                    cls_id = self.metadata.image_classes.get(img_id)
                    if cls_id is not None:
                        class_counts = attr_class_counts.setdefault(attr_id, {})
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        return attr_class_counts

    def audit_attributes(
        self,
        counting_rule: str = "class_majority_50",
        min_classes: int = 10,
        train_only: bool = True,
        min_certainty: int = 1,
        image_attr_file: Optional[Path] = None,
    ) -> AttributeAuditReport:
        """
        Performs audit across all attributes according to the specified rule.

        Supported counting rules:
          - "class_majority_50": Attribute is present in a species if >= 50% of images
            of that species exhibit the attribute.
          - "class_any_positive": Attribute is present in a species if at least 1 image
            of that species exhibits the attribute.
          - "class_min_images_k": Attribute is present in a species if >= k images
            (e.g., 2) of that species exhibit the attribute.
        """
        attr_class_counts = self.parse_image_attributes_stream(
            filepath=image_attr_file,
            train_only=train_only,
            min_certainty=min_certainty,
        )

        # Compute total images per class (used for percentage rules)
        class_image_totals: Dict[int, int] = {}
        target_image_ids = self.metadata.train_image_ids if train_only else list(self.metadata.images.keys())
        for img_id in target_image_ids:
            cls_id = self.metadata.image_classes[img_id]
            class_image_totals[cls_id] = class_image_totals.get(cls_id, 0) + 1

        qualifying_ids: List[int] = []
        stats: List[AttributeStat] = []

        for attr_id, attr_name in sorted(self.metadata.attributes.items()):
            classes_dict = attr_class_counts.get(attr_id, {})
            train_occurrences = sum(classes_dict.values())

            qualifying_classes_count = 0
            for cls_id, count in classes_dict.items():
                total_in_class = class_image_totals.get(cls_id, 1)

                if counting_rule == "class_majority_50":
                    if (count / total_in_class) >= 0.50:
                        qualifying_classes_count += 1
                elif counting_rule == "class_any_positive":
                    if count >= 1:
                        qualifying_classes_count += 1
                elif counting_rule.startswith("class_min_images_"):
                    k = int(counting_rule.split("_")[-1])
                    if count >= k:
                        qualifying_classes_count += 1
                else:
                    raise ValueError(f"Unknown counting rule: '{counting_rule}'")

            meets_threshold = qualifying_classes_count >= min_classes
            if meets_threshold:
                qualifying_ids.append(attr_id)

            stats.append(
                AttributeStat(
                    attribute_id=attr_id,
                    attribute_name=attr_name,
                    total_image_occurrences=train_occurrences,  # tracked occurrences
                    train_image_occurrences=train_occurrences,
                    qualifying_classes_count=qualifying_classes_count,
                    qualifies=meets_threshold,
                )
            )

        return AttributeAuditReport(
            counting_rule=counting_rule,
            min_classes_threshold=min_classes,
            total_attributes_audited=len(self.metadata.attributes),
            qualifying_attribute_ids=qualifying_ids,
            attribute_stats=stats,
            target_concept_count=112,
        )


def save_concept_mapping(
    selected_attribute_ids: List[int],
    attributes_dict: Dict[int, str],
    output_path: Path,
    metadata_info: Optional[Dict] = None,
) -> Path:
    """
    Saves the concept selection mapping into a structured JSON file.
    Explicitly distinguishes between:
      - concept_idx: 0-indexed integer position in model concept vector [0, K-1].
      - cub_attribute_id: 1-indexed original CUB attribute ID [1, 312].
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    concepts = []
    for idx, attr_id in enumerate(selected_attribute_ids):
        concepts.append(
            {
                "concept_idx": idx,
                "cub_attribute_id": attr_id,
                "attribute_name": attributes_dict.get(attr_id, f"Attribute_{attr_id}"),
            }
        )

    payload = {
        "metadata": {
            "description": "Mapping from model concept index to original CUB-200-2011 attribute ID",
            "num_concepts": len(concepts),
            "generated_at": datetime.now().isoformat(),
            **(metadata_info or {}),
        },
        "concepts": concepts,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_path


def load_concept_mapping(mapping_path: Path) -> Dict[str, any]:
    """Loads and validates a concept mapping JSON file."""
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Concept mapping file not found at: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "concepts" not in data:
        raise ValueError(f"Invalid concept mapping format in {mapping_path}: missing 'concepts' key.")

    return data

