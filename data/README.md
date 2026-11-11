# CUB-200-2011 Dataset & Pipeline Documentation

This document describes the data pipeline, metadata parsing, partition generation, and concept selection methodology for the **Caltech-UCSD Birds-200-2011 (CUB-200-2011)** dataset in the *Calibrated Concept-Based Feedback Model* project.

---

## 1. Dataset Overview and Statistics

- **Total Images:** 11,788 photos across 200 bird species.
- **Attributes:** 312 binary visual attributes per image.
- **Official Benchmark Splits:**
  - **Train partition:** 5,994 images.
  - **Test partition:** 5,794 images (strictly frozen and untouched until final evaluation).

---

## 2. Manual Download and Extraction

1. **Download:** Obtain `CUB_200_2011.tgz` (~1.15 GB) from Caltech Vision:
   - URL: [https://www.vision.caltech.edu/datasets/cub_200_2011/](https://www.vision.caltech.edu/datasets/cub_200_2011/)
2. **Extraction:** Extract directly into `data/`:
   ```powershell
   tar -xvzf "$HOME\Downloads\CUB_200_2011.tgz" -C data/
   ```
3. **Target Structure:** The dataset must sit at `data/CUB_200_2011/`:
   ```
   data/CUB_200_2011/
   ├── images/                          # 200 species subfolders
   ├── images.txt                       # 11,788 rows: <image_id> <path>
   ├── image_class_labels.txt           # 11,788 rows: <image_id> <class_id 1..200>
   ├── train_test_split.txt             # 11,788 rows: <image_id> <is_train 0/1>
   ├── classes.txt                      # 200 rows: <class_id> <species_name>
   └── attributes/
       ├── attributes.txt               # 312 rows: <attribute_id> <name>
       └── image_attribute_labels.txt   # ~3.6M rows: <image_id> <attr_id> <is_present> <certainty> <time>
   ```

---

## 3. Metadata Parser (`src/data/metadata_parser.py`)

The pipeline includes a strict annotation parser that enforces dataset integrity:
- **`parse_images_txt()`:** Validates unique positive integer image IDs and relative paths.
- **`parse_class_labels_txt()`:** Enforces valid species IDs in range `[1, 200]`.
- **`parse_train_test_split_txt()`:** Ensures split flags are strictly binary (`0` or `1`).
- **`parse_attributes_txt()`:** Validates the 312 attribute mappings.
- **Cross-Validation:** Ensures that every image in `images.txt` has a matching record in `image_class_labels.txt` and `train_test_split.txt`.
- **Error Handling:** Raises `DatasetNotFoundError` for missing files and `CorruptAnnotationError` with line numbers for malformed or out-of-range records.

---

## 4. Reproducible Data Partitions (`src/data/partitions.py`)

In accordance with proposal Section 5.1:
- The **official test set (5,794 images)** is kept completely isolated and untouched.
- The **official training set (5,994 images)** is partitioned deterministically using random seed `42`:
  - **Train split (~80%, ~4,795 images):** Used to train backbone feature extractors and concept heads.
  - **Validation split (~10%, ~600 images):** Used for model selection and fitting temperature scaling parameter $T$.
  - **Calibration split (~10%, ~599 images):** Held out strictly for calculating split-conformal nonconformity quantiles.
- **Isolation Guarantees:**
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Calib} = \emptyset, \quad \text{Val} \cap \text{Calib} = \emptyset$$
  $$(\text{Train} \cup \text{Val} \cup \text{Calib}) \cap \text{Test} = \emptyset$$
- Partitions can be saved to disk (`splits/train_ids.txt`, `splits/val_ids.txt`, `splits/calib_ids.txt`, `splits/test_ids.txt`).

---

## 5. Concept Selection Audit & Methodology (`src/data/concept_audit.py`)

The proposal requires **112 filtered concepts** derived from the 312 raw attributes. Because the proposal does not explicitly enumerate the 112 attribute IDs in text, the pipeline implements an auditable selection engine rather than hardcoding arbitrary IDs.

### Candidate Counting Rules
To determine if an attribute occurs in a bird species class:
1. **Rule A (`class_majority_50`):** Attribute is present in a species if $\ge 50\%$ of images in that species exhibit the attribute.
2. **Rule B (`class_any_positive`):** Attribute is present in a species if at least 1 image in that species exhibits the attribute.
3. **Rule C (`class_min_images_2`):** Attribute is present in a species if $\ge 2$ images in that species exhibit the attribute.

Once the dataset is extracted:
1. Run `python scripts/inspect_dataset.py` to audit all candidate rules.
2. Evaluate which rule matches the 112 concepts target.
3. Upon review and approval, save the finalized mapping to `data/selected_concepts_112.json`.

### Concept Mapping Format
The mapping file explicitly distinguishes between the model concept index and the original CUB attribute ID:
```json
{
  "metadata": {
    "description": "Mapping from model concept index to original CUB-200-2011 attribute ID",
    "num_concepts": 112,
    "selection_rule": "class_majority_50",
    "min_classes_threshold": 10
  },
  "concepts": [
    {
      "concept_idx": 0,
      "cub_attribute_id": 1,
      "attribute_name": "has_bill_shape::all-purpose"
    }
  ]
}
```
