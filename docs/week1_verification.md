# Week 1 Verification and Quality Audit Report

**Project:** Calibrated Concept-Based Feedback Model (CSE 4385)  
**Date of Audit:** October 2026  
**Auditor:** Automated Diagnostic Suite & Code Inspection  
**Scope:** Verification of Week 1 Scaffolding, Configuration, Environment, and Proposal Alignment

---

## 1. Executive Summary

This verification audit confirms that Week 1 deliverables strictly comply with the project proposal and lab requirements.
- **Unit Tests:** All 5 automated unit tests pass completely (`5 passed in 0.16s`).
- **Dependencies & Environment:** An isolated virtual environment (`venv`) was created and verified with all required libraries (`torch 2.15.0+cpu`, `torchvision 0.29.2+cpu`, `numpy`, `pandas`, `scikit-learn`, `pyyaml`, `matplotlib`, `seaborn`, `streamlit`, `pytest`).
- **Boundaries Respected:** No neural network has been trained, no dataset downloaded automatically, and no fabricated results or mock metrics exist.
- **Code Health:** No hardcoded absolute paths exist in the configuration or source code. Scripts handle path resolution dynamically.

---

## 2. Alignment with Proposal & Proposal Summary

| Proposal Specification | Status in Codebase | Verification Details |
| :--- | :---: | :--- |
| **Title & Academic Context** | Verified | "Calibrated Concept-Based Feedback Model" for CSE 4385, Northern University Bangladesh. |
| **Dataset Dimensions** | Verified | Config specifies 200 species classes, 312 raw attributes, and 112 filtered concepts (`configs/default.yaml`). |
| **Data Partitioning Protocol** | Verified | Training portion (5,994 images) partitioned into ~80% Train, ~10% Validation, ~10% Calibration. Official test set (5,794 images) remains untouched. |
| **Model Architectures** | Verified | ResNet-50 specified as primary backbone with ResNet-18 fallback; CBM variants (`independent`, `sequential`, `joint`) configured. |
| **Calibration Method** | Verified | Temperature scaling on validation logits to report ECE and NLL. |
| **Conformal Prediction** | Verified | Split conformal prediction with $\alpha \in \{0.05, 0.10, 0.20\}$; non-singleton flagging rule configured. |
| **Intervention Policies** | Verified | $P_1$ (Lowest confidence), $P_2$ (Conformal-flagged), $P_3$ (Random control); replacements $S_1$ (Probability) and $S_2$ (Percentile logit); noise $\epsilon \in \{0.0, 0.05, 0.10, 0.20\}$. |
| **14-Week Timeline** | Verified | Week 1 is strictly restricted to setup, configuration, and diagnostics. Training scheduled for Weeks 3–6. |

---

## 3. Automated Unit Test Verification

The unit test suite was executed in the project virtual environment using `pytest`:

```powershell
.\venv\Scripts\pytest.exe tests/test_config.py -v
```

### Results
```text
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.2.0, pluggy-1.7.0 -- D:\Ai Claude\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Ai Claude
plugins: anyio-4.15.1
collecting ... collected 5 items

tests/test_config.py::TestConfig::test_default_config_loads_successfully PASSED [ 20%]
tests/test_config.py::TestConfig::test_invalid_concept_count_fails_validation PASSED [ 40%]
tests/test_config.py::TestConfig::test_missing_config_raises_informative_error PASSED [ 60%]
tests/test_config.py::TestConfig::test_missing_required_section_fails_validation PASSED [ 80%]
tests/test_config.py::TestConfig::test_proposal_specified_constants PASSED [100%]

============================== 5 passed in 0.16s ==============================
```

All 5 tests passed without warnings or failures.

---

## 4. Verification of Core Scripts & Applications

### 4.1 Configuration Loader (`src/config.py`)
- **Behavior:** Loads `configs/default.yaml`, performs semantic validation against proposal constants (200 species, 112 concepts, 312 attributes), and features a built-in YAML parser fallback when PyYAML is unavailable.
- **Verification Run:** `.\venv\Scripts\python.exe src/config.py` exited with code `0`.

### 4.2 Environment Checker (`scripts/check_environment.py`)
- **Behavior:** Inspects system Python, OS, installed packages, PyTorch, and CUDA acceleration.
- **Verification Run:** `.\venv\Scripts\python.exe scripts/check_environment.py` exited with code `0`.
- **Finding:** Matplotlib cache redirected to `outputs/.matplotlib_cache` to ensure zero permission warnings on Windows. All core scientific packages verified as `[OK]`.

### 4.3 Dataset Inspector (`scripts/inspect_dataset.py`)
- **Behavior:** Non-destructively inspects `data/CUB_200_2011/` for annotation files and directory structure without downloading files.
- **Verification Run:** Correctly outputs that the dataset is not yet present, directs the user to `data/README.md`, and exits with code `0`.

### 4.4 Streamlit Landing Page (`app/streamlit_app.py`)
- **Behavior:**
  - When launched via `streamlit run app/streamlit_app.py`, renders the Week 1 project dashboard.
  - When launched directly via `python app/streamlit_app.py`, cleanly prints launch instructions without raising `ScriptRunContext` warnings.
- **Integrity Check:** Displays an explicit warning banner stating that the model has **not** been trained yet. No mock predictions or simulated metrics are displayed.

---

## 5. Python & PyTorch Compatibility Audit

- **Current Environment:** Python 3.13.3 (64-bit) on Windows 11.
- **Installed Packages:** `torch 2.15.0+cpu` and `torchvision 0.29.2+cpu`.
- **Compatibility Analysis:**
  - The installed PyTorch CPU version works reliably with Python 3.13 on Windows for development, testing, and debugging.
  - **GPU Consideration:** For users who wish to utilize NVIDIA CUDA acceleration on Windows, PyTorch's official CUDA binaries are currently most comprehensively supported on **Python 3.10 through Python 3.12**.
  - **Recommendation:** Since development is currently in CPU mode, Python 3.13 is fully functional. The codebase includes a pre-configured ResNet-18 fallback to ensure fast training on CPU.

---

## 6. Audit of the 112-Concept Selection Rule

- **Proposal Audit:** Proposal Section 5.1 specifies:
  > *"Following the existing proposal, the project uses 112 filtered attributes as concepts... Apply one documented filtering procedure and save the mapping from the retained 112 concepts to original attribute IDs."*
- **Finding:** The proposal does *not* enumerate the 112 attribute IDs inline, nor does it specify the numerical threshold in text.
- **Protocol for Week 2:** To avoid inventing concept IDs or introducing bias:
  1. A reproducible audit script (`scripts/audit_attributes.py`) will be implemented in Week 2.
  2. The script will apply the standard CBM literature filtering criterion (attributes present in at least 10 classes with sufficient frequency, per Koh et al., 2020).
  3. The resulting concept mapping will be serialized to `data/selected_concepts_112.json` and version-controlled.

---

## 7. Data Partitioning & Test-Set Isolation Audit

- **Official Test Set (5,794 images):** Completely isolated. No scripts in Week 1 read, modify, or split test set files.
- **Calibration Split (~599 images):** Separated from the training split (~4,795 images) and validation split (~600 images).
- **Validation Split (~600 images):** Reserved strictly for model selection and fitting temperature scaling $T$.
- **Finding:** Partitioning configuration completely satisfies Section 5.1 and Section 6.1 (test set lock).

---

## 8. Path & Code Quality Audit

1. **Hardcoded Paths:**
   - `configs/default.yaml` uses clean relative paths (`data/CUB_200_2011`, `outputs/checkpoints`).
   - Python files resolve paths dynamically using `Path(__file__).resolve().parent.parent`.
   - No absolute paths (such as `C:\...` or `D:\...`) exist in configuration or code.
2. **Terminal Working Directory Guidance:**
   - PowerShell commands must be run from `d:\Ai Claude` so relative paths resolve cleanly.

---

## 9. Action Items Before Week 2

1. Review and approve this verification audit report.
2. Ensure you can launch the Streamlit prototype (`streamlit run app/streamlit_app.py` from `d:\Ai Claude`).
3. (Optional) Download `CUB_200_2011.tgz` according to `data/README.md` when preparing for Week 2 dataset auditing.

