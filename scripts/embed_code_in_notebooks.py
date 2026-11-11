"""
Embeds the project code as a self-extracting base64 payload into the Colab notebooks.
This makes the Colab notebooks 100% self-contained so users do not need to manually
upload zip files or clone git repos in Google Colab.
"""

import base64
import json
from pathlib import Path
from package_for_colab import package_code

def embed_code_in_notebooks():
    # 1. Package the latest code
    zip_path = package_code("project_code.zip")
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()
    b64_str = base64.b64encode(zip_bytes).decode("ascii")
    print(f"Base64 payload size: {len(b64_str):,} characters")

    # 2. Patch colab_pretraining_verification.ipynb
    verif_nb_path = Path("notebooks/colab_pretraining_verification.ipynb")
    with open(verif_nb_path, "r", encoding="utf-8") as f:
        v_nb = json.load(f)

    stage3_source = [
        "%cd /content\n",
        "# Ensure project dependencies are present\n",
        "!pip install -q pyyaml scikit-learn pillow\n",
        "\n",
        "import base64\n",
        "import io\n",
        "import os\n",
        "import sys\n",
        "import zipfile\n",
        "from pathlib import Path\n",
        "\n",
        "# 1. Ensure /content is on Python sys.path\n",
        "if '/content' not in sys.path:\n",
        "    sys.path.insert(0, '/content')\n",
        "\n",
        "# 2. Self-Extract Embedded Codebase if src is missing\n",
        "if not (Path('/content/src').exists() and Path('/content/outputs/dataset_manifest.json').exists()):\n",
        "    print('[SETUP] Auto-unpacking embedded research codebase (src, configs, outputs, data)...')\n",
        f"    _EMBEDDED_ZIP_B64 = b'''{b64_str}'''\n",
        "    with zipfile.ZipFile(io.BytesIO(base64.b64decode(_EMBEDDED_ZIP_B64))) as _z:\n",
        "        _z.extractall('/content')\n",
        "    print('[PASS] Research codebase successfully extracted to /content!')\n",
        "\n",
        "from src.cloud.dataset_sync import DatasetStagingManager\n",
        "from src.data.partition_manager import PartitionManager\n",
        "\n",
        "print('=' * 75)\n",
        "print('           STAGE 3: DATASET STAGING & PARTITION ISOLATION AUDIT')\n",
        "print('=' * 75)\n",
        "\n",
        "# Stage dataset from Drive to local SSD (if archive exists on Drive)\n",
        "try:\n",
        "    staging_mgr = DatasetStagingManager(\n",
        "        drive_source_dir=DRIVE_ROOT / 'dataset/CUB_200_2011',\n",
        "        local_staging_dir='/content/data/CUB_200_2011',\n",
        "    )\n",
        "    res = staging_mgr.stage_dataset()\n",
        "    print(f\"Dataset Staging Result: {res['status']} at {res['path']}\")\n",
        "except (FileNotFoundError, RuntimeError) as e:\n",
        "    print(f\"[INFO] CUB-200-2011 raw dataset archive not on Drive yet: {e}\")\n",
        "    print('[INFO] Proceeding with manifest audit & synthetic architecture forward passes (0 raw images needed for hardware verification).')\n",
        "\n",
        "# Verify Partitions and 0% Data Leakage\n",
        "manifest_path = Path('outputs/dataset_manifest.json')\n",
        "if not manifest_path.exists():\n",
        "    manifest_path = DRIVE_ROOT / 'dataset/partitions/dataset_manifest.json'\n",
        "pm = PartitionManager(str(manifest_path))\n",
        "counts = pm.verify_partition_counts()\n",
        "print('Partition Counts:')\n",
        "for split, c in counts['actual_counts'].items():\n",
        "    print(f\"  - {split.upper()}: {c} images (Expected: {pm.EXPECTED_COUNTS[split]})\")\n",
        "assert counts['counts_match'], 'Partition counts mismatch!'\n",
        "\n",
        "iso = pm.verify_isolation()\n",
        "assert iso['is_strictly_isolated'], f\"Data leakage detected! Overlaps: {iso['pairwise_overlaps']}\"\n",
        "print('\\n[PASS] Verified 0% data leakage across all 4 splits!')\n",
        "print('[PASS] Official test split (5,794 images) remains strictly locked.')\n",
    ]

    for cell in v_nb["cells"]:
        if cell["cell_type"] == "code" and any("%cd /content" in s for s in cell["source"]):
            cell["source"] = stage3_source
            break

    with open(verif_nb_path, "w", encoding="utf-8") as f:
        json.dump(v_nb, f, indent=1)
    print(f"Updated {verif_nb_path}")

    # 3. Patch colab_training_pipeline.ipynb
    train_nb_path = Path("notebooks/colab_training_pipeline.ipynb")
    with open(train_nb_path, "r", encoding="utf-8") as f:
        t_nb = json.load(f)

    train_step3_source = [
        "%cd /content\n",
        "# Install essential requirements\n",
        "!pip install -q pyyaml scikit-learn pillow\n",
        "\n",
        "import base64\n",
        "import io\n",
        "import os\n",
        "import sys\n",
        "import zipfile\n",
        "from pathlib import Path\n",
        "\n",
        "# 1. Ensure /content is on Python sys.path\n",
        "if '/content' not in sys.path:\n",
        "    sys.path.insert(0, '/content')\n",
        "\n",
        "# 2. Self-Extract Embedded Codebase if src is missing\n",
        "if not (Path('/content/src').exists() and Path('/content/outputs/dataset_manifest.json').exists()):\n",
        "    print('[SETUP] Auto-unpacking embedded research codebase (src, configs, outputs, data)...')\n",
        f"    _EMBEDDED_ZIP_B64 = b'''{b64_str}'''\n",
        "    with zipfile.ZipFile(io.BytesIO(base64.b64decode(_EMBEDDED_ZIP_B64))) as _z:\n",
        "        _z.extractall('/content')\n",
        "    print('[PASS] Research codebase successfully extracted to /content!')\n",
    ]

    for cell in t_nb["cells"]:
        if cell["cell_type"] == "code" and any("!pip install -q pyyaml" in s for s in cell["source"]):
            cell["source"] = train_step3_source
            break

    with open(train_nb_path, "w", encoding="utf-8") as f:
        json.dump(t_nb, f, indent=1)
    print(f"Updated {train_nb_path}")

if __name__ == "__main__":
    embed_code_in_notebooks()

