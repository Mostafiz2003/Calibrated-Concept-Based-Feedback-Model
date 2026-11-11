import json
from pathlib import Path

def patch_notebook():
    nb_path = Path("notebooks/colab_pretraining_verification.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    new_source = [
        "%cd /content\n",
        "# Ensure project dependencies are present\n",
        "!pip install -q pyyaml scikit-learn pillow\n",
        "\n",
        "import os\n",
        "import sys\n",
        "import zipfile\n",
        "from pathlib import Path\n",
        "\n",
        "# 1. Ensure /content is in Python import path\n",
        "if '/content' not in sys.path:\n",
        "    sys.path.insert(0, '/content')\n",
        "\n",
        "# 2. Check if project code (src) is present in /content; if not, unpack project_code.zip\n",
        "if not Path('/content/src').exists():\n",
        "    print(\"[SETUP] 'src' module not found in /content. Looking for project_code.zip...\")\n",
        "    drive_zip = Path('/content/drive/MyDrive/CalibratedBirdRecognition/project_code.zip')\n",
        "    local_zip = Path('/content/project_code.zip')\n",
        "    \n",
        "    if drive_zip.exists():\n",
        "        print(f\"Found {drive_zip} on Google Drive! Extracting...\")\n",
        "        with zipfile.ZipFile(drive_zip, 'r') as z:\n",
        "            z.extractall('/content')\n",
        "        print(\"[PASS] Codebase extracted from Google Drive to /content\")\n",
        "    elif local_zip.exists():\n",
        "        print(f\"Found {local_zip} in /content! Extracting...\")\n",
        "        with zipfile.ZipFile(local_zip, 'r') as z:\n",
        "            z.extractall('/content')\n",
        "        print(\"[PASS] Codebase extracted to /content\")\n",
        "    else:\n",
        "        print('\\n' + '=' * 75)\n",
        "        print(\"ACTION REQUIRED: Upload 'project_code.zip' (from your PC, ~124 KB)\")\n",
        "        print('=' * 75)\n",
        "        print(\"Option A: Upload project_code.zip to Google Drive (MyDrive/CalibratedBirdRecognition/)\")\n",
        "        print(\"Option B: Select and upload project_code.zip right now below:\")\n",
        "        try:\n",
        "            from google.colab import files\n",
        "            uploaded = files.upload()\n",
        "            for fn in uploaded.keys():\n",
        "                if fn.endswith('.zip'):\n",
        "                    with zipfile.ZipFile(fn, 'r') as z:\n",
        "                        z.extractall('/content')\n",
        "                    print(f\"[PASS] Successfully extracted {fn} to /content\")\n",
        "        except Exception as e:\n",
        "            print(f\"Upload prompt skipped or error: {e}\")\n",
        "\n",
        "assert Path('/content/src').exists(), (\n",
        "    \"CRITICAL ERROR: 'src' folder still not found! Please place 'project_code.zip' \"\n",
        "    \"in /content or in Google Drive under MyDrive/CalibratedBirdRecognition/ and rerun this cell.\"\n",
        ")\n",
        "\n",
        "from src.cloud.dataset_sync import DatasetStagingManager\n",
        "from src.data.partition_manager import PartitionManager\n",
        "\n",
        "print('=' * 75)\n",
        "print('           STAGE 3: DATASET STAGING & PARTITION ISOLATION AUDIT')\n",
        "print('=' * 75)\n",
        "\n",
        "# Stage dataset from Drive to local SSD (if archive exists)\n",
        "try:\n",
        "    staging_mgr = DatasetStagingManager(\n",
        "        drive_source_dir=DRIVE_ROOT / 'dataset/CUB_200_2011',\n",
        "        local_staging_dir='/content/data/CUB_200_2011',\n",
        "    )\n",
        "    res = staging_mgr.stage_dataset()\n",
        "    print(f\"Dataset Staging Result: {res['status']} at {res['path']}\")\n",
        "except FileNotFoundError as e:\n",
        "    print(f\"[INFO] Dataset archive not on Drive yet: {e}\")\n",
        "    print(\"[INFO] Proceeding with manifest audit & synthetic architecture forward passes.\")\n",
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

    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and any("%cd /content" in s for s in cell["source"]):
            cell["source"] = new_source
            break

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook patched successfully!")

if __name__ == "__main__":
    patch_notebook()

