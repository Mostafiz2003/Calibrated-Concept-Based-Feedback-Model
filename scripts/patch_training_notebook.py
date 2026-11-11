import json
from pathlib import Path

def patch_training_notebook():
    nb_path = Path("notebooks/colab_training_pipeline.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    new_step3_source = [
        "%cd /content\n",
        "# Install essential requirements\n",
        "!pip install -q pyyaml scikit-learn pillow\n",
        "\n",
        "import os\n",
        "import sys\n",
        "import zipfile\n",
        "from pathlib import Path\n",
        "\n",
        "# Ensure /content is on Python import path\n",
        "if '/content' not in sys.path:\n",
        "    sys.path.insert(0, '/content')\n",
        "\n",
        "# Check if project code (src) is present in /content; if not, unpack project_code.zip\n",
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
        "    \"CRITICAL ERROR: 'src' folder still not found! Please upload 'project_code.zip' \"\n",
        "    \"to '/content' or place it in Google Drive under 'MyDrive/CalibratedBirdRecognition/'.\"\n",
        ")\n",
        "print(\"[PASS] Project codebase verified at /content\")\n"
    ]

    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and any("!pip install -q pyyaml" in s for s in cell["source"]):
            cell["source"] = new_step3_source
            break

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Training notebook patched successfully!")

if __name__ == "__main__":
    patch_training_notebook()

