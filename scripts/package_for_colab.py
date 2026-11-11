"""
Package the project code and configuration for execution on Google Colab.
Creates a lightweight zip archive (project_code.zip) containing:
  - src/
  - configs/
  - scripts/
  - outputs/dataset_manifest.json
  - data/selected_concepts_112.json
"""

import os
from pathlib import Path
import zipfile

def package_code(output_path: str = "project_code.zip") -> Path:
    out_file = Path(output_path)
    include_dirs = ["src", "configs", "scripts"]
    include_files = [
        "outputs/dataset_manifest.json",
        "data/selected_concepts_112.json",
    ]

    total_files = 0
    with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        for d in include_dirs:
            dir_path = Path(d)
            if not dir_path.exists():
                continue
            for p in dir_path.rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
                    zipf.write(p, arcname=p.as_posix())
                    total_files += 1

        for f in include_files:
            file_path = Path(f)
            if file_path.exists():
                zipf.write(file_path, arcname=file_path.as_posix())
                total_files += 1

    size_kb = out_file.stat().st_size / 1024
    print(f"Successfully packaged {total_files} files into {out_file.name} ({size_kb:.1f} KB)")
    return out_file

if __name__ == "__main__":
    package_code()

