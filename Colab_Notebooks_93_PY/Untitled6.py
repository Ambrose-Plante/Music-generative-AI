from google.colab import drive
drive.mount('/content/drive')

import os

extension = ".ipynb"

for root, dirs, filenames in os.walk('/content/drive/MyDrive'):
    for filename in filenames:
        if filename.lower().endswith(extension.lower()):
            print(os.path.join(root, filename))

from pathlib import Path
import zipfile
from google.colab import drive
from google.colab import files as colab_files


# Root of your Google Drive
drive_path = Path('/content/drive/MyDrive')

# Extensions to collect
extensions = {'.py', '.ipynb', '.bash', '.sh'}

# Find all matching files
matched_files = [
    f for f in drive_path.rglob('*')
    if f.is_file() and f.suffix.lower() in extensions
]

print(f"Found {len(matched_files)} files")

# Create ZIP while preserving folder structure
zip_path = "/content/google_drive_code_files.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file in matched_files:
        relative_path = file.relative_to(drive_path)
        zipf.write(file, arcname=relative_path)

print(f"Created: {zip_path}")

# Download ZIP
colab_files.download(zip_path)
