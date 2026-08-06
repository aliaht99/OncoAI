#!/usr/bin/env python3
"""
Fetch the three public datasets OncoAI trains on.

All three are public on the Hugging Face Hub, so no account or API key is
needed. Everything lands in data/ (git-ignored) and can be re-fetched at any
time — nothing here is redistributed by this repo.

    python src/common/download_data.py            # all three
    python src/common/download_data.py brain lung # a subset
"""
import sys
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

SOURCES = {
    # key: (hf repo id, human description)
    "skin": ("marmal88/skin_cancer",
             "HAM10000 — 10,015 dermatoscopic images, 7 lesion classes (~3.7 GB)"),
    "lung": ("dorsar/lung-cancer",
             "Chest CT — 4 classes: adenocarcinoma, large cell, squamous cell, normal (~100 MB)"),
    "brain": ("miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection",
              "Brain MRI — tumour vs no tumour (~16 MB)"),
}


def fetch(key: str) -> Path:
    repo_id, desc = SOURCES[key]
    dest = DATA / key
    print(f"\n▶ {key}: {desc}")
    print(f"  source: https://huggingface.co/datasets/{repo_id}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        # resume-friendly: re-running skips files already present
    )
    # Some sources ship a single zip — unpack it so training sees plain folders.
    for z in dest.rglob("*.zip"):
        out = z.with_suffix("")
        if out.exists():
            continue
        print(f"  unzipping {z.name} …")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
    print(f"  ✓ ready at {dest.relative_to(ROOT)}")
    return dest


def main():
    keys = [k for k in sys.argv[1:] if k in SOURCES] or list(SOURCES)
    DATA.mkdir(parents=True, exist_ok=True)
    for k in keys:
        fetch(k)
    print("\nAll requested datasets are in place.")


if __name__ == "__main__":
    main()
