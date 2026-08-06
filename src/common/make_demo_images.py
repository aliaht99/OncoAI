#!/usr/bin/env python3
"""
Pull a handful of held-out test images per task into demo_images/ so the app is
usable straight after cloning, without downloading any dataset.

    python src/common/make_demo_images.py

Only a few small images per class are copied, purely so the demo has something
to run on. The datasets themselves stay where they came from — see the links in
README.md.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data as data_mod   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo_images"
PER_CLASS = 2
MAX_SIDE = 512      # keep the repo light


def export(task: str) -> int:
    try:
        _, _, test_ds, class_names = data_mod.build(task)
    except FileNotFoundError as e:
        print(f"  skip {task}: {e}")
        return 0

    out = DEMO / task
    out.mkdir(parents=True, exist_ok=True)
    taken: dict[int, int] = defaultdict(int)
    written = 0

    for i in range(len(test_ds)):
        if all(taken[c] >= PER_CLASS for c in range(len(class_names))):
            break
        # Reach past the transform to keep the original pixels.
        if hasattr(test_ds, "items"):
            path, label = test_ds.items[i]
            if taken[label] >= PER_CLASS:
                continue
            img = Image.open(path).convert("RGB")
        elif hasattr(test_ds, "blobs"):
            import io
            label = test_ds.labels[i]
            if taken[label] >= PER_CLASS:
                continue
            img = Image.open(io.BytesIO(test_ds.blobs[i])).convert("RGB")
        else:
            continue

        img.thumbnail((MAX_SIDE, MAX_SIDE))
        name = f"{class_names[label].replace('/', '-')}_{taken[label] + 1}.png"
        img.save(out / name, optimize=True)
        taken[label] += 1
        written += 1

    print(f"  {task}: wrote {written} images to demo_images/{task}/")
    return written


def main():
    DEMO.mkdir(exist_ok=True)
    total = sum(export(t) for t in ("brain", "lung", "skin"))
    print(f"\nTotal demo images: {total}")


if __name__ == "__main__":
    main()
