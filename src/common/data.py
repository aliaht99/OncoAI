"""
Dataset builders for the three OncoAI tasks.

Each builder returns (train_ds, val_ds, test_ds, class_names). The three public
sources are shaped very differently — a flat two-folder set, a pre-split folder
tree, and HF parquet shards — so the per-task quirks are handled here and the
training loop stays generic.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_transforms(image_size: int):
    """Light augmentation for training; deterministic resize for eval.

    Flips are safe on all three modalities (no laterality label is predicted).
    Rotation and colour jitter stay mild so lesions keep their diagnostic look.
    """
    train_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfm, eval_tfm


class ListDataset(Dataset):
    """(path, label) pairs — used by the folder-based tasks."""

    def __init__(self, items: list[tuple[Path, int]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class BytesDataset(Dataset):
    """In-memory image bytes + labels — used by the parquet-based skin task."""

    def __init__(self, blobs: list[bytes], labels: list[int], transform=None):
        self.blobs = blobs
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.blobs)

    def __getitem__(self, idx: int):
        img = Image.open(io.BytesIO(self.blobs[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def _stratified_split(items: list[tuple[Path, int]], n_classes: int,
                      val_frac: float, test_frac: float, seed: int = 42):
    """Split per class so every class keeps its proportion in all three sets."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in range(n_classes):
        cls = [it for it in items if it[1] == c]
        idx = rng.permutation(len(cls))
        n_val = max(1, int(round(len(cls) * val_frac)))
        n_test = max(1, int(round(len(cls) * test_frac)))
        for j, i in enumerate(idx):
            if j < n_test:
                test.append(cls[i])
            elif j < n_test + n_val:
                val.append(cls[i])
            else:
                train.append(cls[i])
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# BRAIN — MRI, tumour vs no tumour. One flat folder pair, so we split it here.
# ─────────────────────────────────────────────────────────────────────────────
def build_brain(image_size: int = 224, seed: int = 42):
    base = DATA / "brain" / "Brain MRI Images for Brain Tumor Detection"
    if not base.exists():
        raise FileNotFoundError(f"Missing {base} — run src/common/download_data.py brain")

    class_names = ["no_tumor", "tumor"]
    items: list[tuple[Path, int]] = []
    seen: set[str] = set()
    for label, folder in ((0, "no"), (1, "yes")):
        for p in _images_in(base / folder):
            # The archive ships the same images twice (a nested duplicate copy);
            # de-duplicate on filename so the split cannot leak across sets.
            key = f"{label}/{p.name.lower()}"
            if key in seen:
                continue
            seen.add(key)
            items.append((p, label))

    train, val, test = _stratified_split(items, 2, val_frac=0.15, test_frac=0.15, seed=seed)
    train_tfm, eval_tfm = build_transforms(image_size)
    return (ListDataset(train, train_tfm),
            ListDataset(val, eval_tfm),
            ListDataset(test, eval_tfm),
            class_names)


# ─────────────────────────────────────────────────────────────────────────────
# LUNG — chest CT, 4 classes. Ships pre-split, but the train and test folders
# spell the classes differently (train carries TNM staging in the folder name),
# so names are normalised to a common set before mapping to labels.
# ─────────────────────────────────────────────────────────────────────────────
LUNG_CLASSES = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]


def _normalise_lung(folder_name: str) -> str | None:
    name = folder_name.lower()
    for c in LUNG_CLASSES:
        if name.startswith(c):
            return c
    # e.g. "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa"
    stem = re.split(r"_", name)[0]
    return stem if stem in LUNG_CLASSES else None


def _lung_split(split_dir: Path) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for sub in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        cls = _normalise_lung(sub.name)
        if cls is None:
            continue
        label = LUNG_CLASSES.index(cls)
        items += [(p, label) for p in _images_in(sub)]
    return items


def build_lung(image_size: int = 224, seed: int = 42):
    base = DATA / "lung" / "Data"
    if not base.exists():
        raise FileNotFoundError(f"Missing {base} — run src/common/download_data.py lung")

    train = _lung_split(base / "train")
    val = _lung_split(base / "valid")
    test = _lung_split(base / "test")
    train_tfm, eval_tfm = build_transforms(image_size)
    return (ListDataset(train, train_tfm),
            ListDataset(val, eval_tfm),
            ListDataset(test, eval_tfm),
            list(LUNG_CLASSES))


# ─────────────────────────────────────────────────────────────────────────────
# SKIN — HAM10000 as HF parquet shards (image bytes + dx label).
# ─────────────────────────────────────────────────────────────────────────────
def build_skin(image_size: int = 224, seed: int = 42, max_per_class: int | None = None,
               group_by_lesion: bool = True):
    """HAM10000.

    IMPORTANT: the splits shipped with this mirror are image-level, and HAM10000
    contains several images of the same physical lesion. 72% of the lesions in
    the shipped test split also appear in its train split, which inflates test
    accuracy by a large margin. With ``group_by_lesion`` (the default) we discard
    the shipped splits and re-split on ``lesion_id``, so every image of a lesion
    lands in exactly one set. Pass ``group_by_lesion=False`` to reproduce the
    leaky numbers.
    """
    import pandas as pd

    base = DATA / "skin" / "data"
    if not base.exists():
        raise FileNotFoundError(f"Missing {base} — run src/common/download_data.py skin")

    shards = sorted(base.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards in {base}")

    frames = {"train": [], "test": [], "validation": []}
    for s in shards:
        split = ("test" if "test" in s.name else
                 "validation" if "valid" in s.name else "train")
        frames[split].append(pd.read_parquet(s))

    def concat(key):
        return pd.concat(frames[key], ignore_index=True) if frames[key] else None

    df_train, df_test, df_val = concat("train"), concat("test"), concat("validation")

    label_col = "dx"
    image_col = "image"

    if group_by_lesion:
        pool = pd.concat([d for d in (df_train, df_val, df_test) if d is not None],
                         ignore_index=True)
        if "lesion_id" in pool.columns:
            # One row per lesion decides that lesion's split; stratify on the
            # lesion's label so rare classes survive in every set.
            lesions = (pool[["lesion_id", label_col]]
                       .drop_duplicates("lesion_id")
                       .reset_index(drop=True))
            rng = np.random.default_rng(seed)
            assign: dict[str, str] = {}
            for cls, grp in lesions.groupby(label_col):
                ids = grp["lesion_id"].to_numpy()
                rng.shuffle(ids)
                n_test = max(1, int(round(0.15 * len(ids))))
                n_val = max(1, int(round(0.15 * len(ids))))
                for i, lid in enumerate(ids):
                    assign[lid] = ("test" if i < n_test else
                                   "validation" if i < n_test + n_val else "train")
            part = pool["lesion_id"].map(assign)
            df_train = pool[part == "train"].reset_index(drop=True)
            df_val = pool[part == "validation"].reset_index(drop=True)
            df_test = pool[part == "test"].reset_index(drop=True)

    class_names = sorted(df_train[label_col].astype(str).unique())

    index_of = {c: i for i, c in enumerate(class_names)}

    def to_ds(df, tfm, cap: int | None):
        if df is None:
            return None
        if cap:
            df = (df.groupby(label_col, group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), cap), random_state=seed)))
        # Column-wise extraction: iterrows() builds a Series per row, which copies
        # every image blob and turns loading 10k images into minutes of overhead.
        cells = df[image_col].to_numpy()
        blobs = [c["bytes"] if isinstance(c, dict) else c for c in cells]
        labels = [index_of[str(v)] for v in df[label_col].to_numpy()]
        return BytesDataset(blobs, labels, tfm)

    train_tfm, eval_tfm = build_transforms(image_size)
    train_ds = to_ds(df_train, train_tfm, max_per_class)
    val_ds = to_ds(df_val, eval_tfm, None)
    test_ds = to_ds(df_test, eval_tfm, None)

    # Some mirrors of this dataset ship no validation shard — carve one out of
    # train so early stopping still has an honest signal.
    if val_ds is None:
        n_val = max(1, int(0.1 * len(train_ds)))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(train_ds), generator=g).tolist()
        val_ds = Subset(train_ds, perm[:n_val])
        train_ds = Subset(train_ds, perm[n_val:])

    return train_ds, val_ds, test_ds, class_names


BUILDERS = {"brain": build_brain, "lung": build_lung, "skin": build_skin}


def build(task: str, **kw):
    if task not in BUILDERS:
        raise KeyError(f"Unknown task {task!r}; choose from {list(BUILDERS)}")
    return BUILDERS[task](**kw)
