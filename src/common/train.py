#!/usr/bin/env python3
"""
Train one OncoAI task.

    python src/common/train.py brain
    python src/common/train.py lung  --epochs 25
    python src/common/train.py skin  --backbone efficientnet_b0

Two-phase transfer learning: a short warm-up with the backbone frozen (so the
randomly-initialised head stops producing garbage gradients), then fine-tuning
of the whole network at a lower LR with cosine annealing and early stopping on
validation macro-AUC.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data as data_mod          # noqa: E402
from common.model import OncoNet, get_device, save_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def labels_of(ds) -> np.ndarray:
    """Pull labels without decoding images (works through Subset too)."""
    from torch.utils.data import Subset
    if isinstance(ds, Subset):
        base = labels_of(ds.dataset)
        return base[np.array(ds.indices)]
    if hasattr(ds, "items"):
        return np.array([lab for _, lab in ds.items])
    if hasattr(ds, "labels"):
        return np.array(ds.labels)
    raise TypeError(f"Cannot read labels from {type(ds).__name__}")


def make_sampler(train_ds, num_classes: int) -> WeightedRandomSampler:
    """Balance classes — every one of these datasets is skewed."""
    y = labels_of(train_ds)
    counts = np.bincount(y, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = (1.0 / counts)[y]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(y), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    probs, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.append(yb.numpy())
    probs = np.concatenate(probs)
    targets = np.concatenate(targets)

    preds = probs.argmax(1)
    acc = float((preds == targets).mean())
    try:
        if num_classes == 2:
            auc = float(roc_auc_score(targets, probs[:, 1]))
        else:
            auc = float(roc_auc_score(targets, probs, multi_class="ovr",
                                      average="macro"))
    except ValueError:
        auc = float("nan")      # a split can miss a class on tiny datasets
    return {"acc": acc, "auc": auc, "probs": probs, "targets": targets}


def run(task: str, epochs_warmup: int, epochs_finetune: int, batch_size: int,
        image_size: int, lr_warmup: float, lr_finetune: float, patience: int,
        backbone: str, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    print(f"▶ task={task}  device={device}  backbone={backbone}")

    train_ds, val_ds, test_ds, class_names = data_mod.build(task, image_size=image_size)
    num_classes = len(class_names)
    print(f"  classes ({num_classes}): {class_names}")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # num_workers=0: MPS does not support pinned memory and fork-based workers
    # deadlock with the in-memory byte datasets.
    common = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, sampler=make_sampler(train_ds, num_classes), **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    model = OncoNet(num_classes, backbone=backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    ckpt_path = MODELS / f"{task}_model.pth"

    best_auc, best_state, bad_epochs = -1.0, None, 0
    history = []
    started = time.time()

    for phase, n_epochs, lr, frozen in (
        ("warmup", epochs_warmup, lr_warmup, True),
        ("finetune", epochs_finetune, lr_finetune, False),
    ):
        if n_epochs <= 0:
            continue
        model.freeze_backbone(frozen)
        params = [p for p in model.parameters() if p.requires_grad]
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)
                     if phase == "finetune" else None)
        print(f"\n── {phase}: {n_epochs} epochs, lr={lr}, backbone {'frozen' if frozen else 'trainable'}")

        for epoch in range(1, n_epochs + 1):
            model.train()
            running, seen = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimiser.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimiser.step()
                running += loss.item() * xb.size(0)
                seen += xb.size(0)
            if scheduler:
                scheduler.step()

            val = evaluate(model, val_loader, device, num_classes)
            train_loss = running / max(seen, 1)
            history.append({"phase": phase, "epoch": epoch, "train_loss": train_loss,
                            "val_acc": val["acc"], "val_auc": val["auc"]})
            flag = ""
            if val["auc"] > best_auc:
                best_auc = val["auc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
                flag = "  ← best"
            else:
                bad_epochs += 1
            print(f"  [{phase} {epoch:2d}/{n_epochs}] loss={train_loss:.4f} "
                  f"val_acc={val['acc']:.4f} val_auc={val['auc']:.4f}{flag}")

            if phase == "finetune" and bad_epochs >= patience:
                print(f"  early stopping (no val AUC gain for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(model, test_loader, device, num_classes)
    elapsed = time.time() - started
    print(f"\n✓ {task}: test acc={test['acc']:.4f}  test AUC={test['auc']:.4f}  "
          f"({elapsed/60:.1f} min)")

    meta = {
        "task": task, "backbone": backbone, "image_size": image_size,
        "classes": class_names, "seed": seed,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "best_val_auc": best_auc,
        "test_accuracy": test["acc"], "test_auc": test["auc"],
        "train_minutes": round(elapsed / 60, 2),
        "history": history,
    }
    save_checkpoint(ckpt_path, model, class_names, meta)
    with open(MODELS / f"{task}_metrics.json", "w") as f:
        json.dump(meta, f, indent=2)
    np.savez(RESULTS / f"{task}_test_predictions.npz",
             probs=test["probs"], targets=test["targets"],
             class_names=np.array(class_names))
    print(f"  saved {ckpt_path.relative_to(ROOT)}")
    return meta


def main():
    ap = argparse.ArgumentParser(description="Train an OncoAI task")
    ap.add_argument("task", choices=["brain", "lung", "skin"])
    ap.add_argument("--epochs-warmup", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20, help="fine-tune epochs")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--lr-warmup", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-4, help="fine-tune LR")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    run(a.task, a.epochs_warmup, a.epochs, a.batch_size, a.image_size,
        a.lr_warmup, a.lr, a.patience, a.backbone, a.seed)


if __name__ == "__main__":
    main()
