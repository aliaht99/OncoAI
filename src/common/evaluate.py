#!/usr/bin/env python3
"""
Evaluate a trained OncoAI task and write publication-style figures.

    python src/common/evaluate.py brain
    python src/common/evaluate.py lung --gradcam 6

Produces, under results/<task>/:
    confusion_matrix.png     counts + row-normalised recall
    roc_curves.png           per-class one-vs-rest ROC
    per_class_metrics.csv    sensitivity / specificity / precision / F1 / AUC
    gradcam.png              what the network actually looked at
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data as data_mod                     # noqa: E402
from common.model import get_device, load_checkpoint    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalise(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


# ── metrics ─────────────────────────────────────────────────────────────────
def per_class_table(targets, probs, class_names) -> pd.DataFrame:
    preds = probs.argmax(1)
    cm = confusion_matrix(targets, preds, labels=range(len(class_names)))
    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if tp + fn else np.nan          # recall
        spec = tn / (tn + fp) if tn + fp else np.nan
        prec = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * prec * sens / (prec + sens) if prec and sens else np.nan
        try:
            auc = roc_auc_score((targets == i).astype(int), probs[:, i])
        except ValueError:
            auc = np.nan
        rows.append({"class": name, "n_test": int(cm[i].sum()),
                     "sensitivity": sens, "specificity": spec,
                     "precision": prec, "f1": f1, "auc": auc})
    return pd.DataFrame(rows)


# ── figures ─────────────────────────────────────────────────────────────────
def plot_confusion(targets, preds, class_names, out: Path, task: str):
    cm = confusion_matrix(targets, preds, labels=range(len(class_names)))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(4.6 + 1.5 * len(class_names), 5))
    for ax, mat, title, fmt in (
        (axes[0], cm, "Counts", "d"),
        (axes[1], norm, "Row-normalised (recall)", ".2f"),
    ):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() if fmt == "d" else 1)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontweight="bold", fontsize=10)
        thresh = (mat.max() if fmt == "d" else 1) * 0.6
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        fontsize=8, color="white" if mat[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"OncoAI — {task} confusion matrix", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(targets, probs, class_names, out: Path, task: str):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for i, name in enumerate(class_names):
        y = (targets == i).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        fpr, tpr, _ = roc_curve(y, probs[:, i])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc_score(y, probs[:, i]):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"OncoAI — {task} ROC (one-vs-rest)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ── GradCAM ─────────────────────────────────────────────────────────────────
def gradcam(model, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
    """Vanilla Grad-CAM on the last conv block."""
    layer = model.target_layer()
    acts, grads = {}, {}

    # Hooks must return None — returning a value makes PyTorch treat it as a
    # replacement activation/gradient, which then fails a shape check.
    def fwd_hook(_m, _inp, out):
        acts["v"] = out

    def bwd_hook(_m, _grad_in, grad_out):
        grads["v"] = grad_out[0]

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad()
        logits = model(x)
        idx = int(logits.argmax(1)) if class_idx is None else class_idx
        logits[0, idx].backward()

        a = acts["v"][0]                       # C,H,W
        g = grads["v"][0]
        weights = g.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * a).sum(0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=x.shape[-2:],
                            mode="bilinear", align_corners=False)[0, 0]
        return cam.detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()


def plot_gradcam(model, ds, class_names, device, out: Path, task: str, n: int = 6):
    # Spread the panel across classes rather than showing six of the same thing.
    from collections import defaultdict
    by_class = defaultdict(list)
    for i in range(len(ds)):
        _, lab = ds[i] if not hasattr(ds, "items") else (None, ds.items[i][1])
        by_class[int(lab)].append(i)
        if sum(len(v) for v in by_class.values()) > 400:
            break

    picks: list[int] = []
    while len(picks) < n and any(by_class.values()):
        for c in sorted(by_class):
            if by_class[c] and len(picks) < n:
                picks.append(by_class[c].pop(0))

    cols = min(len(picks), 3)
    rows = int(np.ceil(len(picks) / cols)) * 2
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    axes = np.atleast_2d(axes)

    for k, idx in enumerate(picks):
        x, y = ds[idx]
        xb = x.unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(xb), 1)[0].cpu().numpy()
        cam = gradcam(model, xb)
        pred = int(prob.argmax())

        r, c = (k // cols) * 2, k % cols
        img = denormalise(x)
        axes[r, c].imshow(img)
        axes[r, c].set_title(f"true: {class_names[y]}", fontsize=9)
        axes[r, c].axis("off")

        axes[r + 1, c].imshow(img)
        axes[r + 1, c].imshow(cam, cmap="jet", alpha=0.45)
        ok = "✓" if pred == y else "✗"
        axes[r + 1, c].set_title(f"{ok} pred: {class_names[pred]} ({prob[pred]:.0%})",
                                 fontsize=9)
        axes[r + 1, c].axis("off")

    for ax in axes.flat:
        if not ax.has_data():
            ax.axis("off")
    fig.suptitle(f"OncoAI — {task}: Grad-CAM (what the model looked at)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["brain", "lung", "skin"])
    ap.add_argument("--gradcam", type=int, default=6, help="0 to skip")
    a = ap.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(MODELS / f"{a.task}_model.pth", device)
    class_names = ckpt["class_names"]
    image_size = ckpt["meta"].get("image_size", 224)

    _, _, test_ds, _ = data_mod.build(a.task, image_size=image_size)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    probs, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.softmax(model(xb.to(device)), 1).cpu().numpy())
            targets.append(yb.numpy())
    probs = np.concatenate(probs)
    targets = np.concatenate(targets)
    preds = probs.argmax(1)

    out_dir = RESULTS / a.task
    out_dir.mkdir(parents=True, exist_ok=True)

    table = per_class_table(targets, probs, class_names)
    table.to_csv(out_dir / "per_class_metrics.csv", index=False)
    print(f"\n{a.task} — per-class metrics")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    acc = float((preds == targets).mean())
    macro_auc = float(np.nanmean(table["auc"]))
    print(f"\noverall accuracy = {acc:.4f}   macro AUC = {macro_auc:.4f}")

    plot_confusion(targets, preds, class_names, out_dir / "confusion_matrix.png", a.task)
    plot_roc(targets, probs, class_names, out_dir / "roc_curves.png", a.task)
    if a.gradcam:
        plot_gradcam(model, test_ds, class_names, device,
                     out_dir / "gradcam.png", a.task, n=a.gradcam)

    summary = {"task": a.task, "accuracy": acc, "macro_auc": macro_auc,
               "classes": class_names,
               "per_class": table.to_dict(orient="records")}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"figures written to results/{a.task}/")


if __name__ == "__main__":
    main()
