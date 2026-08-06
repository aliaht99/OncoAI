#!/usr/bin/env python3
"""
The Modality Gate — a guard that runs *before* any diagnostic model.

The audit (modality_audit.py) shows that a cancer classifier handed an image
from the wrong modality does not abstain: it emits a confident disease label,
and its own softmax confidence is useless — sometimes worse than useless — as a
warning signal.

The gate is deliberately boring: one small classifier trained to answer "which
modality is this?", plus a confidence floor so inputs that look like none of the
known modalities are rejected rather than forced into the nearest bucket. It is
cheap (a single extra forward pass), needs no change to the diagnostic models,
and turns a silent wrong answer into an explicit refusal.

    python src/novel/modality_gate.py train
    python src/novel/modality_gate.py evaluate --tau 0.9
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data as data_mod                                  # noqa: E402
from common.model import OncoNet, get_device, load_checkpoint        # noqa: E402
from novel.modality_audit import MALIGNANT, available_tasks          # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
OUT = ROOT / "results" / "novel"
GATE_PATH = MODELS / "modality_gate.pth"


class RelabelDataset(Dataset):
    """Wrap a task dataset so its label becomes the modality id."""

    def __init__(self, base: Dataset, modality_id: int):
        self.base = base
        self.modality_id = modality_id

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, self.modality_id


def build_modality_sets(tasks: list[str], image_size: int = 224):
    train_parts, val_parts, test_parts = [], [], []
    for mid, t in enumerate(tasks):
        tr, va, te, _ = data_mod.build(t, image_size=image_size)
        train_parts.append(RelabelDataset(tr, mid))
        val_parts.append(RelabelDataset(va, mid))
        test_parts.append(RelabelDataset(te, mid))
    return (ConcatDataset(train_parts), ConcatDataset(val_parts),
            ConcatDataset(test_parts))


def train_gate(tasks: list[str], epochs: int, batch_size: int, lr: float,
               image_size: int, seed: int = 42):
    torch.manual_seed(seed)
    device = get_device()
    train_ds, val_ds, _ = build_modality_sets(tasks, image_size)
    print(f"▶ gate over {tasks}: train={len(train_ds)} val={len(val_ds)}")

    model = OncoNet(len(tasks), backbone="resnet18").to(device)
    model.freeze_backbone(True)      # the modalities are visually far apart;
    # a linear probe on frozen ImageNet features is enough and keeps the gate cheap.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best, best_state = -1.0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in vl:
                pred = model(xb.to(device)).argmax(1).cpu()
                correct += int((pred == yb).sum())
                total += len(yb)
        acc = correct / max(total, 1)
        print(f"  [epoch {ep}/{epochs}] val modality accuracy = {acc:.4f}")
        if acc > best:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "modalities": tasks,
                "image_size": image_size, "val_accuracy": best,
                "backbone": "resnet18"}, GATE_PATH)
    print(f"✓ gate saved ({time.time() - t0:.0f}s), best val accuracy {best:.4f}")
    return model, tasks


def load_gate(device=None):
    device = device or get_device()
    ck = torch.load(GATE_PATH, map_location=device, weights_only=False)
    model = OncoNet(len(ck["modalities"]), backbone=ck.get("backbone", "resnet18"),
                    pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


@torch.no_grad()
def gate_scores(gate, ds, device, batch_size=32, limit=None):
    if limit and limit < len(ds):
        idx = np.random.default_rng(0).choice(len(ds), limit, replace=False)
        ds = torch.utils.data.Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    for xb, _ in loader:
        probs.append(torch.softmax(gate(xb.to(device)), 1).cpu().numpy())
    return np.concatenate(probs)


@torch.no_grad()
def diag_scores(model, ds, device, batch_size=32, limit=None):
    if limit and limit < len(ds):
        idx = np.random.default_rng(0).choice(len(ds), limit, replace=False)
        ds = torch.utils.data.Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, targets = [], []
    for xb, yb in loader:
        probs.append(torch.softmax(model(xb.to(device)), 1).cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def evaluate_gate(tasks: list[str], tau: float, gate_tau: float, limit: int):
    """Compare silent-failure rate and native accuracy with and without the gate."""
    device = get_device()
    gate, gk = load_gate(device)
    modalities = gk["modalities"]

    diag, class_names = {}, {}
    for t in tasks:
        m, ck = load_checkpoint(MODELS / f"{t}_model.pth", device)
        diag[t] = m
        class_names[t] = ck["class_names"]

    test_sets = {t: data_mod.build(t)[2] for t in tasks}

    rows = []
    for m_task in tasks:
        mal_idx = {i for i, c in enumerate(class_names[m_task])
                   if c in MALIGNANT[m_task]}
        for d_task in tasks:
            ds = test_sets[d_task]
            dprob, dtarg = diag_scores(diag[m_task], ds, device, limit=limit)
            gprob = gate_scores(gate, ds, device, limit=limit)

            maxp = dprob.max(1)
            preds = dprob.argmax(1)

            # The gate admits an image only if it names this exact modality and
            # is itself confident about that call.
            g_pred = gprob.argmax(1)
            g_conf = gprob.max(1)
            admitted = (g_pred == modalities.index(m_task)) & (g_conf >= gate_tau)

            native = m_task == d_task
            confident_malignant = (maxp >= tau) & np.isin(preds, list(mal_idx))

            row = {
                "model": m_task, "input_data": d_task, "native": native,
                "n": int(len(maxp)),
                "admitted_rate": float(admitted.mean()),
                "sfr_before": float(confident_malignant.mean()),
                "sfr_after": float((confident_malignant & admitted).mean()),
            }
            if native:
                correct = preds == dtarg
                row["accuracy_before"] = float(correct.mean())
                row["accuracy_after_on_admitted"] = (
                    float(correct[admitted].mean()) if admitted.any() else float("nan"))
                row["retained_fraction"] = float(admitted.mean())
            rows.append(row)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "gate_evaluation.csv", index=False)

    foreign = df[~df.native]
    native = df[df.native]
    summary = {
        "gate_val_accuracy": gk["val_accuracy"],
        "tau": tau, "gate_tau": gate_tau,
        "silent_failure_rate_before": float(foreign["sfr_before"].mean()),
        "silent_failure_rate_after": float(foreign["sfr_after"].mean()),
        "worst_sfr_before": float(foreign["sfr_before"].max()),
        "worst_sfr_after": float(foreign["sfr_after"].max()),
        "foreign_admitted_rate": float(foreign["admitted_rate"].mean()),
        "native_admitted_rate": float(native["admitted_rate"].mean()),
        "native_accuracy_before": float(native["accuracy_before"].mean()),
        "native_accuracy_after": float(native["accuracy_after_on_admitted"].mean()),
    }
    with open(OUT / "gate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── per-pair ──")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n── headline ──")
    for k, v in summary.items():
        print(f"  {k:32} {v:.4f}" if isinstance(v, float) else f"  {k:32} {v}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "evaluate"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--gate-tau", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()

    tasks = available_tasks()
    if len(tasks) < 2:
        sys.exit("Need at least two trained task models.")

    if a.mode == "train":
        train_gate(tasks, a.epochs, a.batch_size, a.lr, a.image_size)
    else:
        evaluate_gate(tasks, a.tau, a.gate_tau, a.limit)


if __name__ == "__main__":
    main()
