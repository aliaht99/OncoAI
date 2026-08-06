#!/usr/bin/env python3
"""
Benchmark the modality gate against the standard OOD detection baselines.

The OOD literature for medical imaging has settled on a small set of scoring
functions and two metrics. A guard proposed without comparing to them is not
evaluable, so this script runs all of them on the same data:

  MSP          max softmax probability (Hendrycks & Gimpel)
  Energy       -logsumexp(logits) (Liu et al.)
  Mahalanobis  distance to class-conditional Gaussians fitted on train features
               (Lee et al.)
  kNN          distance to the k-th nearest training feature (Sun et al.)
  Gate (ours)  supervised modality classifier confidence

Metrics: AUROC (higher better) and FPR@95TPR — the fraction of foreign images
still admitted when the threshold is set to admit 95% of legitimate ones. FPR@95
is the operationally meaningful one: it is the rate at which wrong-modality
images reach a diagnostic model in a deployment tuned not to annoy users.

    python src/novel/ood_baselines.py --limit 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data as data_mod                                 # noqa: E402
from common.model import get_device, load_checkpoint                # noqa: E402
from novel.modality_audit import available_tasks                    # noqa: E402
from novel.modality_gate import load_gate                           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
OUT = ROOT / "results" / "novel"


def subsample(ds, limit: int | None, seed: int = 0):
    if limit and limit < len(ds):
        idx = np.random.default_rng(seed).choice(len(ds), limit, replace=False)
        return Subset(ds, idx.tolist())
    return ds


@torch.no_grad()
def extract(model, ds, device, batch_size: int = 32):
    """Return logits and penultimate embeddings for a dataset."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits, embs, labels = [], [], []
    for xb, yb in loader:
        out, emb = model(xb.to(device), return_embedding=True)
        logits.append(out.cpu().numpy())
        embs.append(emb.cpu().numpy())
        labels.append(yb.numpy())
    return (np.concatenate(logits), np.concatenate(embs), np.concatenate(labels))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def msp_score(logits, **_):
    return softmax(logits).max(1)


def energy_score(logits, **_):
    """Negative free energy; higher = more in-distribution."""
    m = logits.max(1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(logits - m).sum(1)))


def fit_mahalanobis(train_emb, train_lab):
    classes = np.unique(train_lab)
    means = {c: train_emb[train_lab == c].mean(0) for c in classes}
    # Shared covariance across classes, as in the original formulation.
    centred = np.concatenate([train_emb[train_lab == c] - means[c] for c in classes])
    cov = np.cov(centred, rowvar=False) + 1e-6 * np.eye(train_emb.shape[1])
    return means, np.linalg.pinv(cov)


def mahalanobis_score(emb, means, prec):
    """Negative minimum Mahalanobis distance; higher = more in-distribution."""
    best = None
    for mu in means.values():
        d = emb - mu
        dist = np.einsum("ij,jk,ik->i", d, prec, d)
        best = dist if best is None else np.minimum(best, dist)
    return -best


def knn_score(emb, train_emb, k: int = 50):
    """Negative distance to the k-th nearest normalised training feature."""
    a = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    b = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-8)
    out = np.empty(len(a))
    step = 256
    for i in range(0, len(a), step):
        d = 1.0 - a[i:i + step] @ b.T
        kk = min(k, d.shape[1] - 1)
        out[i:i + step] = np.partition(d, kk, axis=1)[:, kk]
    return -out


def fpr_at_tpr(id_scores, ood_scores, tpr: float = 0.95) -> float:
    """Fraction of OOD admitted when the threshold admits `tpr` of ID."""
    thr = np.quantile(id_scores, 1 - tpr)      # admit scores >= thr
    return float((ood_scores >= thr).mean())


def evaluate_detector(id_scores, ood_scores):
    y = np.r_[np.ones_like(id_scores), np.zeros_like(ood_scores)]
    s = np.r_[id_scores, ood_scores]
    return {"auroc": float(roc_auc_score(y, s)),
            "fpr_at_95tpr": fpr_at_tpr(id_scores, ood_scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--knn-k", type=int, default=50)
    ap.add_argument("--train-limit", type=int, default=1500,
                    help="training features used to fit Mahalanobis / kNN")
    a = ap.parse_args()

    device = get_device()
    tasks = available_tasks()
    if len(tasks) < 2:
        sys.exit("Need at least two trained models.")
    print(f"▶ OOD baseline benchmark over {tasks}")

    gate, gk, _ = None, None, None
    if (MODELS / "modality_gate.pth").exists():
        gate, gk = load_gate(device)

    # Datasets once.
    built = {t: data_mod.build(t) for t in tasks}
    rows = []

    for t in tasks:
        model, _ = load_checkpoint(MODELS / f"{t}_model.pth", device)
        train_ds, _, test_ds, _ = built[t]

        tr_ds = subsample(train_ds, a.train_limit, seed=1)
        _, tr_emb, tr_lab = extract(model, tr_ds, device)
        means, prec = fit_mahalanobis(tr_emb, tr_lab)

        id_logits, id_emb, _ = extract(model, subsample(test_ds, a.limit), device)
        foreign = [f for f in tasks if f != t]
        ood_logits, ood_emb = [], []
        for f in foreign:
            lg, em, _ = extract(model, subsample(built[f][2], a.limit), device)
            ood_logits.append(lg)
            ood_emb.append(em)
        ood_logits = np.concatenate(ood_logits)
        ood_emb = np.concatenate(ood_emb)

        detectors = {
            "MSP": (msp_score(id_logits), msp_score(ood_logits)),
            "Energy": (energy_score(id_logits), energy_score(ood_logits)),
            "Mahalanobis": (mahalanobis_score(id_emb, means, prec),
                            mahalanobis_score(ood_emb, means, prec)),
            "kNN": (knn_score(id_emb, tr_emb, a.knn_k),
                    knn_score(ood_emb, tr_emb, a.knn_k)),
        }

        if gate is not None and t in gk["modalities"]:
            col = gk["modalities"].index(t)

            @torch.no_grad()
            def gate_conf(ds):
                loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
                out = []
                for xb, _ in loader:
                    out.append(torch.softmax(gate(xb.to(device)), 1)[:, col].cpu().numpy())
                return np.concatenate(out)

            id_g = gate_conf(subsample(test_ds, a.limit))
            ood_g = np.concatenate([gate_conf(subsample(built[f][2], a.limit))
                                    for f in foreign])
            detectors["Gate (ours)"] = (id_g, ood_g)

        for name, (ids, oods) in detectors.items():
            m = evaluate_detector(ids, oods)
            rows.append({"model": t, "detector": name, **m})
            print(f"  {t:6} {name:14} AUROC={m['auroc']:.4f}  "
                  f"FPR@95TPR={m['fpr_at_95tpr']:.4f}")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ood_baselines.csv", index=False)

    print("\n── mean across models ──")
    agg = (df.groupby("detector")[["auroc", "fpr_at_95tpr"]]
             .mean().sort_values("auroc", ascending=False))
    print(agg.to_string(float_format=lambda v: f"{v:.4f}"))

    with open(OUT / "ood_baselines_summary.json", "w") as f:
        json.dump({"per_model": rows,
                   "mean": agg.reset_index().to_dict(orient="records")}, f, indent=2)
    print("\nwritten to results/novel/ood_baselines.csv")


if __name__ == "__main__":
    main()
