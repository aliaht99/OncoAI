#!/usr/bin/env python3
"""
Cross-modality silent-failure audit.

The question this answers: what does a cancer classifier do when it is handed an
image from the wrong modality? A brain-MRI model shown a chest CT, a skin model
shown an MRI slice. In a deployed multi-cancer tool this happens the moment a
user picks the wrong tab — and nothing in a standard pipeline stops it.

We quantify three things per (model, foreign dataset) pair:

  confident rate   fraction of foreign inputs the model scores >= tau on
  SFR              *silent failure rate* — fraction of foreign inputs assigned a
                   MALIGNANT class with confidence >= tau. These are the
                   dangerous ones: a confident cancer call on an image the model
                   has no business interpreting.
  MSP AUROC        how well max-softmax-probability separates in-domain from
                   foreign inputs. At 0.5 confidence is useless as a guard.

    python src/novel/modality_audit.py --tau 0.9
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data as data_mod                     # noqa: E402
from common.model import get_device, load_checkpoint    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
OUT = ROOT / "results" / "novel"

TASKS = ["brain", "lung", "skin"]

# Classes that represent disease. A confident hit on one of these, from an image
# of the wrong modality, is the failure mode we care about.
MALIGNANT = {
    "brain": {"tumor"},
    "lung": {"adenocarcinoma", "large.cell.carcinoma", "squamous.cell.carcinoma"},
    "skin": {"melanoma", "basal_cell_carcinoma", "actinic_keratoses"},
}


def available_tasks() -> list[str]:
    return [t for t in TASKS if (MODELS / f"{t}_model.pth").exists()]


@torch.no_grad()
def score(model, ds, device, batch_size: int = 32, limit: int | None = None):
    """Return (max_prob, pred_idx) for every image in ds."""
    if limit and limit < len(ds):
        idx = np.random.default_rng(0).choice(len(ds), limit, replace=False)
        ds = torch.utils.data.Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    maxp, preds = [], []
    for xb, _ in loader:
        p = torch.softmax(model(xb.to(device)), 1).cpu().numpy()
        maxp.append(p.max(1))
        preds.append(p.argmax(1))
    return np.concatenate(maxp), np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.9,
                    help="confidence threshold that counts as 'confident'")
    ap.add_argument("--limit", type=int, default=400,
                    help="cap images per (model, dataset) pair for runtime")
    a = ap.parse_args()

    device = get_device()
    tasks = available_tasks()
    if len(tasks) < 2:
        sys.exit("Need at least two trained models — train them first.")
    print(f"▶ auditing models: {tasks}  (tau={a.tau})")

    OUT.mkdir(parents=True, exist_ok=True)

    # Load every model and every test set once.
    models, class_names = {}, {}
    for t in tasks:
        m, ck = load_checkpoint(MODELS / f"{t}_model.pth", device)
        models[t] = m
        class_names[t] = ck["class_names"]

    test_sets = {}
    for t in tasks:
        *_, test_ds, _ = data_mod.build(t)
        test_sets[t] = test_ds

    rows = []
    in_domain_scores: dict[str, np.ndarray] = {}
    foreign_scores: dict[str, list[np.ndarray]] = {t: [] for t in tasks}

    for m_task in tasks:
        model = models[m_task]
        mal_idx = {i for i, c in enumerate(class_names[m_task])
                   if c in MALIGNANT[m_task]}
        for d_task in tasks:
            maxp, preds = score(model, test_sets[d_task], device, limit=a.limit)
            native = (m_task == d_task)

            confident = float((maxp >= a.tau).mean())
            malignant_conf = float(
                ((maxp >= a.tau) & np.isin(preds, list(mal_idx))).mean())

            rows.append({
                "model": m_task,
                "input_data": d_task,
                "native": native,
                "n": int(len(maxp)),
                "mean_confidence": float(maxp.mean()),
                "confident_rate": confident,
                "silent_failure_rate": None if native else malignant_conf,
                "malignant_confident_rate": malignant_conf,
            })

            if native:
                in_domain_scores[m_task] = maxp
            else:
                foreign_scores[m_task].append(maxp)

            tag = "native " if native else "FOREIGN"
            print(f"  [{tag}] model={m_task:5} data={d_task:5} "
                  f"mean_conf={maxp.mean():.3f} conf>={a.tau}: {confident:.1%} "
                  f"malignant&confident: {malignant_conf:.1%}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cross_modality_audit.csv", index=False)

    # Can plain softmax confidence be used as a guard? (AUROC 0.5 == useless)
    msp = []
    for t in tasks:
        if not foreign_scores[t]:
            continue
        ind = in_domain_scores[t]
        ood = np.concatenate(foreign_scores[t])
        y = np.r_[np.ones_like(ind), np.zeros_like(ood)]
        s = np.r_[ind, ood]
        msp.append({"model": t, "msp_auroc": float(roc_auc_score(y, s)),
                    "mean_conf_native": float(ind.mean()),
                    "mean_conf_foreign": float(ood.mean())})
    msp_df = pd.DataFrame(msp)
    msp_df.to_csv(OUT / "msp_ood_auroc.csv", index=False)

    print("\n── Silent failure rate (foreign inputs called malignant, confidently) ──")
    piv = (df[~df.native]
           .pivot(index="model", columns="input_data", values="silent_failure_rate"))
    print(piv.to_string(float_format=lambda v: f"{v:.1%}"))

    print("\n── Max-softmax as an OOD guard ──")
    print(msp_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    worst = df[~df.native]["silent_failure_rate"].max()
    summary = {
        "tau": a.tau,
        "tasks": tasks,
        "worst_silent_failure_rate": float(worst),
        "mean_silent_failure_rate": float(df[~df.native]["silent_failure_rate"].mean()),
        "msp_auroc": msp,
        "pairs": rows,
    }
    with open(OUT / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nworst-case silent failure rate: {worst:.1%}")
    print(f"written to results/novel/")


if __name__ == "__main__":
    main()
