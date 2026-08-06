#!/usr/bin/env python3
"""
Figures for the modality-mismatch study.

    python src/novel/make_figures.py

Reads what the audit and gate evaluation wrote and produces:
    fig1_silent_failure_matrix.png   who fails on whose data, and how badly
    fig2_confidence_distributions.png native vs foreign confidence overlap
    fig3_gate_effect.png              silent failure and accuracy, before/after
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "novel"

BLUE, RED, GREY = "#0e7490", "#dc2626", "#94a3b8"


def fig_silent_failure_matrix():
    df = pd.read_csv(OUT / "cross_modality_audit.csv")
    tasks = sorted(set(df.model) | set(df.input_data))
    mat = np.full((len(tasks), len(tasks)), np.nan)
    for _, r in df.iterrows():
        i, j = tasks.index(r.model), tasks.index(r.input_data)
        mat[i, j] = r.malignant_confident_rate

    fig, ax = plt.subplots(figsize=(1.6 * len(tasks) + 3.2, 1.5 * len(tasks) + 2.4))
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=max(0.6, np.nanmax(mat)))
    ax.set_xticks(range(len(tasks)), [t.upper() for t in tasks])
    ax.set_yticks(range(len(tasks)), [t.upper() for t in tasks])
    ax.set_xlabel("Input images come from…", fontweight="bold")
    ax.set_ylabel("Diagnostic model", fontweight="bold")
    ax.set_title("Confident malignant calls\n(diagonal = legitimate use, off-diagonal = silent failure)",
                 fontweight="bold", fontsize=11)

    for i in range(len(tasks)):
        for j in range(len(tasks)):
            if np.isnan(mat[i, j]):
                continue
            native = i == j
            ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if mat[i, j] > 0.35 else "black")
            if native:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor=BLUE, lw=3))
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction ≥ τ confidence")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_silent_failure_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_confidence():
    summary = json.loads((OUT / "audit_summary.json").read_text())
    msp = pd.DataFrame(summary["msp_auroc"])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(msp))
    w = 0.36
    ax.bar(x - w / 2, msp.mean_conf_native, w, label="own modality", color=BLUE)
    ax.bar(x + w / 2, msp.mean_conf_foreign, w, label="foreign modality", color=RED)
    for i, r in msp.iterrows():
        ax.text(i, max(r.mean_conf_native, r.mean_conf_foreign) + 0.03,
                f"MSP AUROC {r.msp_auroc:.2f}", ha="center", fontsize=9,
                fontweight="bold",
                color=RED if r.msp_auroc < 0.6 else "black")
    ax.axhline(0.5, ls="--", lw=1, color=GREY)
    ax.set_xticks(x, [m.upper() for m in msp.model])
    ax.set_ylabel("mean max-softmax confidence")
    ax.set_ylim(0, 1.12)
    ax.set_title("Confidence does not fall on foreign inputs\n"
                 "(AUROC ≈ 0.5 means confidence cannot flag them at all)",
                 fontweight="bold", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_confidence_distributions.png", dpi=160)
    plt.close(fig)


def fig_gate_effect():
    s = json.loads((OUT / "gate_summary.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    ax = axes[0]
    vals = [s["silent_failure_rate_before"], s["silent_failure_rate_after"]]
    worst = [s["worst_sfr_before"], s["worst_sfr_after"]]
    x = np.arange(2)
    ax.bar(x - 0.18, vals, 0.36, label="mean", color=RED)
    ax.bar(x + 0.18, worst, 0.36, label="worst pair", color="#fca5a5")
    for i, (v, w) in enumerate(zip(vals, worst)):
        ax.text(i - 0.18, v + 0.012, f"{v:.1%}", ha="center", fontweight="bold", fontsize=10)
        ax.text(i + 0.18, w + 0.012, f"{w:.1%}", ha="center", fontweight="bold", fontsize=10)
    ax.set_xticks(x, ["no gate", "with gate"])
    ax.set_ylabel("silent failure rate")
    ax.set_title("Confident malignant calls on foreign images",
                 fontweight="bold", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    accs = [s["native_accuracy_before"], s["native_accuracy_after"]]
    ax.bar(["no gate", "with gate"], accs, color=BLUE, width=0.55)
    for i, v in enumerate(accs):
        ax.text(i, v + 0.008, f"{v:.1%}", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy on legitimate inputs")
    ax.set_title(f"Cost of the gate\n({s['native_admitted_rate']:.1%} of valid images still admitted)",
                 fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("The modality gate removes silent failures at almost no accuracy cost",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_gate_effect.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_baselines():
    df = pd.read_csv(OUT / "ood_baselines.csv")
    order = (df.groupby("detector")["auroc"].mean()
               .sort_values(ascending=False).index.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    models = sorted(df.model.unique())
    width = 0.8 / len(models)
    palette = ["#0e7490", "#0891b2", "#67e8f9"]

    for ax, metric, better in ((axes[0], "auroc", "higher is better"),
                               (axes[1], "fpr_at_95tpr", "lower is better")):
        x = np.arange(len(order))
        for k, m in enumerate(models):
            sub = df[df.model == m].set_index("detector").reindex(order)
            ax.bar(x + k * width - 0.4 + width / 2, sub[metric], width,
                   label=m, color=palette[k % len(palette)])
        ax.set_xticks(x, order, rotation=18, ha="right", fontsize=9)
        ax.set_title(f"{'AUROC' if metric == 'auroc' else 'FPR@95TPR'} — {better}",
                     fontweight="bold", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "auroc":
            ax.axhline(0.5, ls="--", lw=1, color=GREY)
            ax.set_ylim(0, 1.05)
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.set_ylim(0, 1.0)

    fig.suptitle("Detecting wrong-modality inputs: the gate vs standard OOD scores",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_ood_baselines.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    if not (OUT / "cross_modality_audit.csv").exists():
        sys.exit("Run src/novel/modality_audit.py first.")
    fig_silent_failure_matrix()
    fig_confidence()
    if (OUT / "gate_summary.json").exists():
        fig_gate_effect()
    if (OUT / "ood_baselines.csv").exists():
        fig_baselines()
    print(f"figures written to results/novel/")


if __name__ == "__main__":
    main()
