#!/usr/bin/env python3
"""
Rule-out triage: turn a classifier into a worklist filter with a *guaranteed*
miss rate.

The gate in `modality_gate.py` stops the wrong kind of image getting a
diagnosis. This module addresses the opposite question, the one a department
actually buys AI to answer:

    "How much of my worklist can this thing take off my hands, and what
     exactly am I promising about the cancers it clears?"

Accuracy does not answer that, and neither does a softmax score — a model that
is 84% accurate and says "0.91" is not telling a radiologist that there is a
91% chance of cancer. So we do two things:

  1. **Calibrate** — temperature scaling on held-out data, so the malignancy
     risk means what it says (measured: ECE and Brier, before and after).

  2. **Bound the miss rate** — pick a rule-out threshold whose miss rate among
     malignant cases is provably below a chosen tolerance, with distribution-
     free finite-sample confidence. Cases scoring below it are auto-cleared;
     everything else goes to a human. No normality assumption, no reliance on
     the calibration being perfect: the bound is a Clopper-Pearson upper limit
     on a binomial proportion, walked over thresholds in a fixed sequence so
     the family-wise error is controlled without a multiplicity correction.

The honest part of the output is the sample-size wall. A guarantee of "at most
5% of cancers missed" cannot be made from 11 malignant calibration cases, no
matter how good the model is, and this script says so instead of printing a
threshold anyway.

    python src/novel/triage.py                 # all three tasks
    python src/novel/triage.py --task skin --alpha 0.05 --delta 0.05

Writes results/novel/triage_<task>.json, triage_summary.csv and figures.
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
from scipy.stats import beta as beta_dist, ks_2samp
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data as data_mod                     # noqa: E402
from common.model import get_device, load_checkpoint    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
NOVEL = RESULTS / "novel"

# Which classes mean "this patient needs a human to look".
# Actinic keratoses are pre-malignant / in-situ (Bowen's) in HAM10000 and are
# treated rather than watched, so they belong on the referral side of the line.
MALIGNANT = {
    "brain": {"tumor"},
    "lung": {"adenocarcinoma", "large.cell.carcinoma", "squamous.cell.carcinoma"},
    "skin": {"actinic_keratoses", "basal_cell_carcinoma", "melanoma"},
}

# Share of the validation split spent on temperature scaling. Temperature is one
# scalar and converges on very little data; the threshold's guarantee is driven
# entirely by how many malignant cases it sees, so it gets the larger share.
TEMP_FRACTION = 0.30


# ── inference ───────────────────────────────────────────────────────────────
def collect_logits(task: str, split: str, force: bool = False):
    """Run the trained model over a split, caching raw logits.

    Logits, not probabilities: temperature scaling needs the pre-softmax scale,
    and results/<task>_test_predictions.npz only stored softmax output.
    """
    cache = NOVEL / f"triage_logits_{task}_{split}.npz"
    if cache.exists() and not force:
        d = np.load(cache, allow_pickle=True)
        return d["logits"], d["targets"], list(d["class_names"])

    device = get_device()
    model, ckpt = load_checkpoint(MODELS / f"{task}_model.pth", device)
    class_names = ckpt["class_names"]
    image_size = ckpt["meta"].get("image_size", 224)

    _, val_ds, test_ds, _ = data_mod.build(task, image_size=image_size)
    ds = {"val": val_ds, "test": test_ds}[split]
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    logits, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits.append(model(xb.to(device)).cpu().numpy())
            targets.append(yb.numpy())
    logits = np.concatenate(logits).astype(np.float32)
    targets = np.concatenate(targets).astype(np.int64)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, logits=logits, targets=targets,
             class_names=np.array(class_names))
    return logits, targets, class_names


# ── calibration ─────────────────────────────────────────────────────────────
def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Single-parameter temperature scaling (Guo et al. 2017), NLL-optimal."""
    lg = torch.tensor(logits)
    tg = torch.tensor(targets)
    log_t = torch.zeros(1, requires_grad=True)          # optimise log T > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), tg)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def malignancy_risk(logits: np.ndarray, class_names: list[str], task: str,
                    temperature: float = 1.0) -> np.ndarray:
    """P(malignant) = softmax mass on the classes that need a human."""
    probs = torch.softmax(torch.tensor(logits) / temperature, dim=1).numpy()
    idx = [i for i, c in enumerate(class_names) if c in MALIGNANT[task]]
    return probs[:, idx].sum(1)


def ece(risk: np.ndarray, label: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of the binary malignancy risk."""
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (risk > lo) & (risk <= hi) if lo > 0 else (risk >= lo) & (risk <= hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(risk[m].mean() - label[m].mean())
    return float(total)


def brier(risk: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean((risk - label) ** 2))


# ── the guarantee ───────────────────────────────────────────────────────────
def cp_upper(k: int, n: int, delta: float) -> float:
    """Clopper-Pearson upper confidence limit for k successes out of n.

    Exact (binomial-tail) rather than normal-approximate, because the whole
    point is the small-n regime where the normal approximation lies.
    """
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1 - delta, k + 1, n - k))


def choose_threshold(pos_risk: np.ndarray, alpha: float, delta: float):
    """Largest rule-out threshold whose miss rate is provably <= alpha.

    Candidates are the calibration positives' own scores. Miss count k(t) is
    non-decreasing in t, so its upper bound is too; we walk thresholds upward
    and stop at the first one that fails. That is fixed-sequence testing, which
    controls the family-wise error without splitting delta across candidates.

    Returns (threshold, k, n, bound). threshold 0.0 means "clear nothing" —
    the only honest answer when the calibration set is too small.
    """
    n = len(pos_risk)
    ordered = np.sort(pos_risk)
    best_t, best_k, best_bound = 0.0, 0, cp_upper(0, n, delta)
    if best_bound > alpha:                     # cannot even promise it at t=0
        return 0.0, 0, n, best_bound

    for i, t in enumerate(ordered):
        k = i + 1                              # thresholding at t misses these
        bound = cp_upper(k, n, delta)
        if bound > alpha:
            break
        best_t, best_k, best_bound = float(t), k, bound
    return best_t, best_k, n, best_bound


def min_positives_needed(alpha: float, delta: float) -> int:
    """Calibration positives required before *any* rule-out is defensible.

    With zero observed misses the bound is 1 - delta^(1/n), so we need
    n >= log(delta) / log(1 - alpha).
    """
    return int(np.ceil(np.log(delta) / np.log(1 - alpha)))


def exchangeability_check(cal_risk: np.ndarray, deploy_risk: np.ndarray,
                          p_thresh: float = 0.05) -> dict:
    """Is the deployment data exchangeable with the calibration data?

    The bound above is distribution-free but *not* assumption-free: it holds
    only if deployment cases are exchangeable with calibration cases. At a new
    site that assumption is exactly what fails, and it fails silently — the
    threshold still returns a number.

    This is a two-sample KS test on the malignancy score, which needs no
    labels, so it can be run on day one at a site that has none. It does not
    prove exchangeability; it detects the shifts big enough to void the
    promise.
    """
    ks = ks_2samp(cal_risk, deploy_risk)
    return {"ks_statistic": float(ks.statistic), "p_value": float(ks.pvalue),
            "shift_detected": bool(ks.pvalue < p_thresh),
            "p_threshold": p_thresh}


# ── evaluation ──────────────────────────────────────────────────────────────
def apply_threshold(risk: np.ndarray, label: np.ndarray, t: float) -> dict:
    cleared = risk < t
    n = len(risk)
    n_pos = int(label.sum())
    missed = int((cleared & (label == 1)).sum())
    referred = ~cleared
    return {
        "threshold": float(t),
        "workload_reduction": float(cleared.mean()),
        "n_cleared": int(cleared.sum()),
        "n_total": n,
        "missed_malignant": missed,
        "n_malignant": n_pos,
        "miss_rate": float(missed / n_pos) if n_pos else float("nan"),
        "sensitivity": float(1 - missed / n_pos) if n_pos else float("nan"),
        "referred_prevalence": (float(label[referred].mean())
                                if referred.sum() else float("nan")),
        "base_prevalence": float(label.mean()),
    }


def sweep(pos_cal: np.ndarray, risk_test: np.ndarray, label_test: np.ndarray,
          delta: float) -> pd.DataFrame:
    """Workload reduction achievable at each tolerated miss rate."""
    rows = []
    for alpha in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        t, k, n, bound = choose_threshold(pos_cal, alpha, delta)
        r = apply_threshold(risk_test, label_test, t)
        rows.append({"alpha": alpha, "threshold": t, "cal_misses": k,
                     "cal_positives": n, "guaranteed_miss_rate": bound,
                     "workload_reduction": r["workload_reduction"],
                     "test_miss_rate": r["miss_rate"],
                     "test_sensitivity": r["sensitivity"],
                     "feasible": t > 0.0})
    return pd.DataFrame(rows)


def stability(task: str, val_logits, val_targets, test_logits, test_targets,
              class_names, alpha: float, delta: float, trials: int) -> dict:
    """Re-draw the calibration split `trials` times and count violations.

    One calibration set gives one threshold; a promise that only holds for the
    split you happened to draw is not a promise. At 95% confidence we expect
    the test miss rate to exceed alpha in at most ~5% of draws.
    """
    mal_idx = [i for i, c in enumerate(class_names) if c in MALIGNANT[task]]
    test_label = np.isin(test_targets, mal_idx).astype(int)
    misses, works = [], []

    for seed in range(trials):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(val_targets))
        n_temp = max(1, int(TEMP_FRACTION * len(perm)))
        i_temp, i_cal = perm[:n_temp], perm[n_temp:]
        temp = fit_temperature(val_logits[i_temp], val_targets[i_temp])
        risk_cal = malignancy_risk(val_logits[i_cal], class_names, task, temp)
        lab_cal = np.isin(val_targets[i_cal], mal_idx).astype(int)
        risk_test = malignancy_risk(test_logits, class_names, task, temp)
        t, *_ = choose_threshold(risk_cal[lab_cal == 1], alpha, delta)
        r = apply_threshold(risk_test, test_label, t)
        misses.append(r["miss_rate"])
        works.append(r["workload_reduction"])

    misses, works = np.array(misses), np.array(works)
    return {"trials": trials,
            "violations": int((misses > alpha).sum()),
            "violation_rate": float((misses > alpha).mean()),
            "miss_rate_mean": float(misses.mean()),
            "miss_rate_max": float(misses.max()),
            "workload_mean": float(works.mean()),
            "workload_min": float(works.min()),
            "workload_max": float(works.max())}


# ── figures ─────────────────────────────────────────────────────────────────
def plot_reliability(risk_raw, risk_cal, label, task, out: Path, n_bins=10):
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    edges = np.linspace(0, 1, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    for risk, name, style in [(risk_raw, "raw softmax", "o--"),
                              (risk_cal, "temperature-scaled", "s-")]:
        xs, ys = [], []
        for lo, hi, mid in zip(edges[:-1], edges[1:], mids):
            m = (risk > lo) & (risk <= hi) if lo > 0 else (risk >= lo) & (risk <= hi)
            if m.sum() < 5:
                continue
            xs.append(risk[m].mean())
            ys.append(label[m].mean())
        ax.plot(xs, ys, style, label=f"{name} (ECE {ece(risk, label):.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
    ax.set_xlabel("predicted P(malignant)")
    ax.set_ylabel("observed fraction malignant")
    ax.set_title(f"OncoAI — {task}: does the number mean what it says?",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(df: pd.DataFrame, task: str, out: Path):
    ok = df[df["feasible"]]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    if len(ok):
        ax.plot(ok["alpha"] * 100, ok["workload_reduction"] * 100, "o-",
                color="#2b6cb0", label="workload removed (test)")
        ax.plot(ok["alpha"] * 100, ok["test_miss_rate"] * 100, "s--",
                color="#c53030", label="cancers missed (test)")
        ax.plot(ok["alpha"] * 100, ok["alpha"] * 100, "k:", lw=1,
                label="the promise")
    infeasible = df[~df["feasible"]]
    for a in infeasible["alpha"]:
        ax.axvspan(a * 100 - 0.4, a * 100 + 0.4, color="grey", alpha=0.15)
    ax.set_xlabel("tolerated miss rate α  (%)")
    ax.set_ylabel("percent of cases")
    ax.set_title(f"OncoAI — {task}: what a guarantee costs",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── driver ──────────────────────────────────────────────────────────────────
def run_task(task: str, alpha: float, delta: float, seed: int, force: bool,
             trials: int = 0) -> dict:
    val_logits, val_targets, class_names = collect_logits(task, "val", force)
    test_logits, test_targets, _ = collect_logits(task, "test", force)

    mal_idx = {i for i, c in enumerate(class_names) if c in MALIGNANT[task]}
    if not mal_idx:
        raise ValueError(f"no malignant classes matched for {task}: {class_names}")
    val_label = np.isin(val_targets, list(mal_idx)).astype(int)
    test_label = np.isin(test_targets, list(mal_idx)).astype(int)

    # Split validation: temperature on one part, threshold on the other, so the
    # guarantee is not made on data the score function was tuned to.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(val_targets))
    n_temp = max(1, int(TEMP_FRACTION * len(perm)))
    i_temp, i_cal = perm[:n_temp], perm[n_temp:]

    temperature = fit_temperature(val_logits[i_temp], val_targets[i_temp])

    risk_test_raw = malignancy_risk(test_logits, class_names, task, 1.0)
    risk_test = malignancy_risk(test_logits, class_names, task, temperature)
    risk_cal = malignancy_risk(val_logits[i_cal], class_names, task, temperature)
    cal_label = val_label[i_cal]
    pos_cal = risk_cal[cal_label == 1]

    validity = exchangeability_check(risk_cal, risk_test)

    need = min_positives_needed(alpha, delta)
    t, k, n, bound = choose_threshold(pos_cal, alpha, delta)
    res = apply_threshold(risk_test, test_label, t)

    out = {
        "task": task,
        "classes": class_names,
        "malignant_classes": sorted(MALIGNANT[task]),
        "alpha": alpha,
        "delta": delta,
        "temperature": temperature,
        "calibration": {
            "n_temp_fit": int(len(i_temp)),
            "n_threshold": int(len(i_cal)),
            "n_threshold_positives": int(len(pos_cal)),
            "positives_needed_for_alpha": need,
            "sufficient": bool(len(pos_cal) >= need),
            "ece_raw": ece(risk_test_raw, test_label),
            "ece_calibrated": ece(risk_test, test_label),
            "brier_raw": brier(risk_test_raw, test_label),
            "brier_calibrated": brier(risk_test, test_label),
        },
        "validity": validity,
        "guarantee": {
            "threshold": t,
            "calibration_misses": k,
            "calibration_positives": n,
            "guaranteed_miss_rate": bound,
            "feasible": t > 0.0,
        },
        "test": res,
    }

    if trials:
        out["stability"] = stability(task, val_logits, val_targets, test_logits,
                                     test_targets, class_names, alpha, delta, trials)

    NOVEL.mkdir(parents=True, exist_ok=True)
    df = sweep(pos_cal, risk_test, test_label, delta)
    df.to_csv(NOVEL / f"triage_sweep_{task}.csv", index=False)
    out["sweep"] = df.to_dict(orient="records")
    with open(NOVEL / f"triage_{task}.json", "w") as f:
        json.dump(out, f, indent=2)

    plot_reliability(risk_test_raw, risk_test, test_label, task,
                     NOVEL / f"triage_reliability_{task}.png")
    plot_tradeoff(df, task, NOVEL / f"triage_tradeoff_{task}.png")
    return out


def report(out: dict) -> None:
    t = out["task"]
    c, g, r, v = out["calibration"], out["guarantee"], out["test"], out["validity"]
    print(f"\n{'=' * 66}\n{t.upper()} — rule-out triage\n{'=' * 66}")
    print(f"malignant = {', '.join(out['malignant_classes'])}")
    print(f"temperature {out['temperature']:.3f}   "
          f"ECE {c['ece_raw']:.3f} -> {c['ece_calibrated']:.3f}   "
          f"Brier {c['brier_raw']:.3f} -> {c['brier_calibrated']:.3f}")
    print(f"calibration positives: {c['n_threshold_positives']}"
          f"  (need >= {c['positives_needed_for_alpha']} for "
          f"alpha={out['alpha']:.0%} at {1 - out['delta']:.0%} confidence)")
    print(f"exchangeability:  KS {v['ks_statistic']:.3f}, p={v['p_value']:.2e}"
          f"  -> {'SHIFT DETECTED' if v['shift_detected'] else 'consistent'}")

    if not g["feasible"]:
        print(f"\n  NO DEFENSIBLE THRESHOLD. With {c['n_threshold_positives']} "
              f"malignant calibration cases the tightest\n  provable miss rate is "
              f"{g['guaranteed_miss_rate']:.1%}, above the {out['alpha']:.0%} asked for. "
              f"Refusing to\n  clear anything is the correct output here.")
        return

    print(f"\n  threshold          P(malignant) < {g['threshold']:.4f}")
    print(f"  promised           <= {g['guaranteed_miss_rate']:.2%} of cancers missed "
          f"({g['calibration_misses']}/{g['calibration_positives']} in calibration)")
    print(f"  delivered (test)   {r['miss_rate']:.2%} missed "
          f"({r['missed_malignant']}/{r['n_malignant']}), "
          f"sensitivity {r['sensitivity']:.1%}")
    print(f"  workload removed   {r['workload_reduction']:.1%} "
          f"({r['n_cleared']}/{r['n_total']} cases auto-cleared)")
    print(f"  prevalence         {r['base_prevalence']:.1%} overall -> "
          f"{r['referred_prevalence']:.1%} in what reaches the radiologist")

    if "stability" in out:
        st = out["stability"]
        print(f"  across {st['trials']} calibration draws: "
              f"{st['violations']} violations of the {out['alpha']:.0%} bound "
              f"(max {st['miss_rate_max']:.2%}),\n"
              f"                     workload cut {st['workload_min']:.1%}"
              f"-{st['workload_max']:.1%} (mean {st['workload_mean']:.1%})")

    if v["shift_detected"]:
        print(f"\n  ** GUARANTEE VOID ** the evaluation set is not exchangeable with\n"
              f"  the calibration set (KS p={v['p_value']:.1e}), so the {g['guaranteed_miss_rate']:.1%} bound\n"
              f"  does not apply to it — and the {r['miss_rate']:.1%} actually missed shows it.\n"
              f"  Recalibrate on data from the deployment site.")
    elif r["miss_rate"] > g["guaranteed_miss_rate"]:
        print(f"\n  ** bound exceeded on test without a detected shift — "
              f"investigate before trusting this threshold.")


def main():
    ap = argparse.ArgumentParser(description="Rule-out triage with a guaranteed miss rate")
    ap.add_argument("--task", choices=["brain", "lung", "skin"], action="append",
                    help="repeatable; default all three")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="tolerated miss rate among malignant cases")
    ap.add_argument("--delta", type=float, default=0.05,
                    help="1-delta is the confidence in that bound")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="recompute cached logits")
    ap.add_argument("--trials", type=int, default=0,
                    help="re-draw the calibration split N times to test the bound")
    a = ap.parse_args()

    tasks = a.task or ["brain", "lung", "skin"]
    rows = []
    for task in tasks:
        out = run_task(task, a.alpha, a.delta, a.seed, a.force, a.trials)
        report(out)
        g, r, c = out["guarantee"], out["test"], out["calibration"]
        rows.append({"task": task, "temperature": out["temperature"],
                     "ece_raw": c["ece_raw"], "ece_cal": c["ece_calibrated"],
                     "cal_positives": c["n_threshold_positives"],
                     "positives_needed": c["positives_needed_for_alpha"],
                     "feasible": g["feasible"], "threshold": g["threshold"],
                     "guaranteed_miss_rate": g["guaranteed_miss_rate"],
                     "test_miss_rate": r["miss_rate"],
                     "workload_reduction": r["workload_reduction"],
                     "ks_p": out["validity"]["p_value"],
                     "shift_detected": out["validity"]["shift_detected"]})

    df = pd.DataFrame(rows)
    NOVEL.mkdir(parents=True, exist_ok=True)
    df.to_csv(NOVEL / "triage_summary.csv", index=False)
    print(f"\n{'=' * 66}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwritten to results/novel/triage_*.{{json,csv,png}}")


if __name__ == "__main__":
    main()
