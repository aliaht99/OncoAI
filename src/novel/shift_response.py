#!/usr/bin/env python3
"""
Predicting the miss rate at a site you have no labels for.

`site_shift.py` establishes the negative result: generic drift statistics do not
answer the safety question. The ones sensitive enough to catch every breach
(KS, Mahalanobis) fire on essentially every deployment, breached or not, which
is precisely the alert fatigue that gets monitoring switched off; the quieter
ones miss over half the real breaches. None of them is usable as a gate.

The reason is that they are all measuring the wrong thing. "Is this data
different?" is not "is my threshold still safe?", and the map between the two is
a property of *this particular model* — which means it can be learned, because
the one thing a deployment does have is its own labelled calibration set and the
ability to corrupt it.

So: simulate acquisition shifts on the labelled source split, record for each
one a label-free feature vector and the true miss rate it caused, and fit a
**conservative upper quantile** of miss rate given features. At a new site,
compute the same features on unlabelled images and read off a predicted upper
bound. Alarm when that bound exceeds the promise.

The obvious objection is that this only works for shifts resembling the
simulated ones, so the evaluation is built to attack exactly that:

  * **leave-one-family-out** — hold out an entire corruption family (all blur
    severities, all JPEG qualities, …) and predict it from the others;
  * **population shifts** — evaluated but never trained on. The training set
    contains no change of patient mix at all, only changes of image.

    python src/novel/shift_response.py
    python src/novel/shift_response.py --force     # recompute source features

Requires `site_shift.py` to have run. Writes results/novel/shift_response*.
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
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data as data_mod                     # noqa: E402
from common.model import get_device, load_checkpoint    # noqa: E402
from novel.triage import MALIGNANT, apply_threshold, malignancy_risk  # noqa: E402
from novel.site_shift import (ACQUISITION, FEATURE_NAMES, MONITOR_NAMES,  # noqa: E402
                              PRETTY_MONITOR, TASK, Monitors,
                              decode_test_images, infer)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
NOVEL = ROOT / "results" / "novel"

# Severities of one corruption are not independent evidence about another, so
# generalisation is measured across these groups, never within one.
FAMILY = {
    "jpeg": "jpeg", "blur": "blur", "bright": "brightness",
    "contrast": "contrast", "gamma": "gamma", "warm": "colour_temp",
    "cool": "colour_temp", "desaturate": "saturation", "saturate": "saturation",
    "res": "resolution", "noise": "noise",
}


def family_of(name: str) -> str:
    return FAMILY.get(name.split("_")[0], "other")


# ── training data: shifts we can score, because we have the labels ──────────
def build_source_table(threshold: float, temp: float, subsamples: int,
                       frac: float, seed: int, force: bool) -> pd.DataFrame:
    """Corrupt the labelled calibration split and record (features, miss rate).

    Subsampling each corrupted set serves two purposes: it multiplies the number
    of training rows, and it exposes the model to the sampling noise a
    small deployment will have.
    """
    cache = NOVEL / "shift_response_source.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)

    device = get_device()
    model, ckpt = load_checkpoint(MODELS / f"{TASK}_model.pth", device)
    class_names = ckpt["class_names"]
    mal_idx = [i for i, c in enumerate(class_names) if c in MALIGNANT[TASK]]

    _, df_val, _, _ = data_mod.skin_frames()
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    y_val = df_val["dx"].astype(str).map(name_to_idx).to_numpy()
    lab = np.isin(y_val, mal_idx).astype(int)
    images = decode_test_images(df_val)

    print("source reference pass …")
    clean_logits, clean_emb = infer(model, images, device)
    mon = Monitors(clean_logits, y_val, clean_emb, class_names, TASK, temp)

    rng = np.random.default_rng(seed)
    rows = []

    def add(name, family, lg, em, y):
        risk = malignancy_risk(lg, class_names, TASK, temp)
        res = apply_threshold(risk, y, threshold)
        if not np.isfinite(res["miss_rate"]):
            return
        feats = mon.compute(lg, em, threshold)
        rows.append({"scenario": name, "family": family, "n": len(y),
                     "miss_rate": res["miss_rate"],
                     **{f"v_{k}": feats[k] for k in FEATURE_NAMES}})

    passes = [("clean", "none", clean_logits, clean_emb)]
    for i, (name, fn) in enumerate(ACQUISITION.items(), 1):
        print(f"source shift {i}/{len(ACQUISITION)}: {name}")
        lg, em = infer(model, [fn(im) for im in images], device)
        passes.append((name, family_of(name), lg, em))

    for name, family, lg, em in passes:
        add(name, family, lg, em, lab)
        for _ in range(subsamples):
            idx = rng.choice(len(lab), size=int(frac * len(lab)), replace=False)
            add(name, family, lg[idx], em[idx], lab[idx])

    df = pd.DataFrame(rows)
    NOVEL.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


# ── the model ───────────────────────────────────────────────────────────────
def make_model(quantile: float) -> GradientBoostingRegressor:
    """Upper-quantile regression: safety wants a bound, not a best guess.

    Deliberately small — the training set is tens of shift settings, not
    thousands, and a deep model would memorise corruption identity instead of
    learning the response.
    """
    return GradientBoostingRegressor(
        loss="quantile", alpha=quantile, max_depth=2,
        n_estimators=300, learning_rate=0.05, random_state=0)


def leave_one_family_out(df: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """Predict each corruption family using only the others."""
    feats = [f"v_{k}" for k in FEATURE_NAMES]
    out = []
    families = [f for f in df["family"].unique() if f != "none"]
    for fam in families:
        tr = df[df["family"] != fam]
        te = df[df["family"] == fam]
        m = make_model(quantile).fit(tr[feats], tr["miss_rate"])
        pred = m.predict(te[feats])
        out.append({"family": fam, "n": len(te),
                    "true_mean": float(te["miss_rate"].mean()),
                    "pred_mean": float(pred.mean()),
                    "coverage": float((pred >= te["miss_rate"]).mean()),
                    "mae": float(np.abs(pred - te["miss_rate"]).mean())})
    return pd.DataFrame(out)


# ── evaluation against the held-out deployments ─────────────────────────────
def evaluate(df_src: pd.DataFrame, df_tgt: pd.DataFrame, alpha: float,
             quantile: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the predictor on site_shift.py's deployments, and the baselines with it.

    The target rows come from a different image split than anything the
    predictor trained on, and the population scenarios are a kind of shift it
    never saw.
    """
    feats = [f"v_{k}" for k in FEATURE_NAMES]
    model = make_model(quantile).fit(df_src[feats], df_src["miss_rate"])

    tgt = df_tgt[np.isfinite(df_tgt["miss_rate"])].copy()
    tgt["predicted"] = model.predict(tgt[feats])
    tgt["alarm_predictor"] = tgt["predicted"] > alpha

    breach = tgt["breach"].to_numpy().astype(bool)
    rows = []

    def score(label, fired):
        tp = int((fired & breach).sum()); fn = int((~fired & breach).sum())
        fp = int((fired & ~breach).sum()); tn = int((~fired & ~breach).sum())
        rows.append({"monitor": label, "detected": tp, "missed": fn,
                     "detection_rate": tp / (tp + fn) if tp + fn else np.nan,
                     "false_alarms": fp,
                     "false_alarm_rate": fp / (fp + tn) if fp + tn else np.nan})

    for k in MONITOR_NAMES:
        score(PRETTY_MONITOR[k], tgt[f"a_{k}"].to_numpy().astype(bool))
    score("shift-response model (ours)", tgt["alarm_predictor"].to_numpy())

    summary = pd.DataFrame(rows)
    summary["balance"] = summary["detection_rate"] - summary["false_alarm_rate"]
    return tgt, summary.sort_values("balance", ascending=False)


def plot_predicted_vs_true(tgt: pd.DataFrame, alpha: float, out: Path):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for kind, colour, marker in [("acquisition", "#2b6cb0", "o"),
                                 ("population", "#b7791f", "s"),
                                 ("reference", "#2f855a", "D")]:
        sub = tgt[tgt["kind"] == kind]
        if len(sub):
            ax.scatter(sub["miss_rate"] * 100, sub["predicted"] * 100,
                       c=colour, marker=marker, s=55, alpha=0.85,
                       edgecolor="white", linewidth=0.8, label=kind)

    lim = max(tgt["miss_rate"].max(), tgt["predicted"].max()) * 100 * 1.12
    ax.plot([0, lim], [0, lim], "k:", lw=1, label="perfect")
    ax.axhline(alpha * 100, color="#c53030", lw=1.2, ls="--")
    ax.axvline(alpha * 100, color="#c53030", lw=1.2, ls="--")
    ax.text(lim * 0.98, alpha * 100 + lim * 0.015, "alarm level", ha="right",
            fontsize=8, color="#c53030")
    ax.set_xlabel("true miss rate at the frozen threshold (%)")
    ax.set_ylabel("predicted upper bound, no labels used (%)")
    ax.set_title("OncoAI — predicting an unsafe deployment before it is measured",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Learn how this model's miss rate responds to shift")
    ap.add_argument("--quantile", type=float, default=0.90,
                    help="upper quantile the predictor targets")
    ap.add_argument("--subsamples", type=int, default=6)
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    tri = json.loads((NOVEL / f"triage_{TASK}.json").read_text())
    shift_path = NOVEL / "site_shift.json"
    if not shift_path.exists():
        sys.exit("Missing results/novel/site_shift.json — run src/novel/site_shift.py first.")
    shift = json.loads(shift_path.read_text())
    alpha, threshold, temp = shift["alpha"], shift["threshold"], tri["temperature"]

    df_src = build_source_table(threshold, temp, a.subsamples, a.frac, a.seed, a.force)
    df_tgt = pd.read_csv(NOVEL / "site_shift_scenarios.csv")

    lofo = leave_one_family_out(df_src, a.quantile)
    tgt, summary = evaluate(df_src, df_tgt, alpha, a.quantile)

    lofo.to_csv(NOVEL / "shift_response_lofo.csv", index=False)
    summary.to_csv(NOVEL / "shift_response_monitors.csv", index=False)
    tgt.to_csv(NOVEL / "shift_response_predictions.csv", index=False)
    plot_predicted_vs_true(tgt, alpha, NOVEL / "shift_response_scatter.png")

    by_kind = (tgt.assign(hit=tgt["alarm_predictor"] == tgt["breach"])
                  .groupby("kind")
                  .agg(n=("hit", "size"), breaches=("breach", "sum"),
                       correct=("hit", "sum"))
                  .reset_index())

    with open(NOVEL / "shift_response.json", "w") as f:
        json.dump({"alpha": alpha, "quantile": a.quantile,
                   "n_source_rows": len(df_src),
                   "lofo": lofo.to_dict(orient="records"),
                   "monitors": summary.to_dict(orient="records"),
                   "by_kind": by_kind.to_dict(orient="records")},
                  f, indent=2, default=float)

    print(f"\n{'=' * 76}\nSHIFT-RESPONSE MODEL — {TASK}\n{'=' * 76}")
    print(f"trained on {len(df_src)} corrupted views of the labelled calibration "
          f"split\ntarget: the {a.quantile:.0%} upper quantile of miss rate\n")

    print("leave-one-family-out — each family predicted from the others only:")
    print(lofo.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nmean coverage (bound held): {lofo['coverage'].mean():.1%}")

    print("\nalarm quality on the held-out deployments:")
    print(summary[["monitor", "detected", "missed", "detection_rate",
                   "false_alarms", "false_alarm_rate"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\npredictor accuracy by shift kind (population never seen in training):")
    print(by_kind.to_string(index=False))
    print(f"\nwritten to results/novel/shift_response*")


if __name__ == "__main__":
    main()
