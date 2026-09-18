#!/usr/bin/env python3
"""
Does the rule-out promise survive the next hospital — and can you tell without
labels?

`triage.py` produces a threshold whose miss rate is provably bounded *on data
exchangeable with the calibration set*. The moment a model is installed
somewhere else, that precondition is the first thing to go: different
dermatoscope, different illumination, different patient mix. Published external
validations lose a median of ~0.06 accuracy, and the site that just installed
the model has no labels on day one with which to notice.

This module asks the deployment-shaped version of that question. Not "how much
accuracy did we lose" — a department cannot act on that either — but:

    "Has my rule-out threshold become unsafe, and would anything have told me
     before a patient was silently cleared?"

We build shifted versions of the dermatoscopy test set — acquisition changes
that mimic a different camera, and natural population slices taken from the
HAM10000 metadata — then, for each one, measure the *true* miss rate at the
frozen threshold (labels used only for scoring, never by the monitors) and ask
which label-free monitor would have raised the alarm.

The evaluation metric is deliberately not "did the monitor notice a shift".
Monitors that fire on every shift are the documented reason clinicians switch
alerts off. What matters is whether a monitor fires when, and only when, the
safety property actually breaks.

    python src/novel/site_shift.py
    python src/novel/site_shift.py --alpha 0.05 --bootstrap 400

Writes results/novel/site_shift_*.{json,csv,png}.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data as data_mod                     # noqa: E402
from common.model import get_device, load_checkpoint    # noqa: E402
from novel.triage import (MALIGNANT, apply_threshold, cp_upper,   # noqa: E402
                          malignancy_risk, min_positives_needed)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
NOVEL = ROOT / "results" / "novel"

TASK = "skin"          # the only task with a calibration set big enough to bound
WORK_SIZE = 256        # corruptions are applied here, then resized to the model's 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── acquisition shifts ──────────────────────────────────────────────────────
# Each mimics a plausible difference between one clinic's imaging chain and
# another's. Severities are ordered so the sweep shows where safety breaks.
def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _gamma(img: Image.Image, g: float) -> Image.Image:
    lut = [int(np.clip(((i / 255.0) ** g) * 255.0, 0, 255)) for i in range(256)]
    return img.point(lut * 3)


def _colour_temp(img: Image.Image, k: float) -> Image.Image:
    """Warm (k>1) or cool (k<1) white balance — dermatoscopes differ a lot here."""
    a = np.asarray(img).astype(np.float32)
    a[..., 0] *= k
    a[..., 2] /= k
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _resolution(img: Image.Image, frac: float) -> Image.Image:
    small = img.resize((max(8, int(img.width * frac)), max(8, int(img.height * frac))),
                       Image.BILINEAR)
    return small.resize(img.size, Image.BILINEAR)


def _noise(img: Image.Image, sigma: float, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    a = np.asarray(img).astype(np.float32)
    a += rng.normal(0, sigma, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


ACQUISITION = {
    "jpeg_q50":      lambda im: _jpeg(im, 50),
    "jpeg_q25":      lambda im: _jpeg(im, 25),
    "jpeg_q10":      lambda im: _jpeg(im, 10),
    "blur_0.7":      lambda im: im.filter(ImageFilter.GaussianBlur(0.7)),
    "blur_1.5":      lambda im: im.filter(ImageFilter.GaussianBlur(1.5)),
    "blur_3.0":      lambda im: im.filter(ImageFilter.GaussianBlur(3.0)),
    "bright_0.7":    lambda im: ImageEnhance.Brightness(im).enhance(0.7),
    "bright_1.3":    lambda im: ImageEnhance.Brightness(im).enhance(1.3),
    "bright_1.6":    lambda im: ImageEnhance.Brightness(im).enhance(1.6),
    "contrast_0.6":  lambda im: ImageEnhance.Contrast(im).enhance(0.6),
    "contrast_1.6":  lambda im: ImageEnhance.Contrast(im).enhance(1.6),
    "gamma_0.6":     lambda im: _gamma(im, 0.6),
    "gamma_1.8":     lambda im: _gamma(im, 1.8),
    "warm_1.10":     lambda im: _colour_temp(im, 1.10),
    "warm_1.25":     lambda im: _colour_temp(im, 1.25),
    "cool_0.90":     lambda im: _colour_temp(im, 0.90),
    "cool_0.80":     lambda im: _colour_temp(im, 0.80),
    "desaturate":    lambda im: ImageEnhance.Color(im).enhance(0.5),
    "saturate":      lambda im: ImageEnhance.Color(im).enhance(1.6),
    "res_0.5":       lambda im: _resolution(im, 0.5),
    "res_0.25":      lambda im: _resolution(im, 0.25),
    "noise_8":       lambda im: _noise(im, 8),
    "noise_16":      lambda im: _noise(im, 16),
}


def population_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Natural slices of the test set — the shift a new catchment area brings."""
    loc = df["localization"].astype(str)
    age = pd.to_numeric(df["age"], errors="coerce")
    head = loc.isin(["face", "ear", "neck", "scalp"])
    acral = loc.isin(["hand", "foot", "acral"])
    trunk = loc.isin(["back", "trunk", "abdomen", "chest"])
    limbs = loc.isin(["upper extremity", "lower extremity"])
    masks = {
        "pop_head_neck": head.to_numpy(),
        "pop_acral": acral.to_numpy(),
        "pop_trunk": trunk.to_numpy(),
        "pop_limbs": limbs.to_numpy(),
        "pop_age_under50": (age < 50).fillna(False).to_numpy(),
        "pop_age_70plus": (age >= 70).fillna(False).to_numpy(),
        "pop_female": (df["sex"].astype(str) == "female").to_numpy(),
        "pop_male": (df["sex"].astype(str) == "male").to_numpy(),
    }
    # Deliberately no dx_type slice: in HAM10000 every malignant diagnosis is
    # histology-confirmed, so slicing on it partly slices on the label, and the
    # resulting "shift" would be an artefact of the annotation process.
    return {k: m for k, m in masks.items() if m.sum() >= 120}


# ── inference ───────────────────────────────────────────────────────────────
def decode_test_images(df: pd.DataFrame) -> list[Image.Image]:
    """Decode once at WORK_SIZE; every scenario is a cheap transform of these."""
    out = []
    for cell in df["image"].to_numpy():
        blob = cell["bytes"] if isinstance(cell, dict) else cell
        im = Image.open(io.BytesIO(blob)).convert("RGB")
        out.append(im.resize((WORK_SIZE, WORK_SIZE), Image.BILINEAR))
    return out


@torch.no_grad()
def infer(model, images: list[Image.Image], device, batch: int = 64):
    """Logits and penultimate embeddings for a list of PIL images."""
    logits, embs = [], []
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        arr = np.stack([np.asarray(im.resize((224, 224), Image.BILINEAR),
                                   dtype=np.float32) / 255.0 for im in chunk])
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(device)
        lg, em = model(x, return_embedding=True)
        logits.append(lg.cpu().numpy())
        embs.append(em.cpu().numpy())
    return np.concatenate(logits).astype(np.float32), np.concatenate(embs).astype(np.float32)


# ── label-free monitors ─────────────────────────────────────────────────────
# Every monitor is a scalar where larger = more alarming, computed from model
# outputs alone. None of them may touch `targets`.
def softmax_np(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Monitors:
    """Source-side reference statistics, plus the monitors that use them."""

    def __init__(self, src_logits, src_targets, src_emb, class_names, task, temp):
        self.class_names, self.task, self.temp = class_names, task, temp
        p = softmax_np(src_logits)
        self.src_conf = p.max(1)
        self.src_err = float((p.argmax(1) != src_targets).mean())
        self.src_risk = malignancy_risk(src_logits, class_names, task, temp)
        self.src_energy = -np.log(np.exp(src_logits).sum(1))

        # ATC: the confidence level below which source errors accumulate at the
        # source error rate. Target error is then estimated by counting alone.
        self.atc_tau = float(np.quantile(self.src_conf, self.src_err))

        # Mahalanobis needs a source Gaussian over embeddings.
        mu = src_emb.mean(0)
        cov = np.cov(src_emb, rowvar=False) + 1e-3 * np.eye(src_emb.shape[1])
        self.emb_mu, self.emb_prec = mu, np.linalg.inv(cov)

    def _maha(self, emb):
        d = emb - self.emb_mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.emb_prec, d))

    def compute(self, logits, emb, threshold) -> dict:
        p = softmax_np(logits)
        conf = p.max(1)
        risk = malignancy_risk(logits, self.class_names, self.task, self.temp)
        energy = -np.log(np.exp(logits).sum(1))
        cleared = risk < threshold

        # Plug-in miss rate: trust the model's own risk on the pile it cleared.
        # Expected missed cancers / expected cancers, no labels involved.
        denom = risk.sum()
        cleared_mass = float(risk[cleared].sum() / denom) if denom > 0 else 0.0

        return {
            "conf_drop": float(self.src_conf.mean() - conf.mean()),
            "ks_risk": float(ks_2samp(self.src_risk, risk).statistic),
            "atc_err_rise": float((conf < self.atc_tau).mean() - self.src_err),
            "mahalanobis": float(self._maha(emb).mean()),
            "energy": float(energy.mean()),
            "cleared_mass": cleared_mass,
            # Extra label-free descriptors. Not benchmarked as standalone
            # alarms — they are the feature vector shift_response.py learns on.
            "cleared_frac": float(cleared.mean()),
            "mean_risk": float(risk.mean()),
            "risk_q10": float(np.quantile(risk, 0.10)),
            "risk_q25": float(np.quantile(risk, 0.25)),
            "conf_mean": float(conf.mean()),
        }


# The six benchmarked as standalone alarms.
MONITOR_NAMES = ["conf_drop", "ks_risk", "atc_err_rise", "mahalanobis",
                 "energy", "cleared_mass"]
# Everything compute() returns — what the shift-response model sees.
FEATURE_NAMES = MONITOR_NAMES + ["cleared_frac", "mean_risk", "risk_q10",
                                 "risk_q25", "conf_mean"]

PRETTY_MONITOR = {
    "conf_drop": "mean-confidence drop",
    "ks_risk": "KS on risk score",
    "atc_err_rise": "ATC estimated error rise",
    "mahalanobis": "Mahalanobis (embeddings)",
    "energy": "mean free energy",
    "cleared_mass": "plug-in cleared-risk mass",
}


def null_thresholds(mon: Monitors, src_logits, src_emb, n: int, threshold: float,
                    bootstrap: int, rng) -> dict:
    """Alarm level per monitor: the 95th percentile under no shift, at size n.

    Calibrated at the *target's* sample size, because several of these statistics
    (KS especially) depend on it, and a slice of 300 cases must not look alarming
    merely for being small.
    """
    vals = {k: [] for k in MONITOR_NAMES}
    for _ in range(bootstrap):
        idx = rng.integers(0, len(src_logits), size=n)
        v = mon.compute(src_logits[idx], src_emb[idx], threshold)
        for k in MONITOR_NAMES:
            vals[k].append(v[k])
    return {k: float(np.quantile(vals[k], 0.95)) for k in MONITOR_NAMES}


# ── figures ─────────────────────────────────────────────────────────────────
def plot_scenarios(df: pd.DataFrame, alpha: float, out: Path):
    """Every simulated deployment, ranked by how many cancers it silently cleared."""
    d = df.sort_values("miss_rate")
    colours = {"reference": "#1a202c", "population": "#b7791f", "acquisition": "#2b6cb0"}
    fig, ax = plt.subplots(figsize=(7.2, 0.26 * len(d) + 1.6))
    ax.barh(range(len(d)), d["miss_rate"] * 100,
            color=[colours[k] for k in d["kind"]])
    ax.axvline(alpha * 100, color="#c53030", ls="--", lw=1.6,
               label=f"the promise ({alpha:.0%})")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["scenario"], fontsize=7)
    ax.set_xlabel("cancers missed by the frozen rule-out threshold (%)")
    ax.set_title("OncoAI — one threshold, many deployments", fontweight="bold", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    ax.legend(handles + [plt.Line2D([0], [0], color="#c53030", ls="--")],
              list(colours) + [f"the promise ({alpha:.0%})"], fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_monitors(sm: pd.DataFrame, out: Path):
    """Detection of real breaches vs noise generated when nothing is wrong."""
    d = sm.sort_values("detection_rate")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(d) + 1.8))
    ax.barh(y + 0.19, d["detection_rate"] * 100, height=0.36,
            color="#2f855a", label="breaches caught")
    ax.barh(y - 0.19, d["false_alarm_rate"] * 100, height=0.36,
            color="#c53030", label="false alarms when safe")
    ax.set_yticks(y)
    ax.set_yticklabels(d["monitor"], fontsize=8)
    ax.set_xlabel("percent")
    ax.set_xlim(0, 100)
    ax.set_title("OncoAI — a monitor that always fires is not a monitor",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_figures(df: pd.DataFrame, sm: pd.DataFrame, alpha: float) -> None:
    plot_scenarios(df, alpha, NOVEL / "site_shift_scenarios.png")
    plot_monitors(sm, NOVEL / "site_shift_monitors.png")


# ── driver ──────────────────────────────────────────────────────────────────
def build_scenarios(logits_clean, emb_clean, labels, df):
    """(name, kind, logits, emb, labels) for every deployment we simulate.

    Population slices are index views of the clean pass — the images are
    unchanged, only who walks through the door is.
    """
    scen = [("clean", "reference", logits_clean, emb_clean, labels)]
    for name, mask in population_masks(df).items():
        scen.append((name, "population", logits_clean[mask], emb_clean[mask], labels[mask]))
    return scen


def main():
    ap = argparse.ArgumentParser(description="Does the rule-out promise survive a new site?")
    ap.add_argument("--alpha", type=float, default=None,
                    help="tolerated miss rate; default: whatever triage.py used")
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="resamples used to set each monitor's alarm level")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw from results/novel/site_shift.json")
    a = ap.parse_args()

    if a.figures_only:
        saved = json.loads((NOVEL / "site_shift.json").read_text())
        df = pd.DataFrame(saved["scenarios"])
        sm = pd.DataFrame(saved["monitors"])
        write_figures(df, sm, saved["alpha"])
        print("figures redrawn from results/novel/site_shift.json")
        return df, sm, saved

    tri_path = NOVEL / f"triage_{TASK}.json"
    if not tri_path.exists():
        sys.exit(f"Missing {tri_path} — run `python src/novel/triage.py --task {TASK}` first.")
    tri = json.loads(tri_path.read_text())
    if not tri["guarantee"]["feasible"]:
        sys.exit("triage.py found no defensible threshold; nothing to monitor.")
    temp = tri["temperature"]
    threshold = tri["guarantee"]["threshold"]
    alpha = a.alpha if a.alpha is not None else tri["alpha"]
    delta = tri["delta"]

    device = get_device()
    model, ckpt = load_checkpoint(MODELS / f"{TASK}_model.pth", device)
    class_names = ckpt["class_names"]
    mal_idx = [i for i, c in enumerate(class_names) if c in MALIGNANT[TASK]]

    print("decoding images …")
    _, df_val, df_test, _ = data_mod.skin_frames()
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    y_val = df_val["dx"].astype(str).map(name_to_idx).to_numpy()
    y_test = df_test["dx"].astype(str).map(name_to_idx).to_numpy()
    im_val = decode_test_images(df_val)
    im_test = decode_test_images(df_test)

    print("source pass (calibration split) …")
    src_logits, src_emb = infer(model, im_val, device)
    mon = Monitors(src_logits, y_val, src_emb, class_names, TASK, temp)

    print("clean target pass …")
    clean_logits, clean_emb = infer(model, im_test, device)

    lab_test = np.isin(y_test, mal_idx).astype(int)
    scenarios = build_scenarios(clean_logits, clean_emb, lab_test, df_test)

    for i, (name, fn) in enumerate(ACQUISITION.items(), 1):
        print(f"acquisition shift {i}/{len(ACQUISITION)}: {name}")
        shifted = [fn(im) for im in im_test]
        lg, em = infer(model, shifted, device)
        scenarios.append((name, "acquisition", lg, em, lab_test))

    rng = np.random.default_rng(a.seed)
    rows = []
    for name, kind, lg, em, lab in scenarios:
        risk = malignancy_risk(lg, class_names, TASK, temp)
        res = apply_threshold(risk, lab, threshold)
        breach = bool(res["miss_rate"] > alpha)

        vals = mon.compute(lg, em, threshold)
        cuts = null_thresholds(mon, src_logits, src_emb, len(lab), threshold,
                               a.bootstrap, rng)
        alarms = {k: bool(vals[k] > cuts[k]) for k in MONITOR_NAMES}

        row = {"scenario": name, "kind": kind, "n": len(lab),
               "n_malignant": int(lab.sum()),
               "cleared": res["workload_reduction"],
               "miss_rate": res["miss_rate"], "breach": breach}
        row.update({f"v_{k}": vals[k] for k in FEATURE_NAMES})
        row.update({f"a_{k}": alarms[k] for k in MONITOR_NAMES})
        rows.append(row)

    df = pd.DataFrame(rows)
    NOVEL.mkdir(parents=True, exist_ok=True)
    df.to_csv(NOVEL / "site_shift_scenarios.csv", index=False)

    # ── how well does each monitor track the thing that actually matters? ────
    breaches = df["breach"].to_numpy()
    summary = []
    for k in MONITOR_NAMES:
        fired = df[f"a_{k}"].to_numpy()
        tp = int((fired & breaches).sum())
        fn = int((~fired & breaches).sum())
        fp = int((fired & ~breaches).sum())
        tn = int((~fired & ~breaches).sum())
        summary.append({
            "monitor": PRETTY_MONITOR[k],
            "key": k,
            "detected": tp, "missed": fn,
            "detection_rate": tp / (tp + fn) if tp + fn else np.nan,
            "false_alarms": fp, "quiet_when_safe": tn,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else np.nan,
        })
    sm = pd.DataFrame(summary).sort_values(
        ["detection_rate", "false_alarm_rate"], ascending=[False, True])
    sm.to_csv(NOVEL / "site_shift_monitors.csv", index=False)

    # ── the recertification cost ────────────────────────────────────────────
    need = min_positives_needed(alpha, delta)
    prevalence = float(lab_test.mean())
    cases_to_label = int(np.ceil(need / prevalence))

    out = {"task": TASK, "alpha": alpha, "delta": delta, "threshold": threshold,
           "temperature": temp, "bootstrap": a.bootstrap,
           "n_scenarios": len(df), "n_breaches": int(breaches.sum()),
           "monitors": summary,
           "recertification": {"malignant_cases_needed": need,
                               "source_prevalence": prevalence,
                               "cases_to_label_at_source_prevalence": cases_to_label},
           "scenarios": df.to_dict(orient="records")}
    with open(NOVEL / "site_shift.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ── report ──────────────────────────────────────────────────────────────
    pd.set_option("display.width", 200)
    print(f"\n{'=' * 78}\nRULE-OUT UNDER SITE SHIFT — {TASK}\n{'=' * 78}")
    print(f"frozen threshold {threshold:.4f}, promise <= {alpha:.0%} of cancers missed\n")

    worst = df.sort_values("miss_rate", ascending=False).head(10)
    print("worst 10 deployments by true miss rate (labels used only to score):")
    print(worst[["scenario", "kind", "n", "cleared", "miss_rate", "breach"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\n{int(breaches.sum())} of {len(df)} simulated deployments break the promise.\n")
    print("would anything have warned you, without labels?")
    print(sm[["monitor", "detected", "missed", "detection_rate",
              "false_alarms", "false_alarm_rate"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nrecertification: {need} labelled malignant cases are needed to rebuild the\n"
          f"bound at a new site — about {cases_to_label} consecutive cases at the "
          f"source prevalence of {prevalence:.1%}.")
    write_figures(df, sm, alpha)
    print(f"\nwritten to results/novel/site_shift*.{{json,csv,png}}")
    return df, sm, out


if __name__ == "__main__":
    main()
