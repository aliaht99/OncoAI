# OncoAI — Multi-Cancer Detection, and the Failure Nobody Measures

> **Brain MRI · Lung CT · Skin lesions** — three interpretable classifiers behind one
> interface, plus a study of what happens when you give a cancer model the wrong
> kind of image.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> ⚕️ **Research and education only.** Not a medical device. Not cleared for clinical
> or diagnostic use. Cannot diagnose cancer. Do not upload identifiable patient data.

---

## The finding

Train a brain-MRI tumour classifier. Show it a **chest CT** — an image with no brain
tissue in it at all. It does not abstain, and it does not hedge:

> **55.9%** of chest CT slices are labelled **"tumour" with ≥90% confidence** — more
> than twice the rate at which it makes confident tumour calls on its *own* test set.
> Its mean confidence on those foreign images (**0.876**) is *higher* than on its own
> data (**0.820**).

**Silent failure rate** — how often a foreign image gets a confident malignant label.
Rows = model, columns = where the images actually came from:

| Model ↓ / Input → | Brain | Lung | Skin |
|---|---|---|---|
| **Brain** | *native* | **55.9%** | **14.7%** |
| **Lung** | 5.3% | *native* | 0.0% |
| **Skin** | 0.0% | 0.0% | *native* |

Exposure is wildly model-dependent: the 7-class skin model shrugs it off, the binary
brain model — which has nowhere to put an unfamiliar image except "tumour" or
"no tumour" — is catastrophically exposed. **You cannot predict this from published
accuracy. It has to be measured.**

## The guards that don't work

We benchmarked the standard OOD detectors on the same data. FPR@95TPR = how many
foreign images still get through when you admit 95% of legitimate ones.

| Detector | AUROC ↑ | FPR@95TPR ↓ |
|---|---|---|
| **Modality gate (ours)** | **1.000** | **0.0%** |
| Mahalanobis | 0.959 | 22.1% |
| kNN | 0.926 | 19.8% |
| MSP (max softmax) | 0.777 | 68.6% |
| Energy | 0.773 | 58.2% |

The cheap guards fail exactly where they're needed. On the *brain* model — the one
actually producing 55.9% false malignancies — **MSP scores 0.529 and Energy 0.508**,
both indistinguishable from a coin flip. "We only show results above 90% confidence"
is not a safety measure here. And even Mahalanobis, the best unsupervised detector,
still lets **1 in 5** foreign images through.

## The fix

A **modality gate**: one small classifier that answers *"which modality is this?"*
before any diagnostic model sees the image. Frozen ImageNet trunk, linear head,
**99.94%** accurate. If its answer doesn't match the selected task — or it isn't
confident — the system **refuses** instead of diagnosing.

| | No gate | With gate |
|---|---|---|
| Mean silent failure rate | 12.6% | **0.0%** |
| Worst-pair silent failure rate | 55.9% | **0.0%** |
| Foreign images admitted | 100% | **0.0%** |
| Legitimate images admitted | 100% | **100%** |
| Accuracy on legitimate inputs | 84.20% | **84.20%** |

Zero silent failures, zero measurable cost. It beats every unsupervised detector for a
simple reason — not sophistication, but *information*: an unsupervised score has to
infer the boundary of "normal" from one class of data, while a deployment already
knows the exact list of modalities it serves. The gate never touches the diagnostic
models, so it can be bolted onto an already-validated system.

📄 Full write-up: [`paper/modality_mismatch.md`](paper/modality_mismatch.md)

---

## The second question: what can it take off the worklist?

Refusing bad inputs makes the system safe. It does not make it *useful*. The
reason a department buys diagnostic AI is not diagnosis — it is **triage**:
removing the clearly-normal studies from a human worklist. That needs a number
no accuracy figure contains.

> *"It removes X% of your list, and of the cancers in what it removes, it misses
> no more than Y%."*

Softmax cannot supply it, and temperature scaling barely helps — on skin it moves
ECE only **0.050 → 0.044**, because the miscalibration is localised (cases scored
~0.65 are malignant ~40% of the time), not a uniform sharpness error one scalar
can absorb. So we stop relying on the score being a probability and put the
guarantee on the **threshold** instead: a Clopper-Pearson upper bound walked over
thresholds in a fixed sequence. Distribution-free, finite-sample, and valid even
if the calibration is wrong.

**Skin — the promise holds:**

| | |
|---|---|
| Promised miss rate | ≤ 4.96% (95% confidence) |
| Delivered on test | **2.57%** (10/389) — sensitivity **97.4%** |
| Worklist removed | **38.3%** (650/1699) |
| Prevalence, before → after | 22.9% → **36.1%** |
| Violations across 50 calibration draws | **0/50** |

What a stricter promise costs:

| Tolerated miss rate | 2% | 5% | 10% | 15% | 20% | 30% |
|---|---|---|---|---|---|---|
| Worklist removed | 2.1% | **38.3%** | 56.6% | 62.8% | 66.2% | 71.8% |
| Cancers missed (test) | 0.0% | 2.6% | 8.0% | 12.6% | 15.4% | 21.3% |

### Two walls, and both are reported instead of papered over

**A sample-size wall.** Promising a 5% miss rate at 95% confidence needs at least
**59** malignant calibration cases — `n ≥ log δ / log(1−α)` — *before model
quality enters the argument*. Brain has 15 and lung has 41, so despite AUCs of
0.988 and 0.972 neither can support the promise, and the tool clears nothing and
says why. A perfect classifier calibrated on 40 cancers still cannot promise 5%.

**An exchangeability wall.** The bound assumes deployment cases are exchangeable
with calibration cases. Lung's public splits ship as fixed folders, not random
draws, and it shows: a threshold promising **≤14.6%** misses delivered **31.9%**.
The bound wasn't wrong — its precondition was. That specific failure is
detectable **without labels**, which is the situation at any new site on day one:

| Task | KS | p | verdict |
|---|---|---|---|
| Lung | 0.364 | 2.4 × 10⁻⁷ | **shift — guarantee void** |
| Brain | 0.132 | 0.90 | consistent |
| Skin | 0.030 | 0.42 | consistent |

The check is a precondition, not a footnote: when it fires the app reports the
guarantee as void and refuses to auto-clear.

The gate stops a failure of *commission* — a confident label on an image the
model should never have seen. This stops a failure of *omission* — a case
silently cleared and never read. A triage deployment creates the second one, so
both guards are needed, and neither touches the diagnostic model.

📄 Full write-up: [`paper/rule_out_triage.md`](paper/rule_out_triage.md)

---

## Per-task performance

Ordinary transfer-learning baselines — the point of this repo is the study above,
not a leaderboard entry.

| Task | Modality | Classes | Test accuracy | Macro AUC |
|---|---|---|---|---|
| **Brain** | MRI | tumour / no tumour | 89.5% | **0.988** |
| **Lung** | CT | 3 carcinoma subtypes + normal | 87.8% | **0.972** |
| **Skin** | Dermatoscopy | 7 lesion types | 80.5% | **0.962** |

Lung, per class: *normal* is separated essentially perfectly (sensitivity 98.2%,
specificity 100%, AUC 0.9998); the three carcinoma subtypes are harder to tell apart
from each other (AUC 0.943–0.976), which is the clinically expected pattern.

### ⚠️ Two leaky public splits, fixed

Both fixes lower the headline numbers. That is the point.

- **Brain** — the archive ships a nested duplicate copy of every image. Split naively
  and the same image lands on both sides. We de-duplicate on filename first.
- **Skin** — HAM10000 contains several photographs of the *same lesion*, and the
  splits shipped with the popular HF mirror are drawn at image level:
  **72.4% of the test lesions also appear in train.** Training on those splits gives
  **94.4% accuracy / 0.997 AUC** — comfortably above published state of the art for
  7-class HAM10000, which is the tell. We re-split on `lesion_id`, which gives the
  honest **80.5% / 0.962** above.

If you benchmark against this mirror's default splits, your numbers are not
comparable to the literature.

---

## Try it

```bash
git clone https://github.com/aliaht99/OncoAI.git
cd OncoAI
pip install -r requirements.txt
./run.sh
```

Bundled demo images mean you need no dataset to see it work — pick one from
`demo_images/` in the sidebar. Then deliberately feed a **lung CT to the brain model**
and watch the gate refuse.

---

## Reproduce everything

```bash
# 1. Public datasets — no account, no API key
python src/common/download_data.py                 # or: brain lung skin

# 2. Train the three diagnostic models
python src/common/train.py brain
python src/common/train.py lung
python src/common/train.py skin

# 3. Per-task evaluation: metrics, ROC, confusion matrix, Grad-CAM
python src/common/evaluate.py brain

# 4. The study
python src/novel/modality_audit.py --tau 0.9       # measure the failure
python src/novel/modality_gate.py train            # build the guard
python src/novel/modality_gate.py evaluate         # measure the fix
python src/novel/ood_baselines.py                  # vs MSP / Energy / Mahalanobis / kNN
python src/novel/make_figures.py

# 5. the triage study
python src/novel/triage.py                         # all three tasks
python src/novel/triage.py --task skin --trials 50 # re-draw the calibration split
python src/novel/triage.py --alpha 0.10            # a looser promise
```

---

## Layout

```
OncoAI/
├── app.py                      Streamlit app — all three tasks + live gate
├── run.sh                      one-command launcher
├── src/
│   ├── common/
│   │   ├── download_data.py    fetch the three public datasets
│   │   ├── data.py             per-task dataset builders
│   │   ├── model.py            shared backbone + head
│   │   ├── train.py            two-phase transfer learning
│   │   ├── evaluate.py         metrics, ROC, confusion, Grad-CAM
│   │   └── make_demo_images.py sample images for the app
│   └── novel/
│       ├── modality_audit.py   cross-modality silent-failure audit
│       ├── modality_gate.py    the gate: train + evaluate
│       ├── ood_baselines.py     MSP / Energy / Mahalanobis / kNN comparison
│       ├── triage.py           calibration + bounded rule-out threshold
│       └── make_figures.py     paper figures
├── models/                     trained weights + metrics JSON
├── results/
│   ├── brain|lung|skin/        per-task figures and metrics
│   └── novel/                  audit + gate results and figures
├── paper/
│   ├── modality_mismatch.md    write-up: silent failure + the gate
│   └── rule_out_triage.md      write-up: bounded worklist reduction
└── demo_images/                sample images, no dataset needed
```

---

## Datasets

All public, all fetched by the script above, none redistributed here.

| Task | Source |
|---|---|
| Brain | [Brain MRI Images for Brain Tumor Detection](https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection) |
| Lung | [Chest CT-Scan images](https://huggingface.co/datasets/dorsar/lung-cancer) |
| Skin | [HAM10000](https://huggingface.co/datasets/marmal88/skin_cancer) |

---

## Honest limitations

- These are **image-level classifiers**, not detectors. They do not localise or stage
  disease; Grad-CAM is an explanation aid, not a segmentation.
- Datasets are modest and come from specific scanners and populations. Accuracy on
  your images will very likely be **lower** than the numbers above — that gap is the
  norm in medical imaging.
- Raw outputs are **not calibrated** — a "90%" score is not a 90% chance of
  cancer, and temperature scaling only partly fixes it (see the triage study).
  The rule-out bound is deliberately built not to depend on calibration.
- The rule-out guarantee holds **only under exchangeability**, which the KS check
  can refute but never confirm. A shift that preserves the score distribution
  while changing the risk-label relationship would pass it.
- Rule-out is evaluated at **image level**. Real screening decides per patient,
  across views and priors, and would need per-patient calibration units.
- Nothing here has been validated prospectively or reviewed by any regulator.
- The gate is tested against *known* foreign modalities. Inputs belonging to no
  medical modality at all are handled only by its confidence floor.

---

## Sibling project

**[MammoAI](https://github.com/aliaht99/MammoAI)** — the breast-cancer counterpart: a
deeper multi-stage system with clinical-feature fusion, SHAP explanations and
calibrated ensembling.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Ali Hamza** — [github.com/aliaht99](https://github.com/aliaht99)
