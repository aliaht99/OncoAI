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

> **55.9%** of chest CT slices are labelled **"tumour" with ≥90% confidence.**
> The model's mean confidence on these foreign images (**0.876**) is *higher* than on
> its own test set (**0.820**).

The usual safety net does not catch this. Using max-softmax confidence to separate
legitimate inputs from foreign ones gives **AUROC 0.41** for that model — *worse than
random*. Thresholding on confidence would preferentially reject the real images and
admit the wrong ones.

This matters because every public cancer demo is a file-upload box, and multi-cancer
tools add a dropdown that the least-informed person in the loop is expected to set
correctly.

## The fix

A **modality gate**: one small classifier that answers *"which modality is this?"*
before any diagnostic model is allowed to see the image. Frozen ImageNet trunk,
linear head, **trained in 10 seconds**, 99.1% accurate. If the gate's answer does not
match the selected task — or the gate itself is unsure — the system **refuses** instead
of diagnosing.

| | No gate | With gate |
|---|---|---|
| Mean silent failure rate | 30.6% | **0.0%** |
| Worst-pair silent failure rate | 55.9% | **0.0%** |
| Foreign images admitted | 100% | **0.0%** |
| Accuracy on legitimate inputs | 88.64% | **88.58%** |
| Legitimate images still admitted | 100% | 93.6% |

Zero silent failures, at a cost of **0.06 percentage points** of accuracy. The gate
never touches the diagnostic models — it is a pre-filter, so it can be bolted onto an
already-validated system.

📄 Full write-up: [`paper/modality_mismatch.md`](paper/modality_mismatch.md)

---

## Per-task performance

Ordinary transfer-learning baselines — the point of this repo is the study above,
not a leaderboard entry.

| Task | Modality | Classes | Test accuracy | Macro AUC |
|---|---|---|---|---|
| **Brain** | MRI | tumour / no tumour | 89.5% | **0.988** |
| **Lung** | CT | 3 carcinoma subtypes + normal | 87.8% | **0.972** |
| **Skin** | Dermatoscopy | 7 lesion types | see `results/skin/` | — |

Lung, per class: *normal* is separated essentially perfectly (sensitivity 98.2%,
specificity 100%, AUC 0.9998); the three carcinoma subtypes are harder to tell apart
from each other (AUC 0.943–0.976), which is the clinically expected pattern.

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
python src/novel/make_figures.py
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
│       └── make_figures.py     paper figures
├── models/                     trained weights + metrics JSON
├── results/
│   ├── brain|lung|skin/        per-task figures and metrics
│   └── novel/                  audit + gate results and figures
├── paper/modality_mismatch.md  the write-up
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
- Outputs are **not calibrated**: a "90%" score is not a 90% chance of cancer.
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
