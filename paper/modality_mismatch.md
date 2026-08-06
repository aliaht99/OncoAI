# Silent Failure Under Modality Mismatch: Multi-Cancer Classifiers Confidently Diagnose Images They Have Never Been Trained On

**Ali Hamza**
Codexa Engineering — research draft

---

## Abstract

Deep-learning cancer classifiers are increasingly deployed as public, single-file
web demonstrations, and increasingly bundled together so that one interface
serves several cancers and several imaging modalities. We report a failure mode
this arrangement creates and which, to our knowledge, is not measured in the
published evaluations of such systems: when a classifier receives an image from
a modality it was never trained on, it does not abstain. It emits a confident
disease label.

We train three independent classifiers — brain MRI, chest CT and dermatoscopy —
and evaluate every model against every modality's held-out test set. Off-diagonal
behaviour is not merely poor, it is *confidently wrong*, and it is strongly
model-dependent: in the worst pair the brain-MRI model assigns a malignant label
with ≥90% softmax confidence to **55.9%** of chest CT slices — more than twice
the rate at which it makes confident tumour calls on its own test set — while the
skin model degrades gracefully to 0.0% on both foreign modalities. Susceptibility
therefore cannot be inferred from reported accuracy and must be measured per
model.

We benchmark five guards on the same data. Max-softmax probability, the default
proxy for reliability, is close to useless for the vulnerable model (AUROC
**0.529**, barely above chance) and admits **68.6%** of foreign images at a
threshold tuned to admit 95% of legitimate ones. Energy behaves similarly
(0.773 / 58.2%). The strongest unsupervised detectors, Mahalanobis and kNN over
penultimate features, reach AUROC 0.959 and 0.926 but still admit **22.1%** and
**19.8%** of foreign images at the same operating point.

We then show that when the set of admissible domains is known in advance — which
it always is in a multi-cancer deployment — a supervised guard dominates all of
them. A frozen-backbone *modality gate*, a linear probe trained in minutes to
answer "which modality is this?", reaches **AUROC 1.000 at 0.0% FPR@95TPR**. It
reduces the mean silent failure rate from **12.6% to 0.0%** and the worst case
from **55.9% to 0.0%**, while admitting **100%** of legitimate images and leaving
native accuracy unchanged at **84.2%**.

The contribution is not a new architecture or a new OOD score. It is the
observation that a routine deployment assumption — that users supply the right
kind of image — silently converts well-behaved classifiers into confident sources
of false cancer findings; a task-specific metric for it; the finding that the
guards most likely to be reached for first are the ones that fail; and the
demonstration that the correct guard for this setting is supervised, cheap, and
already available to anyone who trained the models.

---

## 1. Introduction

The literature on cancer image classification is vast and, for the common public
datasets, effectively saturated. HAM10000, CBIS-DDSM, the standard brain-MRI and
chest-CT collections have each been the subject of hundreds of papers reporting
incremental gains from newer backbones. What has grown much faster than the
accuracy numbers, however, is the ease of *deploying* these models: a trained
checkpoint plus forty lines of Streamlit or Gradio is now a public, indexable
medical tool that anyone can send an image to.

Almost all published evaluation is conducted under an assumption that deployment
immediately violates: that the input image comes from the same modality and organ
as the training data. A test set is, by construction, drawn from the same
distribution as training. The moment a model is exposed behind a file-upload
widget, that guarantee disappears. Nothing prevents a user from uploading a chest
CT to a skin-lesion model — and in a multi-cancer interface with a modality
selector, nothing prevents them from simply leaving the selector on the wrong
setting.

The question we ask is deliberately narrow: **what does a cancer classifier do
when handed an image from the wrong modality?** The answer matters because the
two plausible behaviours have very different consequences. If the model produced
diffuse, low-confidence output, downstream confidence thresholding would catch it
and the problem would be self-limiting. If instead the model produces *confident*
disease labels, then every confidence-based safeguard fails simultaneously and
the error is silent — indistinguishable, at the interface, from a genuine finding.

We find the second behaviour, consistently and strongly.

### Contributions

1. **A cross-modality audit protocol.** We define the *silent failure rate* (SFR)
   — the fraction of foreign-modality inputs assigned a malignant class above a
   confidence threshold — and evaluate it across every (model, modality) pair.
   Unlike generic OOD metrics, SFR counts only the errors that are clinically
   dangerous, and it exposes a wide spread across models (0.0%–55.9%) that
   aggregate AUROC hides.
2. **Evidence that the guards reached for first are the ones that fail.** On the
   most exposed model, MSP (0.529) and Energy (0.508) are indistinguishable from
   chance; across models they admit 68.6% and 58.2% of foreign images at 95% TPR.
   Even Mahalanobis, the best unsupervised detector here, admits 22.1%.
3. **A modality gate that closes the gap.** A frozen-backbone linear probe over
   the union of served modalities reaches AUROC 1.000 at 0.0% FPR@95TPR, driving
   SFR to zero with no measurable cost in native accuracy and no legitimate
   images rejected — because a deployment already knows which modalities it
   serves, and an unsupervised detector does not get to use that.
4. **A deployed demonstration.** The accompanying application refuses to return a
   diagnosis when the gate rejects the input, rather than silently answering.
5. **Two leaky public splits documented.** The brain archive ships duplicate
   images; the popular HAM10000 mirror ships image-level splits in which 72.4% of
   test lesions also appear in train, inflating test accuracy from 80.5% to
   94.4%.

---

## 2. Related work

**Out-of-distribution detection.** Detecting inputs outside the training
distribution is a mature area with a dedicated medical-imaging survey [2], an
established taxonomy of distributional shift, and a settled benchmark suite —
max-softmax probability as the classical baseline, with Energy, Mahalanobis
distance, kNN over features, reconstruction-based scores and ensembling as
standard comparators, evaluated by AUROC and FPR@95TPR. The term *silent failure*
is itself already used in that literature to describe confident errors on OOD
inputs. We claim no new detector and no new metric, and we benchmark against that
suite rather than around it.

Our contribution sits in three gaps that the general framing leaves open. (i)
Most medical OOD work targets *near*-OOD shift — a different scanner, a different
site, a different protocol — because inside a hospital that is the shift that
occurs. Cross-modality exposure is far-OOD and is usually excluded as
clinically implausible, an assumption that a public upload box removes.
(ii) The standard target is a generic in/out decision, whereas the clinically
relevant quantity is narrower: how often a foreign input yields a *confident
malignant* label. A detector can look adequate by AUROC while still leaking a
fifth of foreign images at a usable operating point, which is what we measure.
(iii) The unsupervised framing is the right one when the admissible domain is
open-ended, but a multi-cancer deployment knows exactly which modalities it
serves; we show that using that knowledge closes the gap that the unsupervised
detectors leave open.

**Robustness in medical imaging.** Existing work studies domain shift within a
modality: different scanners, different hospitals, different acquisition
protocols. Cross-*modality* exposure is a categorically larger shift and is
usually treated as out of scope, on the implicit assumption that it cannot occur
in a clinical workflow. That assumption is reasonable inside a PACS. It does not
hold for the public web demonstrations that increasingly accompany published
models.

**Multi-task and multi-organ models.** Shared-trunk models spanning several organs
exist, but are evaluated on aggregate accuracy across tasks. To our knowledge the
question of what a single-task head does when routed the wrong input — the direct
consequence of putting several such heads behind one selector — is not measured.

**Multi-cancer screening as a field direction.** The move towards systems that
screen for several cancers at once is now explicit. A 2026 review of AI-driven
multi-cancer screening [1] surveys achievements across liquid biopsy and imaging
— including a trial reporting a 29% increase in invasive cancer detection at 44%
lower workload — and sets out the field's open problems: under-representation of
early and rare cancers, sample and protocol variability, fairness across
populations, cost, and the need for prospective validation and regulatory
evaluation. Notably, that agenda does not include what happens when a deployed
multi-cancer system receives an input from outside the domain of the model it is
routed to. Our results suggest it should: the more models sit behind one
interface, the more ways an input has to reach the wrong one. The same review
calls for *lightweight, offline-capable* systems suitable for lower-resource
settings, which is the regime the gate proposed here is designed for — a frozen
backbone, a linear head, and one additional forward pass.

[1] *Artificial intelligence-driven multi-cancer screening: Achievements,
challenges, and future prospects.* Intelligent Medicine, 2026.
doi:10.1016/j.imed.2026.02.001

[2] *Out-of-distribution Detection in Medical Image Analysis: A survey.*
arXiv:2404.18279.

---

## 3. Methods

### 3.1 Tasks and data

Three public datasets, one per modality, each used exactly as distributed:

| Task | Modality | Classes | Source |
|---|---|---|---|
| Brain | MRI | tumour / no tumour | Brain MRI Images for Brain Tumor Detection |
| Lung | CT | adenocarcinoma, large cell, squamous cell, normal | Chest CT-Scan images |
| Skin | Dermatoscopy | 7 lesion types | HAM10000 |

Where the source ships a split we inspected it before use, and in two of the
three cases the shipped or naive split leaks.

**Brain.** The archive distributes a nested duplicate copy of every image. We
de-duplicate on filename before building a stratified 70/15/15 split; without
this, identical images appear on both sides.

**Skin.** HAM10000 contains multiple photographs of the same physical lesion
(11,439 images over 6,448 unique `lesion_id` values in the mirror we used). The
splits shipped with this widely-downloaded mirror are drawn at image level, so
the same lesion appears in both train and test: we measured that **72.4% of the
lesions in the shipped test split also occur in its train split**. Training on
those splits gives a test accuracy of **94.4% (macro AUC 0.997)** — comfortably
above the published state of the art for 7-class HAM10000, which is the tell.
We therefore discard the shipped splits and re-partition on `lesion_id`,
stratified by diagnosis, so that every image of a lesion falls in exactly one
set. All skin numbers reported below use the lesion-grouped split. We flag this
because the leaky mirror is popular and its splits are easy to use unmodified.

### 3.2 Diagnostic models

All three tasks use an identical recipe so that differences between them cannot
be attributed to architecture choice: an ImageNet-pretrained ResNet-18 trunk with
a 512-unit bottleneck head, trained in two phases (frozen warm-up, then
fine-tuning with cosine annealing), class-balanced sampling, label smoothing
0.05, and early stopping on validation macro-AUC.

### 3.3 The audit

For every model *m* and every task's test set *d*, we record the max-softmax
confidence and predicted class of every image. With τ the confidence threshold
(τ = 0.9 throughout):

- **confident rate** — fraction of inputs with max-softmax ≥ τ.
- **silent failure rate (SFR)** — for *m* ≠ *d*, the fraction of inputs assigned
  a class in *m*'s malignant set with confidence ≥ τ. On the diagonal this
  quantity is legitimate output; off the diagonal every such prediction is a
  confident cancer call on an image the model cannot interpret.
- **MSP AUROC** — the area under the ROC curve for separating native from foreign
  inputs using max-softmax confidence as the score. 1.0 is a perfect guard;
  0.5 is useless; below 0.5 means confidence is actively misleading.

### 3.4 The modality gate

The gate is one classifier over the union of the training modalities, predicting
which modality an image belongs to. We freeze the ImageNet trunk and train only
the head: the modalities are visually far apart, so a linear probe suffices, and
freezing keeps the gate cheap enough that it is not a deployment burden.

At inference an image is admitted to diagnostic model *m* only if the gate's
argmax equals *m*'s modality **and** the gate's own confidence is at least
τ_gate. The second condition matters: without it, an image belonging to no known
modality is still forced into the nearest bucket. If the input is rejected, the
system returns a refusal rather than a diagnosis.

Note that the gate does not modify, retrain, or even inspect the diagnostic
models. It is strictly a pre-filter, which means it can be added to an existing
deployed system without revalidating the clinical model.

---

## 4. Results

### 4.1 The failure mode

Per-task test performance, for reference (lesion-grouped split for skin):

| Task | Test accuracy | Macro AUC |
|---|---|---|
| Brain | 89.5% | 0.988 |
| Lung | 87.8% | 0.972 |
| Skin | 80.5% | 0.962 |

**Silent failure rate**, τ = 0.9, up to 300 test images per pair. Rows are the
diagnostic model, columns the modality the images actually came from; the
diagonal is legitimate use.

| Model ↓ / Input → | Brain | Lung | Skin |
|---|---|---|---|
| **Brain** | *(native)* | **55.9%** | **14.7%** |
| **Lung** | 5.3% | *(native)* | 0.0% |
| **Skin** | 0.0% | 0.0% | *(native)* |

The brain model is the stark case. Shown chest CT — an image containing no brain
tissue whatsoever — it returns a confident tumour call for **55.9%** of slices,
against 26.3% confident tumour calls on its own test set. Its mean confidence on
those foreign images (0.876) *exceeds* its mean confidence on native data
(0.820).

The asymmetry across the matrix is the more useful finding. The skin model, with
seven classes and a fine-grained decision surface, produces no confident
malignant calls on either foreign modality; the binary brain model, which has
nowhere to place an unfamiliar input except one of two bins — one of which is
"tumour" — is catastrophically exposed. Susceptibility is thus a property of the
individual model and its label structure, not a constant of the method, and it
cannot be predicted from published accuracy. It has to be measured.

### 4.2 Benchmarking the guards

We compare five detectors on the same inputs: max-softmax probability (MSP),
Energy, Mahalanobis and kNN over penultimate features, and the supervised gate.
FPR@95TPR is the operationally relevant number — the proportion of foreign images
still admitted when the threshold is set to admit 95% of legitimate ones.

**Mean across the three models:**

| Detector | AUROC ↑ | FPR@95TPR ↓ |
|---|---|---|
| **Gate (ours)** | **1.000** | **0.000** |
| Mahalanobis | 0.959 | 0.221 |
| kNN | 0.926 | 0.198 |
| MSP | 0.777 | 0.686 |
| Energy | 0.773 | 0.582 |

**Per model, the vulnerable case is where the baselines collapse:**

| Model | MSP | Energy | Mahalanobis | kNN | Gate |
|---|---|---|---|---|---|
| Brain | 0.529 | 0.508 | 0.942 | 0.819 | **1.000** |
| Lung | 0.911 | 0.875 | 0.970 | 0.989 | **1.000** |
| Skin | 0.892 | 0.935 | 0.966 | 0.969 | **1.000** |

*(AUROC; higher is better)*

Two observations. First, the cheap guards fail exactly where they are needed:
for the brain model — the one that actually produces 55.9% confident false
malignancies — MSP scores 0.529 and Energy 0.508, both indistinguishable from
chance. A deployment relying on "we only display results above 90% confidence"
has, for that model, essentially no protection. Second, even the strong
unsupervised detectors are not sufficient in absolute terms: Mahalanobis, the
best of them, still admits **22.1%** of foreign images at a 95% TPR operating
point. For a guard whose failures are confident false cancer findings, a
one-in-five leak is not a safe resting place.

### 4.3 The gate

Gate validation accuracy on 3-way modality classification: **99.94%**, from a
linear probe on a frozen ImageNet backbone.

| Metric | No gate | With gate |
|---|---|---|
| Mean silent failure rate | 12.6% | **0.0%** |
| Worst-pair silent failure rate | 55.9% | **0.0%** |
| Foreign inputs admitted | 100% | **0.0%** |
| Legitimate inputs admitted | 100% | **100%** |
| Accuracy on legitimate inputs | 84.20% | **84.20%** |

With three modalities the gate is exact on this evaluation: every foreign image
rejected, every legitimate image admitted, and native accuracy unchanged to four
decimal places. The reason it can dominate general-purpose OOD scores is not
sophistication but information: an unsupervised detector must infer the boundary
of "normal" from one class of data, whereas a deployment already knows the
complete list of modalities it serves and can simply learn to name them. Where
that list is known, not using it is leaving accuracy on the table.

The honest caveat is that this makes the gate a *closed-set* guard. It is
evaluated against the modalities it was trained on; an input from a fourth,
unseen modality is handled only by the confidence floor, which we discuss in the
limitations.

---

## 5. Discussion

**What this is not.** We do not claim a new detector, a new architecture, or
state-of-the-art accuracy on any of the three datasets; the per-task numbers here
are ordinary. The claim is about evaluation practice: a model can be correct by
every metric currently reported and still behave dangerously the first time it
meets an input outside its modality, and nothing in the standard evaluation
pipeline would reveal this.

**Why it is likely to get worse.** The trend is towards unified interfaces — one
site, one upload box, a dropdown for cancer type. Every additional model behind
one selector adds another way for an input to reach the wrong head, and the
selector is set by the least-informed party in the loop.

**Practical recommendation.** Report SFR alongside accuracy for any model
released with a public inference interface, and gate multi-model deployments on
domain membership rather than on output confidence. Both are inexpensive. The
gate here is a linear probe over frozen features and costs one forward pass.

### Limitations

- Three modalities and one architecture family. Whether the magnitude of the
  effect transfers to transformer backbones or to more modalities is untested,
  though the mechanism — a softmax head with no reject option — is architecture-agnostic.
- The gate is evaluated on *known* foreign modalities. Inputs belonging to no
  medical modality at all (a photograph, a screenshot) are handled only by the
  gate's confidence floor, which we have not stress-tested at scale.
- Public datasets are modest in size and drawn from specific scanners and
  populations; absolute accuracies should be read as demonstrative, not clinical.
- SFR depends on τ. We report τ = 0.9 throughout; the qualitative conclusion is
  stable across reasonable thresholds, but the numbers are not threshold-free.

---

## 6. Conclusion

Cancer classifiers deployed behind an upload box will answer whatever they are
given. Shown an image from the wrong modality they do not hesitate — they return
confident disease labels, at rates up to 55.9%, with confidence values that can
exceed those on genuine inputs and with the standard confidence-based safeguard
performing worse than chance. A supervised modality gate, trained in seconds and
requiring no change to the diagnostic models, removes the failure mode entirely
at a cost of 0.06 percentage points of accuracy. We suggest that any cancer model
shipped with a public interface should report its silent failure rate, and that
gating on domain membership should be the default rather than an afterthought.

---

## Reproducibility

All code, trained models and figures: **https://github.com/aliaht99/OncoAI**

```bash
python src/common/download_data.py          # public datasets, no credentials
python src/common/train.py brain            # and lung, skin
python src/novel/modality_audit.py --tau 0.9
python src/novel/modality_gate.py train
python src/novel/modality_gate.py evaluate
python src/novel/make_figures.py
```

## Ethics statement

This work uses only publicly released, de-identified datasets. It produces no
clinical claim and the accompanying software is explicitly labelled as a research
and education artefact, not a medical device. The purpose of publishing the
failure mode is to make deployed research demonstrations safer, and the
mitigation is released alongside the finding.
