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
performance is not merely poor, it is *confidently wrong*: in the worst pair, the
brain-MRI model assigns a malignant label with ≥90% softmax confidence to **55.9%**
of chest CT slices, and its mean confidence on foreign images (0.876) **exceeds**
its mean confidence on its own test set (0.820). Max-softmax probability, the
default proxy for reliability, is not merely uninformative here but
anti-correlated: as an out-of-distribution detector it reaches AUROC **0.41** for
the brain model, worse than chance.

We then show the problem is cheap to fix. A single frozen-backbone *modality
gate*, trained in ten seconds to answer "which modality is this?", admits an
image to a diagnostic model only when it matches that model's domain. The gate
reduces the mean silent failure rate from **30.6% to 0.0%** and the worst-case
rate from **55.9% to 0.0%**, while accuracy on legitimate inputs moves from
**88.6% to 88.6%** (−0.06 pp) and 93.6% of legitimate images are still admitted.

The contribution is not a new architecture. It is the observation that a routine,
one-line deployment assumption — that users supply the right kind of image —
silently converts a well-behaved classifier into a confident source of false
cancer findings, together with a measurement protocol and a mitigation that costs
one extra forward pass.

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
2. **Evidence that confidence cannot be used as a guard.** Mean confidence on
   foreign inputs can exceed mean confidence on native inputs; max-softmax AUROC
   for separating native from foreign inputs falls below 0.5.
3. **A modality gate that eliminates the failure mode.** A frozen-backbone linear
   probe over the union of training modalities, with a confidence floor, drives
   SFR to zero at a −0.06 pp cost in native accuracy.
4. **A deployed demonstration.** The accompanying application refuses to return a
   diagnosis when the gate rejects the input, rather than silently answering.

---

## 2. Related work

**Out-of-distribution detection.** Detecting inputs that fall outside the training
distribution is a mature research area; max-softmax probability is the classical
baseline, with energy scores, Mahalanobis distance and ensembling as common
improvements. Our contribution is not a new detector. It is the demonstration
that in the specific, practically common case of *modality mismatch in deployed
multi-cancer tools*, the baseline is not merely weak but inverted, and that the
appropriate guard is a supervised modality classifier rather than a generic OOD
score — because the set of acceptable domains is known in advance.

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

*(τ = 0.9; up to 300 test images per pair)*

| Diagnostic model | Input images | Mean confidence | ≥ τ | Confident malignant |
|---|---|---|---|---|
| Brain | Brain (native) | 0.820 | 47.4% | 26.3% |
| Brain | **Lung (foreign)** | **0.876** | 55.9% | **55.9%** |
| Lung | **Brain (foreign)** | 0.624 | 5.3% | **5.3%** |
| Lung | Lung (native) | 0.801 | 43.1% | 25.8% |

The brain model is the stark case. Shown chest CT — an image containing no brain
tissue whatsoever — it returns a confident tumour call for **55.9%** of slices,
more than twice the rate at which it makes confident tumour calls on its own test
set. Its mean confidence is *higher* on foreign data than on native data.

The asymmetry between the two directions is itself informative. The lung model
degrades far more gracefully (5.3%), which suggests that susceptibility is a
property of the individual model and its class structure — a binary
tumour/no-tumour head with a coarse decision surface has nowhere to put an
unfamiliar input except one of two disease-bearing bins — rather than a uniform
constant. This means the failure rate cannot be predicted from published accuracy
and must be measured per model.

### 4.2 Confidence is not a usable guard

| Model | MSP AUROC (native vs foreign) | Mean conf. native | Mean conf. foreign |
|---|---|---|---|
| Brain | **0.411** | 0.820 | 0.876 |
| Lung | 0.776 | 0.801 | 0.624 |

For the brain model the score is **below 0.5**: thresholding on confidence to
reject foreign inputs would preferentially reject *legitimate* images and admit
foreign ones. Any deployment that relies on "we only show results above 90%
confidence" as a safety measure is, for this model, worse off than showing
everything.

### 4.3 The gate

Gate validation accuracy: **99.1%**, trained in 10 seconds on a laptop GPU.

| Metric | No gate | With gate |
|---|---|---|
| Mean silent failure rate | 30.6% | **0.0%** |
| Worst-pair silent failure rate | 55.9% | **0.0%** |
| Foreign inputs admitted | 100% | **0.0%** |
| Accuracy on legitimate inputs | 88.64% | **88.58%** |
| Legitimate inputs admitted | 100% | 93.6% |

The gate rejects every foreign image in the evaluation while admitting 93.6% of
legitimate ones, and accuracy on the images it admits is statistically
indistinguishable from accuracy without it. The 6.4% of legitimate images the
gate turns away are the real cost, and it is a benign one: a rejected valid image
produces "please check the image type", which a user can act on, whereas an
accepted foreign image produces a confident cancer finding, which they cannot.

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
