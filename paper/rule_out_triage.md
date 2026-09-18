# What Can This Model Safely Take Off the Worklist? Bounded Rule-Out Triage for Cancer Classifiers

**Ali Hamza**
Codexa Engineering — research draft

---

## Abstract

Imaging volume is growing faster than the workforce reading it, and the use
case that actually motivates buying diagnostic AI is not diagnosis but
**triage**: safely removing the clearly-normal studies from a human worklist. A
prospective trial of AI-triaged screening mammography in 31,301 women cut
radiologist workload by 63.6% without reducing cancer detection. Delivering that
requires something an accuracy figure cannot provide — a defensible answer to
"what exactly are you promising about the cancers you clear?"

We show that the softmax score of an otherwise well-behaved cancer classifier
cannot answer it, and that the standard repair, temperature scaling, barely
helps: on our 7-class dermatoscopy model it moves expected calibration error
only from 0.050 to 0.044, leaving the model visibly overconfident in the
0.5–0.8 risk band. We therefore make the rule-out threshold itself the object
of the guarantee, choosing it by a Clopper-Pearson bound walked over thresholds
in a fixed sequence. The result is distribution-free and finite-sample: it does
not assume the calibration is correct, only that deployment cases are
exchangeable with calibration cases.

On the skin task this removes **38.3% of the worklist at 97.4% sensitivity**,
promising at most 4.96% of cancers missed and delivering 2.57%. Across 50
independent calibration draws the bound is violated **0/50** times.

Two negative results matter as much as that number. First, a **sample-size
wall**: promising a 5% miss rate at 95% confidence requires at least 59
malignant calibration cases *before any model quality is considered*, so our
brain (15) and lung (41) tasks cannot support the promise at all, and the
correct output for them is a refusal to clear anything. Second, an
**exchangeability wall**: on the lung task, whose splits ship as fixed folders
rather than being drawn at random, a threshold promising ≤14.6% misses delivered
**31.9%** — the guarantee failed silently. We show this specific failure is
detectable *without labels* by a two-sample test on the score distribution
(KS p = 2.4 × 10⁻⁷ for lung; p = 0.42 for the exchangeable skin task), and make
that check a precondition for reporting a threshold at all.

The contribution is a deployment-shaped reframing: not "how accurate is this
model" but "how much work can it remove, under what promise, and when is that
promise void" — together with the finding that for two of our three tasks the
honest answer is *none*.

---

## 1. The question a department actually asks

Published cancer-classifier evaluations optimise a number — accuracy, AUC —
that no radiology department can act on. AUC is threshold-free, which is
precisely the problem: a deployment *is* a threshold. What a department needs to
know before adopting a tool is a pair of numbers with a promise attached:

> *"It will take X% of studies off your list, and of the cancers in the studies
> it takes, it will miss no more than Y%."*

Neither number is recoverable from an AUC. And the obvious way to produce them —
pick a softmax cut-off, measure sensitivity on the test set — produces a point
estimate with no guarantee, on data the threshold was chosen using.

This paper builds the pair of numbers properly, and reports honestly where it
cannot be built.

## 2. Why calibration is necessary but not sufficient

The first obstacle is that a classifier's softmax output is not a probability.
We define a per-case malignancy risk as the softmax mass on the classes that
require a human (for skin: melanoma, basal cell carcinoma, actinic keratoses),
and fit a single temperature on a held-out 30% of the validation split.

| Task | T | ECE raw → scaled | Brier raw → scaled |
|---|---|---|---|
| Brain | 0.726 | 0.131 → 0.137 | 0.068 → 0.059 |
| Lung | 1.061 | 0.039 → 0.043 | 0.007 → 0.007 |
| Skin | 1.076 | 0.050 → 0.044 | 0.104 → 0.103 |

Temperature scaling does very little here, and on two of three tasks it makes
ECE slightly *worse*. The reliability diagram for skin shows why: the
miscalibration is not a uniform sharpness error that one scalar can absorb, but
a localised one — cases scored around 0.65 are malignant only ~40% of the time.
A single temperature cannot bend that region into line without damaging others.

The practical consequence is that a rule-out threshold must not *depend* on the
calibration being right. We keep the temperature because it makes the number
shown to a user more honest, and then choose the threshold by a method that
would remain valid even if the temperature were wrong.

## 3. A threshold with a bound

Let the calibration set contain `n` malignant cases with risks `s₁ … sₙ`.
Thresholding at `t` misses `k(t) = #{i : sᵢ < t}` of them. We want

> P(missed | malignant) ≤ α, with confidence 1 − δ.

Since `k(t)` is non-decreasing in `t`, so is any upper confidence bound on the
miss rate. We therefore compute the exact Clopper-Pearson upper limit
`U(k, n, δ)` and walk candidate thresholds upward from zero, stopping at the
first one for which `U > α`. Walking a monotone family in a fixed order is
fixed-sequence testing, which controls the family-wise error rate without
spending δ on a multiplicity correction. The bound is exact-binomial rather
than normal-approximate, because the regime that matters is exactly the one
where the normal approximation is optimistic.

This immediately yields a **sample-size wall**. With zero observed misses the
bound is `1 − δ^(1/n)`, so no threshold above zero is defensible unless

> n ≥ log δ / log(1 − α),

which is **59** malignant calibration cases for α = 5%, δ = 5%. This depends
only on the promise, not on the model: a perfect classifier calibrated on 40
cancers still cannot promise a 5% miss rate.

## 4. Results

### 4.1 Skin — the guarantee holds

With 262 malignant calibration cases the procedure selects a threshold at
P(malignant) < 0.0338, conceding 7 misses in calibration for a bound of 4.96%.

| | value |
|---|---|
| Promised miss rate | ≤ 4.96% (95% confidence) |
| Delivered on test | **2.57%** (10 / 389) |
| Sensitivity of the filter | **97.4%** |
| Worklist removed | **38.3%** (650 / 1699) |
| Malignant prevalence, before → after | 22.9% → 36.1% |

The prevalence shift is the point of the exercise: the radiologist's remaining
list is enriched 1.6-fold in disease.

A single calibration draw proves nothing, so we re-drew the split 50 times:

| | across 50 draws |
|---|---|
| Violations of the 5% bound | **0 / 50** |
| Test miss rate | mean 2.22%, max 3.08% |
| Worklist removed | mean 34.8%, range 26.1–44.3% |

The realised miss rate sits well under the bound, as it should — Clopper-Pearson
is conservative, and buying a guarantee means paying for it in workload.

### 4.2 The exchange rate between promise and workload

| Tolerated α | Worklist removed | Cancers missed (test) |
|---|---|---|
| 1% | *infeasible* | — |
| 2% | 2.1% | 0.0% |
| 5% | 38.3% | 2.6% |
| 10% | 56.6% | 8.0% |
| 15% | 62.8% | 12.6% |
| 20% | 66.2% | 15.4% |
| 30% | 71.8% | 21.3% |

The curve is steep at the safe end and flattens quickly: going from a 5% to a
10% tolerated miss rate buys 18 points of workload, while going from 20% to 30%
buys only 6. Whatever α a department chooses, it is choosing on this curve, and
the curve is the artefact that should be published alongside a model.

### 4.3 Brain and lung — the honest refusal

| Task | Malignant calibration cases | Needed for α = 5% | Tightest provable bound | Threshold offered |
|---|---|---|---|---|
| Brain | 15 | 59 | 18.1% | **none** |
| Lung | 41 | 59 | 7.0% | **none** |

Both tasks have respectable headline metrics — brain AUC 0.988, lung AUC 0.972 —
and neither can support a rule-out promise. The limit is the size of the
validation split, not the quality of the model, and no amount of further
training changes it. The system's correct behaviour is to send every case to a
human and say why.

### 4.4 The exchangeability wall

The bound is distribution-free but not assumption-free: it requires deployment
cases to be exchangeable with calibration cases. The lung task violates this,
because its public splits ship as fixed folders rather than being drawn at
random. Forcing a threshold through at α = 15% exposes the consequence:

| | promised | delivered |
|---|---|---|
| Lung, α = 15% | ≤ 14.6% missed | **31.9% missed** |

The bound was not wrong; its precondition was. The score distributions make the
violation plain — the 5th percentile of malignant risk is 0.966 on the
validation folder and 0.820 on the test folder, so the model is markedly more
confident on one than the other.

Crucially this is detectable **without labels**, which is the situation at any
new site on day one. A two-sample Kolmogorov-Smirnov test on the malignancy
score alone separates the three tasks cleanly:

| Task | KS | p | verdict |
|---|---|---|---|
| Lung | 0.364 | 2.4 × 10⁻⁷ | **shift** |
| Brain | 0.132 | 0.90 | consistent |
| Skin | 0.030 | 0.42 | consistent |

We make this check a precondition: when it fires, the tool reports the
guarantee as void and refuses to clear rather than quoting a bound that does not
apply. It does not prove exchangeability — no label-free test can — but it
catches shifts large enough to break the promise, and it runs on unlabelled
data.

## 5. Relation to the modality gate

The companion study in `modality_mismatch.md` addresses a failure of
*commission*: a foreign image receives a confident malignant label. This one
addresses a failure of *omission*, which is the failure mode a triage deployment
creates — a case that is silently cleared and never seen. The two guards
compose and are needed together: the gate decides whether the model may speak at
all, and the threshold decides whether its answer is trustworthy enough to act
on without a human. Neither subsumes the other, and both sit outside the
diagnostic model, so either can be added to a system that has already been
validated.

## 6. Limitations

- The bound is conditional on exchangeability, which our KS test can refute but
  never confirm. A shift that preserves the marginal score distribution while
  changing the risk-label relationship would pass it.
- Our "deployment" set is a held-out split, not a different hospital. The lung
  result is a natural experiment in split-induced shift, not a site-transfer
  study; the real magnitude of site shift is typically larger.
- Rule-out is evaluated at image level. Real screening decisions are made per
  patient across multiple views and priors, and a per-patient bound would need
  per-patient calibration units.
- Clopper-Pearson is conservative; a tighter bound would buy workload at the
  same promise. We prefer the conservative one because the cost of the two
  errors is not symmetric.
- None of this has been validated prospectively, and none of it makes the system
  a medical device.

## 7. Conclusion

A cancer classifier becomes useful to a department at the moment it can say how
much work it removes and what it promises about what it removes. We built that
statement for three models and could honestly make it for one, at 38.3% workload
removed and a miss rate provably under 5%. For the other two the finding is that
the promise cannot be made — not because the models are bad, but because a
guarantee has a sample-size cost and an exchangeability precondition that
published accuracy figures never surface. Reporting those two walls is the part
most likely to transfer.
