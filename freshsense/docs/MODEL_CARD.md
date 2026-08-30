# Model card — FreshSense spoilage classifier v0.1.0

## Overview

| | |
|---|---|
| **Task** | Binary image classification: `fresh` vs `spoiled` food |
| **Architecture** | ViT-B/16 (ImageNet-1k pretrained), backbone frozen, linear head retrained |
| **Trained parameters** | 1,538 of 86.6 M (the head only) |
| **Input** | One RGB photograph, resized to 224×224, ImageNet-normalised |
| **Output** | A probability per class, then a verdict under the decision policy below |
| **Artefact** | `artifacts/vit_head.pt`, 8.2 KB (head only; the backbone is public weights) |
| **Intended use** | Decision support for a trained food-handling operator |
| **Out of scope** | Any use as the sole basis for releasing food to a consumer |

## Training data

`PD-lab/dataset/…/spoilage_detection`, split 70/15/15.

| Split | fresh | spoiled | total |
|---|---:|---:|---:|
| train | 456 | 142 | 598 |
| val | 97 | 30 | 127 |
| test | 99 | 30* | 129 |

\* The test directory holds 31 spoiled files, but one is `.avif`, which
`torchvision.datasets.ImageFolder` silently skips — it is not in `IMG_EXTENSIONS`.
The image is not corrupt and not logged as skipped; it simply never enters the
loader. Every count reported here is of images actually seen. This is a real
data-pipeline defect, recorded rather than papered over.

The corpus is small, single-source and imbalanced roughly 3:1 toward `fresh`.
A classifier that answers "fresh" unconditionally scores **76% accuracy** on
this distribution, which is why accuracy alone is not reported as a headline.

## Results on the held-out test split

Reproduce with `make eval`. Full output in `artifacts/eval_metrics.json`.

### Raw model (argmax)

| | fresh | spoiled |
|---|---:|---:|
| **recall** | 1.000 | 0.967 |
| **precision** | 0.990 | 1.000 |
| **F1** | 0.995 | 0.983 |

Macro-F1 **0.989**, accuracy **0.992**. Confusion matrix `[[99, 0], [1, 29]]` —
one spoiled item was called fresh.

### Under the deployed decision policy

| outcome | count | share |
|---|---:|---:|
| released as fresh | 96 | 74.4% |
| condemned as spoiled | 31 | 24.0% |
| abstained → manual review | 2 | 1.6% |

**Spoiled items released as fresh: 0.**

The single argmax false negative fell into the abstain band rather than being
released. That is the policy doing precisely the job it was designed for, and it
is the number that matters more than the accuracy figure above.

## Decision policy

Probabilities are not verdicts. The service applies an asymmetric rule, because
missing spoiled food and wasting fresh food are not equally costly:

```
p(spoiled) >= 0.30  ->  SPOILED    (discard)
p(fresh)   >= 0.85  ->  FRESH      (release)
otherwise           ->  UNCERTAIN  (manual review)
```

Condemning needs weak evidence; clearing needs strong evidence. The gap between
them — `p(spoiled)` in [0.15, 0.30) — is where the model is explicitly not
trusted to decide alone.

Both thresholds are configuration (`FRESHSENSE_SPOILED_THRESHOLD`,
`FRESHSENSE_ABSTAIN_THRESHOLD`), and a startup validator rejects any pair that
leaves the abstain band empty, since that silently removes the human from the
loop while still looking like a working config.

**These thresholds are not calibrated.** They encode a risk posture, chosen
deliberately; they were not fitted to a cost matrix, because no cost matrix for
this domain has been agreed. Tuning them is the first thing to do with a real
operator in the room.

## Known limitations

* **One frame, no context.** Odour, temperature history, packaging integrity,
  internal condition and provenance are all invisible to the model.
* **A `fresh` verdict is not a safety clearance.** The most dangerous pathogens
  — *Salmonella*, *Listeria*, *Campylobacter*, pathogenic *E. coli* — produce no
  visible change. The service says so in every fresh explanation, enforced by a
  guardrail rather than by prompt instruction alone.
* **Small, narrow training set.** 598 training images from one source. Expect
  degradation on unfamiliar foods, lighting, plating or camera hardware, and
  expect confidence *not* to fall reliably when that happens.
* **Uncalibrated probabilities.** The softmax outputs have not been checked
  against observed frequencies; a reported 0.9 does not mean 90% of such cases
  are spoiled. Temperature scaling on a proper calibration set is the next step.
* **Frozen backbone.** Features are ImageNet's, not food's. Unfreezing the last
  blocks is the obvious accuracy lever and was not pulled here, because with 598
  images it would mostly buy overfitting.
* **The abstention rate is 1.6% on in-distribution data**, which is low. On
  genuinely unfamiliar inputs it should be far higher; that it will not rise on
  its own is the main reason this needs a drift monitor, not a dashboard.

## Ethical and operational considerations

* **Failure asymmetry is a policy decision, not a modelling one.** It is written
  in one function, `freshsense.vision.predict.decide`, and tested there, so the
  people who own the risk can read and argue with it.
* **The explanation cannot overrule the verdict.** The LLM narrates; it never
  re-decides. Guardrails block tasting advice, safety assurances, absolute
  certainty and medical advice, and drop any explanation citing a source that
  was not retrieved.
* **Auditability.** Every assessment carries a request id, the probabilities,
  the policy note, the retrieved citations, the guardrail notes and a per-node
  execution trace. A disputed decision can be reconstructed.
* **Food waste has a cost too.** The asymmetry deliberately discards some fresh
  food. At a 24% condemnation rate on this test set, that trade-off should be
  revisited with real operational data.

## Reproducing

```bash
make train    # fine-tune on PD-lab's train/val splits, seeded (1337)
make eval     # metrics on the held-out test split
make test-all # includes a real 2-epoch training run
```

Training selects the best epoch on **spoiled-recall**, tie-broken on macro-F1 —
not on accuracy, which on a 3:1 split rewards the failure mode this domain
cannot tolerate.
