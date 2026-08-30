# Responsible use of automated food inspection

An image classifier trained on a few hundred photographs is a **decision-support
tool for a trained operator**, not an authority.

## Known limitations
* Judgements come from a single 224×224 RGB frame. Odour, temperature history,
  packaging integrity, internal condition and provenance are all invisible to it.
* Performance degrades on food types, lighting, plating and camera hardware
  unlike the training set. Confidence does not fall reliably when it does —
  a confidently wrong output looks exactly like a confidently right one.
* The model reports correlation with the appearance of spoilage, not the
  presence of pathogens.

## Required practice
* Never use the output as the sole basis for releasing food to a consumer.
* Preserve the image, the probabilities and the model version for every
  automated decision, so a disputed outcome can be reconstructed.
* Monitor the input distribution, not just accuracy — silent drift in what is
  being photographed is the usual failure mode.
* Escalate low-confidence cases to a person rather than forcing a verdict; an
  abstention rate of zero means the confidence thresholds are miscalibrated.
* Communicate uncertainty to the operator in plain language. A probability
  presented without its limitations invites over-trust.
