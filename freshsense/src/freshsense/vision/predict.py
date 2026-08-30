"""Inference and the decision policy that sits on top of it.

The classifier returns probabilities; *policy* turns them into a verdict. Those
are deliberately separate, because the policy is the part a food-safety owner
gets to argue with, and it is asymmetric on purpose — see `decide`.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from freshsense.config import Settings, get_settings
from freshsense.domain import Action, Prediction, Verdict
from freshsense.vision.model import eval_transform, load_classifier_head

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 10 * 1024 * 1024


class InvalidImageError(ValueError):
    """Raised for anything we could not decode as an image."""


def load_image(data: bytes) -> Image.Image:
    """Decode untrusted bytes into RGB, with a size cap and a decode-bomb guard."""
    if not data:
        raise InvalidImageError("empty upload")
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidImageError(f"image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit")
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()  # cheap structural check before we commit to decoding
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"could not decode image: {exc}") from exc
    return image


def decide(probabilities: dict[str, float], settings: Settings) -> Prediction:
    """Turn class probabilities into a verdict under an asymmetric safety policy.

    Missing spoiled food is materially worse than wasting fresh food, so the two
    directions get different evidentiary bars:

      * p(spoiled) >= spoiled_threshold          -> SPOILED     (condemn on weak evidence)
      * p(fresh)   >= abstain_threshold          -> FRESH       (clear only on strong evidence)
      * otherwise                                -> UNCERTAIN   (a human looks at it)

    The band between the two is where the model is not trusted to decide alone.
    """
    p_spoiled = probabilities.get("spoiled", 0.0)
    p_fresh = probabilities.get("fresh", 0.0)

    if p_spoiled >= settings.spoiled_threshold:
        return Prediction(
            verdict=Verdict.SPOILED,
            confidence=p_spoiled,
            probabilities=probabilities,
            policy_note=(
                f"p(spoiled)={p_spoiled:.2f} at or above the {settings.spoiled_threshold:.2f} "
                f"safety threshold; spoilage is condemned on weak evidence by design."
            ),
        )
    if p_fresh >= settings.abstain_threshold:
        return Prediction(
            verdict=Verdict.FRESH,
            confidence=p_fresh,
            probabilities=probabilities,
            policy_note=(
                f"p(fresh)={p_fresh:.2f} clears the {settings.abstain_threshold:.2f} bar "
                f"required to release an item."
            ),
        )
    return Prediction(
        verdict=Verdict.UNCERTAIN,
        confidence=max(p_fresh, p_spoiled),
        probabilities=probabilities,
        policy_note=(
            f"p(spoiled)={p_spoiled:.2f} sits in the abstain band "
            f"[{1 - settings.abstain_threshold:.2f}, {settings.spoiled_threshold:.2f}); "
            f"the model is not confident enough to decide alone."
        ),
    )


def action_for(verdict: Verdict) -> Action:
    return {
        Verdict.FRESH: Action.RELEASE,
        Verdict.SPOILED: Action.DISCARD,
        Verdict.UNCERTAIN: Action.MANUAL_REVIEW,
    }[verdict]


class Classifier:
    """Lazily-loaded ViT wrapper. Construct cheaply, pay for weights on first use."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._class_names: tuple[str, ...] = self.settings.class_names
        self._transform = eval_transform()

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._class_names

    def load(self) -> Classifier:
        if self._model is None:
            self._model, self._class_names = load_classifier_head(
                Path(self.settings.checkpoint), device=self.settings.device
            )
            logger.info(
                "classifier loaded",
                extra={"checkpoint": str(self.settings.checkpoint), "classes": self._class_names},
            )
        return self

    def probabilities(self, image: Image.Image) -> dict[str, float]:
        self.load()
        tensor = self._transform(image).unsqueeze(0).to(self.settings.device)
        with torch.inference_mode():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
        return {name: float(probs[i]) for i, name in enumerate(self._class_names)}

    def predict(self, image: Image.Image) -> Prediction:
        return decide(self.probabilities(image), self.settings)

    def predict_bytes(self, data: bytes) -> Prediction:
        return self.predict(load_image(data))
