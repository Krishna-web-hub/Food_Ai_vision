"""The FreshSense triage workflow, expressed as a state graph.

    ingest ─► classify ─► retrieve ─┬─(model has a signal)──► explain ─┐
                                    └─(model is guessing)──► cheap ───┴─► guard ─► finalise

Each node is a small function of state, so any of them can be tested alone and
the whole path is traced.

Two decisions are worth calling out:

* **Retrieval is verdict-conditional.** A spoiled item needs disposal procedure,
  a fresh one needs the limits of the clearance, an abstention needs escalation
  policy. Asking the knowledge base the same question regardless of outcome is
  how RAG systems end up citing irrelevant passages.
* **The LLM is skipped when the classifier is near-uniform.** If the model is
  effectively coin-flipping, a fluent generated narrative adds authority the
  evidence does not support — and costs a token spend to do it. That case gets
  the fixed escalation text instead.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from freshsense.agent.graph import END, StateGraph
from freshsense.config import Settings, get_settings
from freshsense.domain import Assessment, Explanation, Prediction, Verdict
from freshsense.genai import guardrails
from freshsense.genai.llm import LLMClient, LLMError, OfflineClient, build_llm_client
from freshsense.genai.rag import KnowledgeBase
from freshsense.logging_setup import new_request_id
from freshsense.vision.predict import Classifier, action_for, load_image

logger = logging.getLogger(__name__)

# Verdict-specific retrieval queries. A spoiled item needs disposal procedure; a
# fresh one needs the limits of the clearance; an abstention needs escalation.
_QUERIES = {
    Verdict.SPOILED: (
        "visible mould discolouration slime spoilage indicators; isolate bag discard "
        "sanitise surfaces after handling spoiled food"
    ),
    Verdict.FRESH: (
        "limits of visual inspection; pathogens invisible without spoilage signs; "
        "use-by date and refrigeration temperature control before use"
    ),
    Verdict.UNCERTAIN: (
        "uncertain assessment escalate to human inspector; responsible use of automated "
        "inspection; abstention and low confidence handling"
    ),
}


class TriageAgent:
    """Composes the classifier, the knowledge base and the LLM into one assessment."""

    def __init__(
        self,
        classifier: Classifier | None = None,
        knowledge_base: KnowledgeBase | None = None,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.classifier = classifier or Classifier(self.settings)
        self.kb = knowledge_base or KnowledgeBase(self.settings.kb_dir)
        self.llm = llm or build_llm_client(self.settings)
        self._offline = OfflineClient()
        self.graph = self._build().compile()

    # --- nodes ---------------------------------------------------------------

    def _ingest(self, state: dict[str, Any]) -> dict[str, Any]:
        data = state["image_bytes"]
        return {"image": load_image(data), "image_bytes_len": len(data)}

    def _classify(self, state: dict[str, Any]) -> dict[str, Any]:
        prediction = self.classifier.predict(state["image"])
        logger.info(
            "classified",
            extra={
                "verdict": prediction.verdict.value,
                "confidence": round(prediction.confidence, 4),
            },
        )
        return {"prediction": prediction}

    def _retrieve(self, state: dict[str, Any]) -> dict[str, Any]:
        prediction: Prediction = state["prediction"]
        query = _QUERIES[prediction.verdict]
        if hint := state.get("food_hint"):
            query = f"{hint} {query}"
        citations = tuple(self.kb.search(query, top_k=self.settings.rag_top_k))
        logger.info(
            "retrieved context",
            extra={"n": len(citations), "ids": [c.source_id for c in citations]},
        )
        return {"citations": citations}

    def _explain(self, state: dict[str, Any]) -> dict[str, Any]:
        prediction: Prediction = state["prediction"]
        citations = state["citations"]
        hint = state.get("food_hint")

        fallback = self._offline.explain(prediction, citations, hint)
        try:
            payload = self.llm.explain(prediction, citations, hint)
            mode = self.llm.mode
        except LLMError as exc:
            logger.warning("llm failed; using offline explanation", extra={"reason": str(exc)})
            payload, mode = fallback, "offline"

        return {"raw_explanation": payload, "fallback_explanation": fallback, "llm_mode": mode}

    def _guard(self, state: dict[str, Any]) -> dict[str, Any]:
        prediction: Prediction = state["prediction"]
        result = guardrails.apply(
            state["raw_explanation"],
            prediction.verdict,
            state["citations"],
            state["fallback_explanation"],
        )
        return {"guarded": result}

    def _finalise(self, state: dict[str, Any]) -> dict[str, Any]:
        prediction: Prediction = state["prediction"]
        guarded = state["guarded"]
        return {
            "explanation": Explanation(
                summary=guarded.summary,
                handling_guidance=guarded.handling_guidance,
                citations=guarded.citations,
                llm_mode=state["llm_mode"],
                grounded=guarded.grounded,
                guardrail_notes=guarded.notes,
            ),
            "action": action_for(prediction.verdict),
        }

    def _cheap_explain(self, state: dict[str, Any]) -> dict[str, Any]:
        """Deterministic explanation for cases that do not warrant a generated one."""
        prediction: Prediction = state["prediction"]
        payload = self._offline.explain(prediction, state["citations"], state.get("food_hint"))
        return {"raw_explanation": payload, "fallback_explanation": payload, "llm_mode": "offline"}

    # --- wiring --------------------------------------------------------------

    def _route_explanation(self, state: dict[str, Any]) -> str:
        """Spend an LLM call only when the classifier actually has a signal."""
        prediction: Prediction = state["prediction"]
        if not self.settings.llm_enabled:
            return "cheap"
        top = max(prediction.probabilities.values(), default=0.0)
        if top < self.settings.llm_min_confidence:
            logger.info("skipping llm: near-uniform prediction", extra={"top_prob": round(top, 4)})
            return "cheap"
        return "llm"

    def _build(self) -> StateGraph:
        graph = StateGraph(max_steps=16)
        graph.add_node("ingest", self._ingest)
        graph.add_node("classify", self._classify)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("explain", self._explain)
        graph.add_node("cheap_explain", self._cheap_explain)
        graph.add_node("guard", self._guard)
        graph.add_node("finalise", self._finalise)

        graph.set_entry_point("ingest")
        graph.add_edge("ingest", "classify")
        graph.add_edge("classify", "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self._route_explanation,
            {"llm": "explain", "cheap": "cheap_explain"},
        )
        graph.add_edge("explain", "guard")
        graph.add_edge("cheap_explain", "guard")
        graph.add_edge("guard", "finalise")
        graph.add_edge("finalise", END)
        return graph

    # --- entry point ---------------------------------------------------------

    def assess(
        self,
        image_bytes: bytes,
        food_hint: str | None = None,
        request_id: str | None = None,
    ) -> Assessment:
        """Run the graph over one image and return a complete, auditable assessment."""
        started = time.perf_counter()
        request_id = request_id or new_request_id()

        final = self.graph.invoke({"image_bytes": image_bytes, "food_hint": food_hint})
        if err := final.get("error"):
            # Re-raise the original exception so callers can distinguish a bad
            # upload from a missing checkpoint from a genuine internal fault.
            if exc := final.get("error_exception"):
                raise exc
            raise RuntimeError(f"triage graph failed: {err}")

        return Assessment(
            request_id=request_id,
            prediction=final["prediction"],
            action=final["action"],
            explanation=final["explanation"],
            trace=final["trace"],
            latency_ms=(time.perf_counter() - started) * 1000,
        )
