# Architecture

## The shape of the system

```
                    HTTP (FastAPI)                    CLI
                          │                            │
                          └──────────┬─────────────────┘
                                     ▼
                            TriageAgent (state graph)
                                     │
   ingest ─► classify ─► retrieve ─┬─(model has a signal)─► explain (Claude) ─┐
                                   └─(model is guessing)──► cheap (offline) ──┴─► guard ─► finalise
                                     │              │                    │
                                     ▼              ▼                    ▼
                              ViT-B/16 head   TF-IDF over kb/      guardrails
```

One `Assessment` comes out: verdict, action, probabilities, policy note,
explanation, citations, guardrail notes, per-node trace, latency.

## Layers, and what each is allowed to know

| Layer | Module | Knows about |
|---|---|---|
| Domain | `domain.py` | Nothing. Plain dataclasses. |
| Vision | `vision/` | Torch, images. Not HTTP, not LLMs. |
| GenAI | `genai/` | Retrieval, prompts, the Anthropic SDK, guardrails. Not torch. |
| Agent | `agent/` | Composes the above. Not HTTP. |
| API | `api/` | HTTP. Delegates everything else. |

The dependency arrows only point downward. That is what lets the whole agent be
tested with a fake classifier and no network — the layers above vision never
import torch.

## Six decisions worth defending

### 1. The decision policy is separate from the model

`Classifier.probabilities` returns a distribution. `decide` turns it into a
verdict. They are separate functions because they answer to different people:
the model is a data-science artefact, the policy is a risk posture a food-safety
owner has to be able to read, argue with and change without retraining anything.
It is asymmetric by design — see the model card.

### 2. The graph engine is ours, not a framework

`agent/graph.py` is a ~120-line `StateGraph`: nodes are `state -> delta`
functions, edges are static or conditional, every step is traced, and there is a
step budget so a cycle fails loudly instead of billing forever.

The API deliberately mirrors LangGraph's (`add_node`, `add_edge`,
`add_conditional_edges`, `compile`, `invoke`) so a workflow written here can move
across without rewriting the nodes. It is hand-rolled because for a six-node
pipeline the orchestration really is that small, and what we actually needed —
tracing, a step budget, typed state, `compile()`-time validation of dangling
edges — is worth owning outright. It knows nothing about food and is the piece
of this repo most reusable elsewhere.

### 3. Retrieval is lexical, and the seam is explicit

Five documents, thirteen chunks. TF-IDF with cosine similarity answers this
correctly and deterministically; an embedding model would add a network
dependency, a warm-up cost and run-to-run variance to solve a problem that is
already solved.

The `Retriever` protocol in `genai/rag.py` is the seam. Swapping in pgvector or
Qdrant means implementing `search(query, top_k) -> list[Citation]`, and nothing
above that module changes. Chunking is on `##` headings rather than fixed
windows, because these documents are written as self-contained sections and a
boundary mid-rule would strand the qualifier that makes the rule safe.

Retrieval returns `[]` rather than padding to `top_k` with low-scoring chunks.
Padding is how a RAG system starts citing passages that do not support the claim.

### 4. The LLM has an offline twin, and it is not a stub

`build_llm_client` returns `AnthropicClient` when a key is configured and
`OfflineClient` when it is not. The offline path produces a complete, cited,
guard-railed explanation with no network — which is what keeps the test suite
hermetic, makes an air-gapped deployment viable, and gives every live call a
fallback when Claude is unavailable.

`test_offline_output_survives_its_own_guardrails` asserts the fallback never
trips the rules it is the fallback for. A fallback that gets blocked is not one.

The graph also skips the LLM entirely when the classifier is near-uniform: a
fluent narrative wrapped around a coin flip adds authority the evidence does not
support, and costs tokens to do it.

### 5. Guardrails treat model output as input

Everything the LLM returns is validated before it leaves the process:

1. **Shape** — missing or non-string fields fall back to the offline text.
2. **Grounding** — every cited `source_id` must be one we actually retrieved. A
   hallucinated citation discards the *whole* explanation, because a model
   inventing sources is not trustworthy on the prose either.
3. **Forbidden content** — tasting advice, "safe to eat", absolute certainty and
   medical advice are replaced wholesale, not edited around.
4. **Disclaimer** — a `fresh` verdict always carries its scope limitation.

These are mechanical and individually tested. The system prompt asks for the
same things; the guardrails are what make it true.

### 6. Checkpoints carry the head, not the backbone

The notebook saved 343 MB to persist 1,538 trained parameters. The backbone is
public ImageNet weights. `save_classifier_head` writes a self-describing 8.2 KB
payload — format tag, class names, backbone id, head state dict — which is the
difference between a model you can put in a container layer or a release asset
and one you cannot. `load_classifier_head` still accepts the legacy full state
dict, loudly, so old artefacts keep working.

## Error handling

The graph catches a node failure, records it in the trace, and **keeps the
original exception object** rather than only its string. `TriageAgent.assess`
re-raises it, so the API can map a bad upload to 422 and a missing checkpoint to
503 instead of collapsing both into a 500. This was a real bug — the first
version stringified the exception, and `test_a_corrupt_upload_is_a_422_not_a_500`
caught it.

## Observability

* JSON logs with a request id on a contextvar, propagated through the middleware
  and echoed as `x-request-id`.
* `/healthz` is liveness — dependency-free, so a wedged model cannot fail it.
* `/readyz` reports each dependency separately, so a failure is diagnosable
  rather than a single red light.
* Every assessment carries a per-node trace with timings — the same structure
  whether it came from HTTP or the CLI.

The knowledge base is warmed at startup; the classifier stays lazy, because a
330 MB backbone load should not block the readiness probe.

## What is deliberately missing

* **No database.** Assessments are returned, not stored. Persistence is a real
  requirement for audit and belongs behind a repository interface.
* **No auth.** The service is unauthenticated. It would sit behind a gateway.
* **No metrics endpoint.** Logs carry the timings; a Prometheus exporter is a
  small addition and was not the point of this build.
* **No drift monitoring.** The model card names this as the main operational
  risk. Detecting it needs production traffic, which does not exist yet.
