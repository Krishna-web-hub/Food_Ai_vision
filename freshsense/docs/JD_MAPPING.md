# FreshSense against the Elite Technology Engineer JD

Accenture Elite Technology Engineer · Management Level 11 · 2027 batch.
Every row points at code you can open, not a claim.

## Required skills

| JD requirement | Where it lives | Honest status |
|---|---|---|
| Strong software engineering fundamentals in **Python** | Typed layered package under `src/freshsense/`, protocols at every seam, 87% line coverage | Solid |
| **Production-grade applications**, full-stack, cloud-native | FastAPI service, `/healthz` + `/readyz`, structured JSON logs, request-id propagation, Docker multi-stage build, compose, GitHub Actions | Backend only — no UI |
| **APIs, containers, CI/CD** | `api/app.py` with an OpenAPI contract; non-root, read-only, healthchecked container; CI runs lint, format, tests, a coverage floor and a container smoke test | Solid |
| **AI-assisted development** throughout the SDLC | This repo was built with Claude Code; the notebook was lifted into it the same way | Real, and the point |
| **LLM APIs integrated with sound engineering practice** | `genai/llm.py`: typed client, timeout, fallback on failure, token usage returned for cost accounting, SDK faked at the transport boundary in tests | Solid |
| **Quality engineering** — unit tests, automation, secure coding, production-readiness | 133 fast tests plus 9 integration tests; upload size cap, decode-bomb guard, non-root container, no secrets in code | Solid |
| **Validating AI output** before production | `genai/guardrails.py` — grounding, forbidden content, shape, disclaimer; each rule tested | This is the centre of the repo |
| **Responsible AI** | `docs/MODEL_CARD.md` with real numbers and named limitations; the abstain band; the enforced scope disclaimer | Solid |
| Reusable IP and engineering accelerators | `agent/graph.py` — domain-free, dependency-free state-graph runtime | Solid |

## Preferred skills

| JD preference | Where it lives | Honest status |
|---|---|---|
| **RAG** | `genai/rag.py` — section chunking, TF-IDF retrieval, cited passages, empty-result honesty | Real RAG, lexical retrieval |
| **Prompt engineering** | `genai/prompts.py` — versioned (`explain.v3`), constrained JSON output, version logged per call | Solid |
| **Agentic frameworks** (LangGraph, LangChain, AutoGen, CrewAI) | `agent/graph.py` mirrors LangGraph's API deliberately | **Built, not imported** — see below |
| **Vector databases / semantic retrieval** | `Retriever` protocol is the seam; no vector DB is wired up | **Gap** — see below |
| **Reusable AI platforms, domain AI** | The graph runtime, plus a swappable classifier/retriever/LLM | Solid |
| **Explainable + responsible AI** | Every verdict ships a policy note, citations and a trace | Solid |
| Open-source / research contribution | Not done | **Gap** |

## The two gaps, stated plainly

**No vector database.** Thirteen chunks do not need one, and pulling in Qdrant to
say the words would be worse engineering than not. What exists instead is the
seam — a `Retriever` protocol — so the swap is one class. If the interview wants
to see it, wiring Qdrant behind that protocol is an afternoon, and the honest
version of the answer is better than a token integration.

**LangGraph is mirrored, not imported.** Deliberate: for six nodes, the
orchestration is ~120 lines, and owning it bought `compile()`-time validation of
dangling edges, a step budget, and a trace shaped for this domain. The API match
means a port costs nothing. Be ready for "why not just use LangGraph?" — the
answer is above, and "because a dependency you cannot debug at 3am is not free"
is the short form.

## What to actually say in the room

Lead with the finding, not the stack. On the held-out test split the raw model
misses one spoiled item — `spoiled_recall` 0.967. Under the deployed asymmetric
policy, **zero spoiled items are released**; that one case lands in the abstain
band and goes to a human, at a cost of a 1.6% abstention rate. That single
sentence demonstrates: you measure the right metric, you understand the cost
asymmetry, you separated policy from model, and you can show the number.

Three more that land:

* **The 343 MB → 8.2 KB checkpoint.** The notebook persisted the whole ViT to
  save 1,538 trained parameters. Noticing that is deployment thinking.
* **The `.avif` file.** One of 31 spoiled test images is silently dropped by
  `ImageFolder` because the extension is not in `IMG_EXTENSIONS`. No error, no
  warning. It is written into the model card rather than quietly rounded away —
  which is the habit the "validate AI-generated outputs" bullet is really asking
  about.
* **The 422 bug.** The graph originally stringified node exceptions, which would
  have turned every corrupt upload into a 500. A test caught it. That is a
  concrete answer to "tell me about a bug you found in your own design."

Do not claim the model is good. 598 training images from one source, frozen
backbone, uncalibrated probabilities — the model card says all of this. The work
on display is the engineering around a mediocre model, which is what the role is
actually about.

## Where this came from

`Projects/pD_final.ipynb` — a working Colab notebook that trains a ViT to 98.4%
validation accuracy. It still runs; `make model` exports its checkpoint into
this service. The notebook is the experiment. This repo is what the JD's phrase
"production-grade software and reusable engineering assets" means in practice:
config, tests, an API, a container, a pipeline, an audit trail, and a written
account of what the model cannot do.
