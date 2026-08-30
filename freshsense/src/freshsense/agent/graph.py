"""A minimal, reusable state-graph runtime for agent workflows.

This is the reusable engineering asset in this repo: it knows nothing about
food, models or HTTP. Nodes are pure `state -> state-delta` functions, edges are
static or conditional, and every step is traced.

The API deliberately mirrors LangGraph's (`add_node`, `add_edge`,
`add_conditional_edges`, `compile`, `invoke`) so a workflow written against this
can move to LangGraph without rewriting the nodes. It is ~120 lines because the
orchestration this class of workflow needs really is that small — the value of
the framework is the tracing, the step budget and the typed state, and those are
worth owning rather than importing.

    graph = StateGraph()
    graph.add_node("classify", classify)
    graph.add_conditional_edges("classify", route, {"explain": "explain", "done": END})
    graph.set_entry_point("classify")
    result = graph.compile().invoke({"image": data})
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

END = "__end__"

State = dict[str, Any]
Node = Callable[[State], Mapping[str, Any] | None]
Router = Callable[[State], str]


class GraphError(RuntimeError):
    """Raised for malformed graphs and for runaway executions."""


class CompiledGraph:
    """An executable graph. Immutable once compiled."""

    def __init__(self, nodes, edges, conditional, entry_point, max_steps) -> None:
        self._nodes = nodes
        self._edges = edges
        self._conditional = conditional
        self._entry_point = entry_point
        self._max_steps = max_steps

    def _next(self, name: str, state: State) -> str:
        if name in self._conditional:
            router, mapping = self._conditional[name]
            key = router(state)
            if key not in mapping:
                raise GraphError(f"router for '{name}' returned unmapped branch '{key}'")
            return mapping[key]
        return self._edges.get(name, END)

    def invoke(self, state: State | None = None) -> State:
        """Run to `END`, recording a trace entry per node under `state["trace"]`."""
        state = dict(state or {})
        state.setdefault("trace", [])
        current = self._entry_point
        steps = 0

        while current != END:
            if steps >= self._max_steps:
                raise GraphError(f"exceeded step budget of {self._max_steps} — cycle?")
            steps += 1

            node = self._nodes[current]
            started = time.perf_counter()
            try:
                delta = node(state) or {}
                error, exception = None, None
            except Exception as exc:
                delta, error, exception = {}, f"{type(exc).__name__}: {exc}", exc
                logger.exception("node failed", extra={"node": current})

            elapsed_ms = (time.perf_counter() - started) * 1000
            state["trace"].append(
                {
                    "step": steps,
                    "node": current,
                    "ms": round(elapsed_ms, 2),
                    "keys": sorted(delta.keys()),
                    **({"error": error} if error else {}),
                }
            )
            if error:
                # Keep the original exception, not just its text: callers need the
                # type to map a failure onto the right response (a bad upload is a
                # 422, not a 500), and stringifying it here would lose that.
                state["error"] = error
                state["error_exception"] = exception
                return state

            state.update(delta)
            current = self._next(current, state)

        return state


class StateGraph:
    """Builder for a `CompiledGraph`."""

    def __init__(self, max_steps: int = 32) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, str] = {}
        self._conditional: dict[str, tuple[Router, Mapping[str, str]]] = {}
        self._entry_point: str | None = None
        self._max_steps = max_steps

    def add_node(self, name: str, fn: Node) -> StateGraph:
        if name in (END, ""):
            raise GraphError(f"'{name}' is not a usable node name")
        if name in self._nodes:
            raise GraphError(f"duplicate node '{name}'")
        self._nodes[name] = fn
        return self

    def add_edge(self, source: str, target: str) -> StateGraph:
        if source in self._conditional:
            raise GraphError(f"'{source}' already has conditional edges")
        self._edges[source] = target
        return self

    def add_conditional_edges(
        self, source: str, router: Router, mapping: Mapping[str, str]
    ) -> StateGraph:
        if source in self._edges:
            raise GraphError(f"'{source}' already has a static edge")
        self._conditional[source] = (router, dict(mapping))
        return self

    def set_entry_point(self, name: str) -> StateGraph:
        self._entry_point = name
        return self

    def compile(self) -> CompiledGraph:
        """Validate the graph, then freeze it. Every reachable target must exist."""
        if self._entry_point is None:
            raise GraphError("no entry point set")
        if self._entry_point not in self._nodes:
            raise GraphError(f"entry point '{self._entry_point}' is not a node")

        targets = set(self._edges.values())
        for _, mapping in self._conditional.values():
            targets.update(mapping.values())
        unknown = {t for t in targets if t != END and t not in self._nodes}
        if unknown:
            raise GraphError(f"edges point at undefined nodes: {sorted(unknown)}")

        dangling = set(self._nodes) - set(self._edges) - set(self._conditional)
        if dangling:
            raise GraphError(f"nodes with no outgoing edge: {sorted(dangling)}")

        return CompiledGraph(
            dict(self._nodes),
            dict(self._edges),
            dict(self._conditional),
            self._entry_point,
            self._max_steps,
        )
