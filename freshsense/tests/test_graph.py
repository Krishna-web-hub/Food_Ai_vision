"""The reusable state-graph runtime, tested independently of FreshSense."""

from __future__ import annotations

import pytest

from freshsense.agent.graph import END, GraphError, StateGraph


def _linear() -> StateGraph:
    g = StateGraph()
    g.add_node("a", lambda s: {"seen": [*s.get("seen", []), "a"]})
    g.add_node("b", lambda s: {"seen": [*s.get("seen", []), "b"]})
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g


def test_linear_graph_runs_in_order():
    result = _linear().compile().invoke({})
    assert result["seen"] == ["a", "b"]


def test_every_node_is_traced():
    trace = _linear().compile().invoke({})["trace"]
    assert [t["node"] for t in trace] == ["a", "b"]
    assert all("ms" in t and t["step"] > 0 for t in trace)


def test_conditional_edges_pick_the_branch():
    g = StateGraph()
    g.add_node("start", lambda s: {"n": s["n"] * 2})
    g.add_node("big", lambda s: {"label": "big"})
    g.add_node("small", lambda s: {"label": "small"})
    g.set_entry_point("start")
    g.add_conditional_edges(
        "start", lambda s: "big" if s["n"] > 10 else "small", {"big": "big", "small": "small"}
    )
    g.add_edge("big", END)
    g.add_edge("small", END)
    compiled = g.compile()

    assert compiled.invoke({"n": 8})["label"] == "big"
    assert compiled.invoke({"n": 2})["label"] == "small"


def test_node_returning_none_is_a_no_op():
    g = StateGraph()
    g.add_node("noop", lambda s: None)
    g.set_entry_point("noop")
    g.add_edge("noop", END)
    assert g.compile().invoke({"x": 1})["x"] == 1


def test_a_failing_node_halts_and_records_the_error():
    def boom(state):
        raise ValueError("node exploded")

    g = StateGraph()
    g.add_node("boom", boom)
    g.add_node("never", lambda s: {"reached": True})
    g.set_entry_point("boom")
    g.add_edge("boom", "never")
    g.add_edge("never", END)

    result = g.compile().invoke({})
    assert "node exploded" in result["error"]
    assert "reached" not in result
    assert result["trace"][-1]["error"].startswith("ValueError")


def test_step_budget_stops_a_cycle():
    g = StateGraph(max_steps=5)
    g.add_node("loop", lambda s: {"i": s.get("i", 0) + 1})
    g.set_entry_point("loop")
    g.add_edge("loop", "loop")
    with pytest.raises(GraphError, match="step budget"):
        g.compile().invoke({})


def test_compile_rejects_a_missing_entry_point():
    g = StateGraph()
    g.add_node("a", lambda s: None)
    g.add_edge("a", END)
    with pytest.raises(GraphError, match="no entry point"):
        g.compile()


def test_compile_rejects_edges_to_undefined_nodes():
    g = StateGraph()
    g.add_node("a", lambda s: None)
    g.set_entry_point("a")
    g.add_edge("a", "ghost")
    with pytest.raises(GraphError, match="undefined nodes"):
        g.compile()


def test_compile_rejects_a_node_with_no_outgoing_edge():
    g = StateGraph()
    g.add_node("a", lambda s: None)
    g.set_entry_point("a")
    with pytest.raises(GraphError, match="no outgoing edge"):
        g.compile()


def test_duplicate_node_names_are_rejected():
    g = StateGraph()
    g.add_node("a", lambda s: None)
    with pytest.raises(GraphError, match="duplicate node"):
        g.add_node("a", lambda s: None)


def test_static_and_conditional_edges_cannot_coexist():
    g = StateGraph()
    g.add_node("a", lambda s: None)
    g.add_edge("a", END)
    with pytest.raises(GraphError, match="already has a static edge"):
        g.add_conditional_edges("a", lambda s: "x", {"x": END})


def test_router_returning_an_unmapped_branch_is_an_error():
    g = StateGraph()
    g.add_node("a", lambda s: None)
    g.set_entry_point("a")
    g.add_conditional_edges("a", lambda s: "nope", {"yes": END})
    with pytest.raises(GraphError, match="unmapped branch"):
        g.compile().invoke({})


def test_invoke_does_not_mutate_the_caller_state():
    original = {"n": 1}
    _linear().compile().invoke(original)
    assert original == {"n": 1}


def test_the_original_exception_is_preserved_for_the_caller():
    """Stringifying the failure would lose the type a caller needs to map a response."""

    class DomainError(ValueError):
        pass

    def boom(state):
        raise DomainError("bad input")

    g = StateGraph()
    g.add_node("boom", boom)
    g.set_entry_point("boom")
    g.add_edge("boom", END)

    result = g.compile().invoke({})
    assert isinstance(result["error_exception"], DomainError)
