"""Construction helpers for the stateful investigation graph."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from agent.model import InvestigationModel, NebiusModelAdapter
from agent.nodes import InvestigationNodes
from agent.routing import (
    route_diagnosis,
    route_evidence,
    route_selection,
)
from agent.schemas import (
    ActionSelection,
    ClarificationRequest,
    DiagnosisResult,
    EvidenceAvailability,
    EvidenceItem,
    HandoffPayload,
    SubmittedEvidence,
    SubmittedInvestigation,
    SymptomAssessment,
    ToolError,
    ToolResult,
    validate_submitted_investigation,
)
from agent.state import InvestigationState


CHECKPOINT_MODEL_ALLOWLIST = (
    ActionSelection,
    DiagnosisResult,
    EvidenceAvailability,
    EvidenceItem,
    HandoffPayload,
    ClarificationRequest,
    SubmittedEvidence,
    SubmittedInvestigation,
    SymptomAssessment,
    ToolError,
    ToolResult,
)


def _default_checkpointer() -> InMemorySaver:
    """Create a session-local saver with explicit trusted model types."""

    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=CHECKPOINT_MODEL_ALLOWLIST,
    )
    return InMemorySaver(serde=serializer)


def build_investigation_graph(
    model: InvestigationModel | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Compile the natural-language workflow with safe in-memory persistence."""

    nodes = InvestigationNodes(model or NebiusModelAdapter())
    builder = StateGraph(InvestigationState)
    builder.add_node("validate_input", nodes.validate_input)
    builder.add_node("assess_question", nodes.assess_question)
    builder.add_node("select_next_action", nodes.select_next_action)
    builder.add_node("execute_tool", nodes.execute_tool)
    builder.add_node("analyze_evidence", nodes.analyze_evidence)
    builder.add_node("route_investigation", nodes.route_investigation)
    builder.add_node("request_clarification", nodes.request_clarification)
    builder.add_node("produce_diagnosis", nodes.produce_diagnosis)
    builder.add_node("prepare_handoff", nodes.prepare_handoff)
    builder.add_node("human_handoff", nodes.human_handoff)
    builder.add_node("format_result", nodes.format_result)

    builder.add_edge(START, "validate_input")
    builder.add_edge("validate_input", "assess_question")
    builder.add_edge("assess_question", "select_next_action")
    builder.add_conditional_edges(
        "select_next_action",
        route_selection,
        {
            "execute": "execute_tool",
            "clarify": "request_clarification",
            "diagnose": "produce_diagnosis",
            "handoff": "prepare_handoff",
        },
    )
    builder.add_edge("execute_tool", "analyze_evidence")
    builder.add_edge("analyze_evidence", "route_investigation")
    builder.add_conditional_edges(
        "route_investigation",
        route_evidence,
        {
            "select": "select_next_action",
            "clarify": "request_clarification",
            "diagnose": "produce_diagnosis",
            "handoff": "prepare_handoff",
        },
    )
    builder.add_edge("request_clarification", "format_result")
    builder.add_conditional_edges(
        "produce_diagnosis",
        route_diagnosis,
        {"format": "format_result", "handoff": "prepare_handoff"},
    )
    builder.add_edge("prepare_handoff", "human_handoff")
    builder.add_edge("human_handoff", "format_result")
    builder.add_edge("format_result", END)
    resolved_checkpointer = (
        checkpointer if checkpointer is not None else _default_checkpointer()
    )
    return builder.compile(checkpointer=resolved_checkpointer)


def graph_input(
    payload: SubmittedInvestigation | dict[str, Any],
) -> dict[str, SubmittedInvestigation]:
    """Validate and sanitize input before it enters checkpointed graph state."""

    if isinstance(payload, SubmittedInvestigation):
        validated = validate_submitted_investigation(
            payload.model_dump(exclude={"redaction_markers"})
        )
    else:
        validated = validate_submitted_investigation(payload)
    return {"submission": validated}


def investigation_config(thread_id: str) -> dict[str, dict[str, str]]:
    """Create a valid session-scoped checkpointer configuration."""

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("thread_id must be a non-empty string.")
    return {"configurable": {"thread_id": thread_id.strip()}}


__all__ = [
    "build_investigation_graph",
    "graph_input",
    "investigation_config",
]
