"""Typed state carried through the LangGraph investigation workflow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from agent.schemas import (
    ActionSelection,
    ClarificationRequest,
    DiagnosisResult,
    EvidenceAvailability,
    EvidenceItem,
    HandoffPayload,
    InvestigationStatus,
    SubmittedInvestigation,
    SymptomAssessment,
    ToolError,
    ToolName,
    ToolResult,
)


class InvestigationState(TypedDict):
    """Sanitized state for one natural-language deployment investigation."""

    submission: SubmittedInvestigation
    question: str
    workload_name: str | None
    namespace: str | None
    symptom_assessment: SymptomAssessment
    evidence_availability: EvidenceAvailability

    evidence: list[EvidenceItem]
    reference_context: list[dict[str, Any]]
    tools_called: list[ToolName]
    successful_tools: list[ToolName]
    attempts_by_tool: dict[ToolName, int]
    errors: list[ToolError]
    suspected_causes: list[dict[str, Any]]
    public_decisions: list[dict[str, Any]]

    next_action: ActionSelection | None
    last_tool_name: ToolName | None
    last_tool_result: ToolResult | None
    step_count: int
    route_decision: Literal[
        "select", "clarify", "diagnose", "handoff"
    ] | None
    routing_reason: str | None
    model_errors: list[str]
    unsafe_evidence: bool
    contradiction_reason: str | None

    clarification_request: ClarificationRequest | None
    diagnosis: DiagnosisResult | None
    confidence: float
    recommended_action: str | None
    handoff_reason: str | None
    handoff: HandoffPayload | None
    human_acknowledged: bool
    final_outcome: Literal[
        "diagnosed", "needs_clarification", "escalated", "failed"
    ] | None
    status: InvestigationStatus


__all__ = ["InvestigationState"]
