"""Stateful investigation workflow package."""

from typing import Any

from agent.schemas import (
    ActionSelection,
    ClarificationRequest,
    DiagnosisResult,
    EvidenceAvailability,
    EvidenceItem,
    HandoffPayload,
    SubmissionValidationError,
    SubmittedEvidence,
    SubmittedInvestigation,
    SymptomAssessment,
    ToolError,
    ToolResult,
    validate_submitted_investigation,
)
from agent.model import (
    NEBIUS_BASE_URL,
    NebiusConfigurationError,
    NebiusModelAdapter,
    NebiusSettings,
    create_chat_model,
    load_nebius_settings,
)
from agent.state import InvestigationState


def __getattr__(name: str) -> Any:
    """Load graph helpers lazily so tools can safely import agent schemas."""

    if name in {
        "build_investigation_graph",
        "graph_input",
        "investigation_config",
    }:
        from agent.graph import (
            build_investigation_graph,
            graph_input,
            investigation_config,
        )

        return {
            "build_investigation_graph": build_investigation_graph,
            "graph_input": graph_input,
            "investigation_config": investigation_config,
        }[name]
    raise AttributeError(f"module 'agent' has no attribute {name!r}")

__all__ = [
    "ActionSelection",
    "ClarificationRequest",
    "DiagnosisResult",
    "EvidenceAvailability",
    "EvidenceItem",
    "HandoffPayload",
    "InvestigationState",
    "NEBIUS_BASE_URL",
    "NebiusConfigurationError",
    "NebiusModelAdapter",
    "NebiusSettings",
    "SubmissionValidationError",
    "SubmittedEvidence",
    "SubmittedInvestigation",
    "SymptomAssessment",
    "ToolError",
    "ToolResult",
    "build_investigation_graph",
    "create_chat_model",
    "graph_input",
    "investigation_config",
    "load_nebius_settings",
    "validate_submitted_investigation",
]
