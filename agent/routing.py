"""Deterministic routing guards for natural-language investigations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final, Literal, cast

from agent.schemas import (
    ActionSelection,
    EvidenceItem,
    EvidenceSourceName,
    ToolName,
)
from agent.state import InvestigationState
from tools.diagnostics import DIAGNOSTIC_TOOL_ALLOWLIST, DiagnosticToolName


MAX_TOOL_EXECUTIONS: Final[int] = 6
MAX_TOOL_ATTEMPTS: Final[int] = 2
MODEL_OUTPUT_ATTEMPTS: Final[int] = 2
MIN_DIAGNOSIS_CONFIDENCE: Final[float] = 0.80

_TOOL_TO_SOURCE: Final[dict[DiagnosticToolName, EvidenceSourceName | None]] = {
    "inspect_workload_status": "workload_status",
    "inspect_kubernetes_events": "kubernetes_events",
    "inspect_container_logs": "container_logs",
    "inspect_manifest_config": "manifest_yaml",
    "search_runbook": None,
}
_CATEGORY_ORDER: Final[dict[str, tuple[DiagnosticToolName, ...]]] = {
    "image_pull": (
        "inspect_kubernetes_events",
        "inspect_workload_status",
        "inspect_manifest_config",
        "inspect_container_logs",
        "search_runbook",
    ),
    "startup_crash": (
        "inspect_container_logs",
        "inspect_workload_status",
        "inspect_kubernetes_events",
        "inspect_manifest_config",
        "search_runbook",
    ),
    "configuration": (
        "inspect_kubernetes_events",
        "inspect_manifest_config",
        "inspect_workload_status",
        "inspect_container_logs",
        "search_runbook",
    ),
    "readiness": (
        "inspect_workload_status",
        "inspect_kubernetes_events",
        "inspect_manifest_config",
        "inspect_container_logs",
        "search_runbook",
    ),
    "scheduling": (
        "inspect_kubernetes_events",
        "inspect_workload_status",
        "inspect_manifest_config",
        "inspect_container_logs",
        "search_runbook",
    ),
    "storage": (
        "inspect_kubernetes_events",
        "inspect_manifest_config",
        "inspect_workload_status",
        "inspect_container_logs",
        "search_runbook",
    ),
    "unknown": (
        "inspect_workload_status",
        "inspect_kubernetes_events",
        "inspect_container_logs",
        "inspect_manifest_config",
        "search_runbook",
    ),
}
_HEALTHY_STATUS_PATTERN = re.compile(
    r"(?i)(?:all replicas (?:are )?ready|successfully rolled out|"
    r"\"ready_?replicas\"\s*:\s*[1-9]\d*|"
    r"\"available_?replicas\"\s*:\s*[1-9]\d*)"
)
# Readiness must be inferred from status polarity, never from the presence of a
# "ready" or "available" field name, which every workload status document has.
_UNREADY_STATUS_PATTERN = re.compile(
    r"(?i)(?:"
    r"\"ready_?replicas\"\s*:\s*0(?!\d)|"
    r"\"available_?replicas\"\s*:\s*0(?!\d)|"
    r"\"unavailable_?replicas\"\s*:\s*[1-9]\d*|"
    r"\"ready\"\s*:\s*false|"
    r"containersnotready|"
    r"containers with unready status"
    r")"
)
_CAUSE_SPECIFIC_SIGNALS: Final[frozenset[str]] = frozenset(
    {
        "missing_config",
        "readiness_failure",
        "scheduling_failure",
        "storage_failure",
    }
)
_READINESS_CORROBORATING_TOOLS: Final[frozenset[ToolName]] = frozenset(
    {"inspect_manifest_config", "inspect_container_logs"}
)


def preferred_tool_order(
    state: InvestigationState,
) -> tuple[DiagnosticToolName, ...]:
    """Return symptom-specific evidence order without reading uninspected text."""

    return _CATEGORY_ORDER[state["symptom_assessment"].category]


def _source_available(
    state: InvestigationState,
    source: EvidenceSourceName,
) -> bool:
    if state["submission"].demo_id is not None:
        return True
    return bool(getattr(state["evidence_availability"], source))


def tool_is_available(
    state: InvestigationState,
    tool_name: DiagnosticToolName,
) -> bool:
    source = _TOOL_TO_SOURCE[tool_name]
    return source is None or _source_available(state, source)


def pending_retry(
    state: InvestigationState,
) -> DiagnosticToolName | None:
    """Return one transient failure eligible for its single retry."""

    for error in reversed(state["errors"]):
        if error.tool_name not in DIAGNOSTIC_TOOL_ALLOWLIST:
            continue
        current_attempt = state["attempts_by_tool"].get(error.tool_name, 0)
        if (
            error.retryable
            and error.attempt < MAX_TOOL_ATTEMPTS
            and current_attempt == error.attempt
        ):
            return cast(DiagnosticToolName, error.tool_name)
    return None


def next_available_incident_tool(
    state: InvestigationState,
) -> DiagnosticToolName | None:
    """Choose an uninspected available incident source in symptom order."""

    attempted = set(state["attempts_by_tool"])
    for tool_name in preferred_tool_order(state):
        if tool_name == "search_runbook":
            continue
        if tool_name not in attempted and tool_is_available(state, tool_name):
            return tool_name
    return None


def recommended_missing_source(
    state: InvestigationState,
) -> EvidenceSourceName | None:
    """Choose the single missing source most likely to resolve uncertainty."""

    if state["submission"].demo_id is not None:
        return None
    for tool_name in preferred_tool_order(state):
        source = _TOOL_TO_SOURCE[tool_name]
        if source is not None and not _source_available(state, source):
            return source
    return None


def _signals_for_evidence(
    item: EvidenceItem,
    *,
    symptom_category: str,
) -> set[str]:
    normalized = item.summary.casefold()
    signals: set[str] = set()
    if any(
        marker in normalized
        for marker in (
            "imagepullbackoff",
            "errimagepull",
            "manifest unknown",
            "pull access denied",
            "no basic auth credentials",
        )
    ):
        signals.add("image_pull")
    if any(
        marker in normalized
        for marker in (
            "crashloopbackoff",
            "back-off restarting",
            "modulenotfounderror",
            "no module named",
            "importerror",
            "filenotfounderror",
        )
    ):
        signals.add("startup_failure")
    if any(
        marker in normalized
        for marker in (
            "createcontainerconfigerror",
            "configmapkeyref",
            "configmapref",
            "secretkeyref",
            "secretref",
            "configmap \\\"",
            "secret \\\"",
        )
    ):
        signals.add("missing_config")
    if any(
        marker in normalized
        for marker in (
            "readinessprobe",
            "readiness probe",
            "statuscode: 404",
            "readiness check failed",
        )
    ):
        signals.add("readiness_failure")
    if any(
        marker in normalized
        for marker in (
            "failedscheduling",
            "insufficient cpu",
            "insufficient memory",
            "unschedulable",
        )
    ):
        signals.add("scheduling_failure")
    if any(
        marker in normalized
        for marker in (
            "failedmount",
            "persistentvolumeclaim",
            "claimname",
            "pvc",
        )
    ):
        signals.add("storage_failure")

    if item.source_tool == "inspect_workload_status":
        if (
            symptom_category == "readiness"
            and _UNREADY_STATUS_PATTERN.search(item.summary)
            and not _HEALTHY_STATUS_PATTERN.search(item.summary)
        ):
            signals.add("readiness_failure")
        if symptom_category == "scheduling" and "pending" in normalized:
            signals.add("scheduling_failure")
        if symptom_category == "storage" and "pending" in normalized:
            signals.add("storage_failure")
    return signals


def has_sufficient_evidence(
    state: InvestigationState,
    evidence: list[EvidenceItem] | None = None,
) -> bool:
    """Require compatible evidence from independent incident sources."""

    incident_evidence = [
        item
        for item in (evidence if evidence is not None else state["evidence"])
        if item.source_tool != "search_runbook"
    ]
    if len({item.source_tool for item in incident_evidence}) < 2:
        return False

    category = state["symptom_assessment"].category
    signal_sources: dict[str, set[ToolName]] = {}
    decisive_signals: set[str] = set()
    for item in incident_evidence:
        signals = _signals_for_evidence(item, symptom_category=category)
        for signal in signals:
            signal_sources.setdefault(signal, set()).add(item.source_tool)
            if item.decisive:
                decisive_signals.add(signal)

    for signal, sources in signal_sources.items():
        if len(sources) < 2:
            continue
        if signal == "readiness_failure" and category == "readiness":
            corroborating_sources = (
                sources & _READINESS_CORROBORATING_TOOLS
            )
            if not corroborating_sources:
                continue
            if (
                signal not in decisive_signals
                and not _READINESS_CORROBORATING_TOOLS.issubset(sources)
            ):
                continue
        if signal in decisive_signals or signal in _CAUSE_SPECIFIC_SIGNALS:
            return True
    return False


def contradiction_reason(
    state: InvestigationState,
) -> str | None:
    """Return a public contradiction summary without exposing hidden reasoning."""

    category = state["symptom_assessment"].category
    decisive_groups: set[str] = set()
    healthy_status = False
    decisive_runtime = False
    for item in state["evidence"]:
        if item.source_tool == "search_runbook":
            continue
        if (
            item.source_tool == "inspect_workload_status"
            and _HEALTHY_STATUS_PATTERN.search(item.summary)
        ):
            healthy_status = True
        if item.decisive:
            decisive_runtime = True
            decisive_groups.update(
                _signals_for_evidence(item, symptom_category=category)
            )
    if healthy_status and decisive_runtime:
        return (
            "Submitted workload health conflicts with a decisive failure signal."
        )
    if len(decisive_groups) > 1:
        return "Submitted evidence supports incompatible failure causes."
    return None


def _public_action_reason(action: str) -> str:
    reasons = {
        "inspect_workload_status": "Inspecting the submitted workload status.",
        "inspect_kubernetes_events": "Inspecting the submitted Kubernetes events.",
        "inspect_container_logs": "Inspecting the submitted container logs.",
        "inspect_manifest_config": "Inspecting the submitted manifest configuration.",
        "search_runbook": "Searching reference-only troubleshooting guidance.",
        "diagnose": "The incident evidence satisfies the diagnosis threshold.",
        "request_clarification": "One specific missing source could resolve the investigation.",
        "handoff": "Engineer review is required to avoid an unsupported conclusion.",
    }
    return reasons[action]


def _terminal_action(
    state: InvestigationState,
) -> Literal["request_clarification", "handoff"]:
    return (
        "request_clarification"
        if recommended_missing_source(state) is not None
        else "handoff"
    )


def normalize_action(
    state: InvestigationState,
    selection: ActionSelection,
) -> tuple[ActionSelection, str | None]:
    """Enforce allowlist, availability, repetition, sufficiency, and limits."""

    if state["step_count"] >= MAX_TOOL_EXECUTIONS:
        action = (
            "diagnose"
            if has_sufficient_evidence(state)
            else "handoff"
        )
        return (
            ActionSelection(action=action, reason=_public_action_reason(action)),
            "The six-execution limit replaced the proposed action.",
        )

    retry_tool = pending_retry(state)
    if retry_tool is not None:
        return (
            ActionSelection(
                action=retry_tool,
                reason="Retrying one transient read-only inspection failure.",
            ),
            "A bounded retry took priority over the proposed action.",
        )

    if state["unsafe_evidence"] or state["contradiction_reason"]:
        return (
            ActionSelection(
                action="handoff",
                reason=_public_action_reason("handoff"),
            ),
            "Unsafe or contradictory evidence requires engineer review.",
        )

    action = selection.action
    if action == "diagnose" and not has_sufficient_evidence(state):
        fallback = next_available_incident_tool(state)
        action = fallback or _terminal_action(state)
        return (
            ActionSelection(action=action, reason=_public_action_reason(action)),
            "Premature diagnosis was replaced by a safe evidence decision.",
        )

    if action == "diagnose":
        return (
            ActionSelection(
                action="diagnose",
                reason=_public_action_reason("diagnose"),
            ),
            None,
        )

    if action == "request_clarification":
        if recommended_missing_source(state) is None:
            action = "handoff"
        return (
            ActionSelection(action=action, reason=_public_action_reason(action)),
            None,
        )

    if action in DIAGNOSTIC_TOOL_ALLOWLIST:
        tool_name = cast(DiagnosticToolName, action)
        already_attempted = tool_name in state["attempts_by_tool"]
        if already_attempted or not tool_is_available(state, tool_name):
            fallback = next_available_incident_tool(state)
            action = fallback or _terminal_action(state)
            note = (
                "A repeated successful inspection was replaced."
                if tool_name in state["successful_tools"]
                else "An unavailable or completed inspection was replaced."
            )
            return (
                ActionSelection(
                    action=action,
                    reason=_public_action_reason(action),
                ),
                note,
            )
        if tool_name == "search_runbook" and not state["evidence"]:
            fallback = next_available_incident_tool(state)
            action = fallback or _terminal_action(state)
            return (
                ActionSelection(
                    action=action,
                    reason=_public_action_reason(action),
                ),
                "Reference guidance cannot replace incident evidence.",
            )
        return (
            ActionSelection(
                action=tool_name,
                reason=_public_action_reason(tool_name),
            ),
            None,
        )

    if action == "handoff":
        return (
            ActionSelection(
                action="handoff",
                reason=_public_action_reason("handoff"),
            ),
            None,
        )

    fallback = next_available_incident_tool(state)
    action = fallback or _terminal_action(state)
    return (
        ActionSelection(action=action, reason=_public_action_reason(action)),
        "A non-allowlisted action was replaced.",
    )


def decide_after_evidence(
    state: InvestigationState,
) -> tuple[Literal["select", "clarify", "diagnose", "handoff"], str]:
    """Choose the next state transition after one bounded inspection."""

    if state["unsafe_evidence"]:
        return "handoff", "Instruction-shaped evidence requires engineer review."
    contradiction = contradiction_reason(state)
    if contradiction:
        return "handoff", contradiction
    if has_sufficient_evidence(state):
        return "diagnose", "Independent incident evidence supports diagnosis."
    if state["step_count"] >= MAX_TOOL_EXECUTIONS:
        return "handoff", "The bounded six-execution limit was reached."
    if pending_retry(state) is not None:
        return "select", "A transient failure is eligible for one retry."
    if next_available_incident_tool(state) is not None:
        return "select", "Another relevant submitted source remains uninspected."
    if recommended_missing_source(state) is not None:
        return "clarify", "A specific missing evidence source could resolve the question."
    return "handoff", "Available evidence is exhausted without safe support."


def route_selection(state: InvestigationState) -> str:
    selection = state["next_action"]
    if selection is None:
        return "handoff"
    if selection.action in DIAGNOSTIC_TOOL_ALLOWLIST:
        return "execute"
    if selection.action == "diagnose":
        return "diagnose"
    if selection.action == "request_clarification":
        return "clarify"
    return "handoff"


def route_evidence(state: InvestigationState) -> str:
    return state["route_decision"] or "handoff"


def route_diagnosis(state: InvestigationState) -> str:
    return "format" if state["status"] == "diagnosed" else "handoff"


__all__ = [
    "MAX_TOOL_ATTEMPTS",
    "MAX_TOOL_EXECUTIONS",
    "MIN_DIAGNOSIS_CONFIDENCE",
    "MODEL_OUTPUT_ATTEMPTS",
    "decide_after_evidence",
    "has_sufficient_evidence",
    "next_available_incident_tool",
    "normalize_action",
    "pending_retry",
    "preferred_tool_order",
    "recommended_missing_source",
    "route_diagnosis",
    "route_evidence",
    "route_selection",
    "contradiction_reason",
    "tool_is_available",
]
