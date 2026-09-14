"""Nodes for the sanitized natural-language investigation workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, cast

from langgraph.types import interrupt
from pydantic import ValidationError

from agent.model import InvestigationModel
from agent.prompts import (
    build_action_messages,
    build_assessment_messages,
    build_diagnosis_messages,
)
from agent.schemas import (
    ActionSelection,
    ClarificationRequest,
    DiagnosisResult,
    EvidenceItem,
    HandoffPayload,
    SubmissionValidationError,
    SubmittedInvestigation,
    SymptomAssessment,
    ToolError,
    ToolName,
    ToolResult,
)
from agent.state import InvestigationState
from agent.routing import (
    MAX_TOOL_EXECUTIONS,
    MIN_DIAGNOSIS_CONFIDENCE,
    MODEL_OUTPUT_ATTEMPTS,
    decide_after_evidence,
    has_sufficient_evidence,
    next_available_incident_tool,
    normalize_action,
    pending_retry,
    recommended_missing_source,
    contradiction_reason,
)
from tools import (
    inspect_container_logs,
    inspect_kubernetes_events,
    inspect_manifest_config,
    inspect_workload_status,
    search_runbook,
)
from tools.safety import neutralize_untrusted_text
from tools.diagnostics import DIAGNOSTIC_TOOL_ALLOWLIST, DiagnosticToolName


TOOL_REGISTRY: Final[dict[DiagnosticToolName, Any]] = {
    "inspect_workload_status": inspect_workload_status,
    "inspect_kubernetes_events": inspect_kubernetes_events,
    "inspect_container_logs": inspect_container_logs,
    "inspect_manifest_config": inspect_manifest_config,
    "search_runbook": search_runbook,
}

if tuple(TOOL_REGISTRY) != DIAGNOSTIC_TOOL_ALLOWLIST:
    raise RuntimeError(
        "The submitted investigation registry must match its fixed allowlist."
    )

_SOURCE_LABELS: Final[dict[str, str]] = {
    "workload_status": "workload or pod status",
    "kubernetes_events": "recent Kubernetes warning events",
    "container_logs": "recent container logs",
    "manifest_yaml": "the relevant workload manifest",
}


def _safe_model_error(operation: str, error: Exception) -> str:
    return f"{operation} failed validation ({type(error).__name__})."


def _category_for_question(question: str) -> str:
    normalized = question.casefold()
    if any(marker in normalized for marker in ("imagepull", "image pull", "registry")):
        return "image_pull"
    if any(
        marker in normalized
        for marker in ("crashloop", "crash", "restart", "starting")
    ):
        return "startup_crash"
    if any(
        marker in normalized
        for marker in ("configmap", "secret", "configuration")
    ):
        return "configuration"
    if any(marker in normalized for marker in ("readiness", "ready", "probe")):
        return "readiness"
    if any(
        marker in normalized
        for marker in ("pvc", "volume", "mount", "storage")
    ):
        return "storage"
    if any(
        marker in normalized
        for marker in ("pending", "scheduling", "schedule", "cpu", "memory")
    ):
        return "scheduling"
    return "unknown"


def _assessment_for_state(
    state: InvestigationState,
    category: str | None = None,
) -> SymptomAssessment:
    category = category or _category_for_question(state["question"])
    summaries = {
        "image_pull": "The question reports an image retrieval failure.",
        "startup_crash": "The question reports a startup or restart failure.",
        "configuration": "The question reports a configuration-related failure.",
        "readiness": "The question reports a readiness or health-check failure.",
        "scheduling": "The question reports a scheduling or capacity failure.",
        "storage": "The question reports a storage or volume failure.",
        "unknown": "The question reports a deployment failure without a narrow category.",
    }
    relevant_by_category = {
        "image_pull": ["kubernetes_events", "workload_status", "manifest_yaml"],
        "startup_crash": ["container_logs", "workload_status", "kubernetes_events"],
        "configuration": ["kubernetes_events", "manifest_yaml", "workload_status"],
        "readiness": ["workload_status", "kubernetes_events", "manifest_yaml"],
        "scheduling": ["kubernetes_events", "workload_status", "manifest_yaml"],
        "storage": ["kubernetes_events", "manifest_yaml", "workload_status"],
        "unknown": ["workload_status", "kubernetes_events", "container_logs"],
    }
    relevant = relevant_by_category[category]
    missing = [
        source
        for source in relevant
        if not getattr(state["evidence_availability"], source)
        and state["submission"].demo_id is None
    ]
    return SymptomAssessment(
        category=category,
        summary=summaries[category],
        observable_signals=[],
        relevant_sources=relevant,
        missing_sources=missing,
    )


def _initial_hypotheses(category: str) -> list[dict[str, Any]]:
    hypotheses = {
        "image_pull": [
            "The image reference may be invalid.",
            "Registry access may be unavailable or unauthorized.",
        ],
        "startup_crash": ["The application may be failing during startup."],
        "configuration": ["A required configuration reference may be unavailable."],
        "readiness": ["The health check may not match application behavior."],
        "scheduling": ["Requested capacity may be unavailable."],
        "storage": ["A required volume claim or mount may be unavailable."],
        "unknown": ["Runtime evidence is needed to narrow the failure."],
    }
    return [
        {"cause": cause, "basis": "question context only", "provisional": True}
        for cause in hypotheses[category]
    ]


def _result_content(result: ToolResult) -> str:
    if not isinstance(result.data, Mapping):
        return ""
    return str(result.data.get("content", ""))[:2_500]


def _summarize_result(tool_name: ToolName, result: ToolResult) -> str:
    content = _result_content(result)
    labels = {
        "inspect_workload_status": "Workload status",
        "inspect_kubernetes_events": "Kubernetes events",
        "inspect_container_logs": "Container logs",
        "inspect_manifest_config": "Manifest configuration",
    }
    return f"{labels.get(tool_name, 'Inspection')}: {content}"[:3_000]


def _is_decisive(tool_name: ToolName, summary: str) -> bool:
    normalized = summary.casefold()
    if tool_name == "inspect_kubernetes_events":
        return any(
            marker in normalized
            for marker in (
                "manifest unknown",
                "no basic auth credentials",
                "pull access denied",
                "not found",
                "statuscode: 404",
                "insufficient cpu",
                "insufficient memory",
                "failedmount",
            )
        )
    if tool_name == "inspect_container_logs":
        return any(
            marker in normalized
            for marker in (
                "modulenotfounderror",
                "no module named",
                "importerror",
                "filenotfounderror",
                "oomkilled",
            )
        )
    if tool_name == "inspect_workload_status":
        return "oomkilled" in normalized
    return False


def _candidate_cause(tool_name: ToolName, summary: str) -> str | None:
    normalized = summary.casefold()
    candidates = (
        ("manifest unknown", "The referenced container image tag may not exist."),
        ("no basic auth credentials", "Registry authentication may be missing."),
        ("pull access denied", "Registry authorization may be invalid."),
        ("configmap", "A required ConfigMap reference may be unavailable."),
        ("secret", "A required Secret reference may be unavailable."),
        ("statuscode: 404", "The readiness probe path may be incorrect."),
        ("insufficient cpu", "The workload may request unavailable CPU capacity."),
        ("persistentvolumeclaim", "A referenced volume claim may be unavailable."),
        ("modulenotfounderror", "The image may be missing a runtime dependency."),
        ("no module named", "The image may be missing a runtime dependency."),
        ("oomkilled", "The container may be exceeding its memory limit."),
    )
    for marker, cause in candidates:
        if marker in normalized:
            return cause
    if tool_name == "inspect_container_logs" and "startup" in normalized:
        return "The application is failing during startup, but the cause is not yet verified."
    return None


class InvestigationNodes:
    """State transitions for one bounded natural-language investigation."""

    def __init__(self, model: InvestigationModel) -> None:
        self._model = model

    def validate_input(
        self, state: InvestigationState
    ) -> dict[str, Any]:
        """Revalidate a pre-sanitized submission before checkpointed use."""

        raw_submission = state.get("submission")
        if not isinstance(raw_submission, SubmittedInvestigation):
            raise SubmissionValidationError(
                ["A validated SubmittedInvestigation is required."]
            )
        try:
            submission = SubmittedInvestigation.model_validate(
                raw_submission.model_dump(exclude={"redaction_markers"})
            )
        except ValidationError:
            raise SubmissionValidationError(
                ["The submitted investigation could not be validated safely."]
            ) from None

        safe_question, removed = neutralize_untrusted_text(submission.question)
        if removed:
            submission = submission.model_copy(update={"question": safe_question})
        return {
            "submission": submission,
            "question": safe_question,
            "workload_name": submission.workload_name,
            "namespace": submission.namespace,
            "evidence_availability": submission.evidence_availability(),
            "evidence": [],
            "reference_context": [],
            "tools_called": [],
            "successful_tools": [],
            "attempts_by_tool": {},
            "errors": [],
            "suspected_causes": [],
            "public_decisions": [],
            "next_action": None,
            "last_tool_name": None,
            "last_tool_result": None,
            "step_count": 0,
            "route_decision": None,
            "routing_reason": None,
            "model_errors": [],
            "unsafe_evidence": bool(removed),
            "contradiction_reason": None,
            "clarification_request": None,
            "diagnosis": None,
            "confidence": 0.0,
            "recommended_action": None,
            "handoff_reason": None,
            "handoff": None,
            "human_acknowledged": False,
            "final_outcome": None,
            "status": "new",
        }

    def _proposed_category(
        self, state: InvestigationState
    ) -> tuple[str | None, str, list[str]]:
        """Prefer a validated model category, falling back to local rules."""

        assessor = getattr(self._model, "assess_symptom", None)
        if assessor is None:
            return None, "rules", []
        try:
            proposed = SymptomAssessment.model_validate(
                assessor(build_assessment_messages(state))
            )
        except Exception as error:
            return (
                None,
                "rules",
                [_safe_model_error("Symptom assessment", error)],
            )
        return proposed.category, "model", []

    def assess_question(
        self, state: InvestigationState
    ) -> dict[str, Any]:
        """Classify the question without treating it as incident evidence."""

        category, classified_by, model_errors = self._proposed_category(state)
        assessment = _assessment_for_state(state, category)
        return {
            "symptom_assessment": assessment,
            "suspected_causes": _initial_hypotheses(assessment.category),
            "status": "investigating",
            "model_errors": [*state["model_errors"], *model_errors],
            "public_decisions": [
                {
                    "phase": "assessment",
                    "category": assessment.category,
                    "summary": assessment.summary,
                    "question_is_evidence": False,
                    "classified_by": classified_by,
                }
            ],
        }

    def select_next_action(
        self, state: InvestigationState
    ) -> dict[str, Any]:
        """Validate a model action and replace unsafe choices deterministically."""

        errors = list(state["model_errors"])
        guard_note: str | None = None
        model_action: str | None = None
        default_action = next_available_incident_tool(state)

        if state["step_count"] >= MAX_TOOL_EXECUTIONS:
            action = (
                "diagnose"
                if has_sufficient_evidence(state)
                else "handoff"
            )
            selection = ActionSelection(
                action=action,
                reason=(
                    "The bounded investigation execution limit was reached."
                ),
            )
        elif state["unsafe_evidence"] or state["contradiction_reason"]:
            selection = ActionSelection(
                action="handoff",
                reason="Engineer review is required to avoid an unsupported conclusion.",
            )
        elif pending_retry(state) is not None:
            retry_tool = pending_retry(state)
            selection = ActionSelection(
                action=cast(str, retry_tool),
                reason="Retrying one transient read-only inspection failure.",
            )
        elif not state["evidence"] and next_available_incident_tool(state) is None:
            selection = ActionSelection(
                action=(
                    "request_clarification"
                    if recommended_missing_source(state) is not None
                    else "handoff"
                ),
                reason="A specific evidence source is needed before diagnosis.",
            )
        else:
            selection: ActionSelection | None = None
            validation_note = None
            for _ in range(MODEL_OUTPUT_ATTEMPTS):
                try:
                    raw_selection = self._model.select_action(
                        build_action_messages(state, validation_note)
                    )
                    selection = ActionSelection.model_validate(raw_selection)
                    break
                except Exception as error:
                    safe_error = _safe_model_error("Action selection", error)
                    errors.append(safe_error)
                    validation_note = safe_error
            if selection is not None:
                model_action = selection.action
            if selection is None:
                fallback = next_available_incident_tool(state)
                fallback_action = fallback or (
                    "request_clarification"
                    if recommended_missing_source(state) is not None
                    else "handoff"
                )
                selection = ActionSelection(
                    action=fallback_action,
                    reason="Using the next bounded public workflow decision.",
                )
            selection, guard_note = normalize_action(state, selection)

        if model_action is None:
            decided_by = "deterministic"
        elif selection.action == model_action:
            decided_by = "model"
        else:
            decided_by = "guard"
        decision: dict[str, Any] = {
            "phase": "selection",
            "action": selection.action,
            "reason": selection.reason,
            "decided_by": decided_by,
            "deviated_from_default": (
                decided_by == "model"
                and default_action is not None
                and selection.action != default_action
            ),
        }
        if guard_note:
            decision["guard_note"] = guard_note
        return {
            "next_action": selection,
            "public_decisions": [*state["public_decisions"], decision],
            "model_errors": errors,
            "route_decision": None,
            "routing_reason": None,
        }

    def execute_tool(
        self, state: InvestigationState
    ) -> dict[str, Any]:
        """Invoke one allowlisted tool with arguments derived only from state."""

        selection = state["next_action"]
        if selection is None or selection.action not in TOOL_REGISTRY:
            raise RuntimeError("A non-allowlisted inspection was blocked.")
        if state["step_count"] >= MAX_TOOL_EXECUTIONS:
            raise RuntimeError("The six-execution limit has been reached.")

        tool_name = cast(DiagnosticToolName, selection.action)
        attempt = state["attempts_by_tool"].get(tool_name, 0) + 1
        tool_input = (
            {
                "query": (
                    f"{state['symptom_assessment'].category} {state['question']}"
                )[:200],
                "top_k": 3,
            }
            if tool_name == "search_runbook"
            else {"submission": state["submission"]}
        )
        try:
            raw_result = TOOL_REGISTRY[tool_name].invoke(tool_input)
            result = ToolResult.model_validate(raw_result)
        except Exception as error:
            result = ToolResult(
                ok=False,
                error=f"Tool invocation failed ({type(error).__name__}).",
                retryable=False,
                source=f"tool:{tool_name}",
            )

        attempts = dict(state["attempts_by_tool"])
        attempts[tool_name] = attempt
        tools_called = list(state["tools_called"])
        if tool_name not in tools_called:
            tools_called.append(tool_name)
        successful = list(state["successful_tools"])
        if result.ok and tool_name not in successful:
            successful.append(tool_name)
        return {
            "attempts_by_tool": attempts,
            "tools_called": tools_called,
            "successful_tools": successful,
            "last_tool_name": tool_name,
            "last_tool_result": result,
            "step_count": state["step_count"] + 1,
            "public_decisions": [
                *state["public_decisions"],
                {
                    "phase": "tool",
                    "tool": tool_name,
                    "attempt": attempt,
                    "ok": result.ok,
                    "retryable": result.retryable,
                },
            ],
        }

    def analyze_evidence(
        self, state: InvestigationState
    ) -> dict[str, Any]:
        """Convert one bounded result into visible evidence or a safe error."""

        tool_name = state["last_tool_name"]
        result = state["last_tool_result"]
        if tool_name is None or result is None:
            raise RuntimeError("No inspection result is available for analysis.")

        attempt = state["attempts_by_tool"][tool_name]
        evidence = list(state["evidence"])
        references = list(state["reference_context"])
        errors = list(state["errors"])
        hypotheses = list(state["suspected_causes"])
        unsafe = state["unsafe_evidence"]

        if result.ok and isinstance(result.data, Mapping):
            removed = result.data.get("untrusted_instructions_removed", 0)
            unsafe = unsafe or (
                isinstance(removed, int)
                and not isinstance(removed, bool)
                and removed > 0
            )
            availability = result.data.get("availability", "provided")
            if tool_name == "search_runbook":
                references.append(
                    {
                        "source": result.source,
                        "summary": "Reference-only runbook guidance was retrieved.",
                        "incident_evidence": False,
                    }
                )
            elif availability == "provided":
                summary = _summarize_result(tool_name, result)
                evidence_item = EvidenceItem(
                    evidence_id=f"{tool_name}:{attempt}",
                    source_tool=tool_name,
                    summary=summary,
                    raw_reference=result.source,
                    decisive=_is_decisive(tool_name, summary),
                )
                evidence.append(evidence_item)
                candidate = _candidate_cause(tool_name, summary)
                if candidate and not any(
                    item.get("cause") == candidate for item in hypotheses
                ):
                    hypotheses.append(
                        {
                            "cause": candidate,
                            "basis": f"visible evidence from {tool_name}",
                            "provisional": True,
                        }
                    )
        elif not result.ok:
            errors.append(
                ToolError(
                    tool_name=tool_name,
                    message=(result.error or "Inspection failed.")[:500],
                    retryable=result.retryable,
                    attempt=attempt,
                )
            )

        return {
            "evidence": evidence,
            "reference_context": references,
            "errors": errors,
            "suspected_causes": hypotheses,
            "unsafe_evidence": unsafe,
        }

    def route_investigation(
        self, state: InvestigationState
    ) -> dict[str, Any]:
        """Record a deterministic public route after each inspection."""

        contradiction = contradiction_reason(state)
        decision, reason = decide_after_evidence(state)
        return {
            "contradiction_reason": contradiction,
            "route_decision": decision,
            "routing_reason": reason,
            "public_decisions": [
                *state["public_decisions"],
                {"phase": "routing", "decision": decision, "reason": reason},
            ],
        }

    def request_clarification(
        self, state: InvestigationState
    ) -> dict[str, Any]:
        """Return one concrete request instead of inspecting absent evidence."""

        source = recommended_missing_source(state)
        if source is None:
            raise RuntimeError("No specific missing source is available to request.")
        label = _SOURCE_LABELS[source]
        request = ClarificationRequest(
            question=f"Please provide {label} for this workload.",
            reason=(
                "The current incident evidence is insufficient, and this is the "
                "most relevant missing source for the reported symptom."
            ),
            requested_sources=[source],
        )
        return {
            "clarification_request": request,
            "status": "needs_clarification",
            "final_outcome": "needs_clarification",
            "public_decisions": [
                *state["public_decisions"],
                {
                    "phase": "clarification",
                    "requested_sources": list(request.requested_sources),
                    "reason": request.reason,
                },
            ],
        }

    def produce_diagnosis(
        self, state: InvestigationState
    ) -> dict[str, Any]:
        """Validate a high-confidence diagnosis against cited visible evidence."""

        errors = list(state["model_errors"])
        diagnosis: DiagnosisResult | None = None
        validation_note = None
        evidence_by_id = {item.evidence_id: item for item in state["evidence"]}

        for _ in range(MODEL_OUTPUT_ATTEMPTS):
            try:
                raw_diagnosis = self._model.produce_diagnosis(
                    build_diagnosis_messages(state, validation_note)
                )
                candidate = DiagnosisResult.model_validate(raw_diagnosis)
            except Exception as error:
                safe_error = _safe_model_error("Diagnosis", error)
                errors.append(safe_error)
                validation_note = safe_error
                continue

            if not set(candidate.evidence_ids).issubset(evidence_by_id):
                errors.append("Diagnosis citation validation failed.")
                validation_note = (
                    "Use only exact evidence_id values listed in "
                    "citation_requirements.available_evidence_ids."
                )
                continue

            cited = [evidence_by_id[item] for item in candidate.evidence_ids]
            if not has_sufficient_evidence(state, cited):
                errors.append("Diagnosis citation support was incomplete.")
                validation_note = (
                    "Cite all evidence IDs required for support, including at "
                    "least two compatible independent incident sources and "
                    "matching context for any decisive runtime signal."
                )
                continue

            diagnosis = candidate
            break

        if diagnosis is None:
            return {
                "model_errors": errors,
                "diagnosis": None,
                "confidence": 0.0,
                "recommended_action": None,
                "handoff_reason": (
                    "The analysis model could not return a valid evidence-cited "
                    "diagnosis after two attempts."
                ),
                "status": "awaiting_human",
            }
        if diagnosis.confidence < MIN_DIAGNOSIS_CONFIDENCE:
            return {
                "model_errors": errors,
                "diagnosis": None,
                "confidence": diagnosis.confidence,
                "recommended_action": None,
                "handoff_reason": "Diagnosis confidence is below 0.80.",
                "status": "awaiting_human",
            }
        return {
            "model_errors": errors,
            "diagnosis": diagnosis,
            "confidence": diagnosis.confidence,
            "recommended_action": diagnosis.recommended_action,
            "handoff_reason": None,
            "handoff": None,
            "status": "diagnosed",
            "final_outcome": "diagnosed",
        }

    def prepare_handoff(
        self, state: InvestigationState
    ) -> dict[str, Any]:
        """Create a JSON-serializable summary before pausing for an engineer."""

        reason = (
            state["handoff_reason"]
            or state["contradiction_reason"]
            or state["routing_reason"]
            or "Available evidence cannot safely support a diagnosis."
        )
        missing = recommended_missing_source(state)
        manual_check = (
            f"An engineer should obtain and inspect {_SOURCE_LABELS[missing]}."
            if missing is not None
            else (
                "An engineer should compare the complete runtime evidence and "
                "verified workload configuration before deciding on any change."
            )
        )
        handoff = HandoffPayload(
            demo_id=state["submission"].demo_id,
            question=state["question"],
            workload_name=state["workload_name"] or "Not provided",
            namespace=state["namespace"] or "Not provided",
            initial_symptom=state["question"],
            evidence=state["evidence"],
            tools_attempted=state["tools_called"],
            tool_errors=state["errors"],
            leading_hypotheses=[
                str(item.get("cause"))
                for item in state["suspected_causes"]
                if item.get("cause")
            ],
            handoff_reason=reason,
            unresolved_question=(
                "Which verified runtime cause explains the deployment failure?"
            ),
            recommended_manual_check=manual_check,
            public_decisions=state["public_decisions"],
        )
        return {
            "handoff_reason": reason,
            "handoff": handoff,
            "human_acknowledged": False,
            "status": "awaiting_human",
        }

    def human_handoff(
        self, state: InvestigationState
    ) -> dict[str, Any]:
        """Pause until an engineer explicitly acknowledges the safe handoff."""

        if state["handoff"] is None:
            raise RuntimeError("A HandoffPayload is required before interruption.")
        payload = state["handoff"].model_dump(mode="json")
        acknowledgement = interrupt(payload)
        while not (
            isinstance(acknowledgement, Mapping)
            and acknowledgement.get("acknowledged") is True
        ):
            acknowledgement = interrupt(payload)
        return {
            "human_acknowledged": True,
            "status": "escalated",
            "final_outcome": "escalated",
        }

    def format_result(
        self, state: InvestigationState
    ) -> dict[str, Any]:
        """Clear transient execution fields without removing public evidence."""

        return {
            "next_action": None,
            "last_tool_name": None,
            "last_tool_result": None,
            "route_decision": None,
        }


__all__ = ["InvestigationNodes", "TOOL_REGISTRY"]
