"""Polished one-page interface for guided deployment investigations."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from typing import Any, Final
from uuid import uuid4

import streamlit as st
from langgraph.types import Command

from agent.graph import (
    build_investigation_graph,
    graph_input,
    investigation_config,
)
from agent.model import NebiusConfigurationError
from agent.schemas import (
    ClarificationRequest,
    DiagnosisResult,
    EvidenceItem,
    HandoffPayload,
    SubmissionValidationError,
    SubmittedInvestigation,
    ToolError,
)
from tools import load_case_metadata


SOURCE_LABELS: Final[dict[str, str]] = {
    "inspect_workload_status": "Workload and pod status",
    "inspect_kubernetes_events": "Kubernetes events",
    "inspect_container_logs": "Container logs",
    "inspect_manifest_config": "Manifest configuration",
    "search_runbook": "Troubleshooting runbook",
}
ACTION_LABELS: Final[dict[str, str]] = {
    **SOURCE_LABELS,
    "diagnose": "Prepare diagnosis",
    "request_clarification": "Request more evidence",
    "handoff": "Request engineer review",
}
PROGRESS_LABELS: Final[dict[str, str]] = {
    "validate_input": "Validated and sanitized the submission",
    "assess_question": "Assessed the reported deployment symptom",
    "select_next_action": "Selected the next safe diagnostic step",
    "execute_tool": "Inspected one read-only evidence source",
    "analyze_evidence": "Checked the latest evidence",
    "route_investigation": "Evaluated whether the evidence is sufficient",
    "request_clarification": "Prepared a focused evidence request",
    "produce_diagnosis": "Prepared an evidence-grounded diagnosis",
    "prepare_handoff": "Prepared context for engineer review",
    "human_handoff": "Waiting for engineer acknowledgment",
    "format_result": "Finalized the investigation result",
}
PROVIDER_SETTING_NAMES: Final[tuple[str, str]] = (
    "NEBIUS_API_KEY",
    "NEBIUS_MODEL",
)
FORM_KEYS: Final[tuple[str, ...]] = (
    "form_question",
    "form_workload_name",
    "form_namespace",
    "form_workload_status",
    "form_kubernetes_events",
    "form_container_logs",
    "form_manifest_yaml",
    "form_demo_id",
    "form_demo_label",
)
SESSION_KEYS: Final[tuple[str, ...]] = (
    "investigation_graph",
    "investigation_result",
    "investigation_thread_id",
    "investigation_error",
    "investigation_processing",
    "last_submission_fingerprint",
    "pending_sanitized_form",
    *FORM_KEYS,
)
DEMONSTRATIONS: Final[dict[str, tuple[str, str]]] = {
    "Normal diagnosis": ("case_003", "Startup crash with diagnostic logs"),
    "Retry and recovery": ("case_009", "Image pull failure with one event timeout"),
    "Engineer handoff": ("case_010", "Incomplete startup evidence requiring review"),
}


def _apply_streamlit_provider_secrets() -> None:
    """Expose root-level Streamlit secrets as provider environment variables."""

    for name in PROVIDER_SETTING_NAMES:
        if os.environ.get(name, "").strip():
            continue
        try:
            value = st.secrets[name]
        except (FileNotFoundError, KeyError):
            continue
        if isinstance(value, str) and value.strip():
            os.environ[name] = value.strip()


def _tool_label(tool_name: str) -> str:
    return ACTION_LABELS.get(tool_name, "Diagnostic step")


def _trace_rows(public_decisions: list[dict[str, Any]]) -> list[str]:
    """Create a public trace without model rationale or internal errors."""

    rows: list[str] = []
    for entry in public_decisions:
        phase = entry.get("phase")
        if phase == "assessment":
            rows.append("Question assessed; reported text remains unverified context")
        elif phase == "selection":
            label = _tool_label(str(entry.get("action", "")))
            decided_by = entry.get("decided_by")
            if decided_by == "guard":
                note = "safety guard replaced the model's proposal"
            elif decided_by == "model":
                note = (
                    "model choice (differs from default order)"
                    if entry.get("deviated_from_default") is True
                    else "model choice (matches default order)"
                )
            else:
                note = "deterministic workflow step"
            rows.append(f"Selected: {label} — {note}")
        elif phase == "tool":
            tool_name = str(entry.get("tool", ""))
            attempt = entry.get("attempt", "?")
            if entry.get("ok") is True:
                outcome = "completed"
            elif entry.get("retryable") is True:
                outcome = "temporary failure; one retry allowed"
            else:
                outcome = "could not be inspected; no retry"
            rows.append(
                f"Attempt {attempt}: {_tool_label(tool_name)} — {outcome}"
            )
        elif phase == "routing":
            decision = str(entry.get("decision", ""))
            labels = {
                "select": "More evidence was needed",
                "clarify": "A specific missing source was identified",
                "diagnose": "The evidence threshold was met",
                "handoff": "Engineer review was required",
            }
            rows.append(labels.get(decision, "The evidence was reviewed"))
        elif phase == "clarification":
            rows.append("Prepared a focused request for missing evidence")
    return rows


def _set_result_from_checkpoint(graph: Any, config: dict[str, Any]) -> None:
    snapshot = graph.get_state(config)
    st.session_state.investigation_result = dict(snapshot.values)


def _submission_fingerprint(submission: SubmittedInvestigation) -> str:
    rendered = submission.model_dump_json(exclude={"redaction_markers"})
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _schedule_sanitized_form(submission: SubmittedInvestigation) -> None:
    """Replace retained widget values on the next rerun with sanitized values."""

    st.session_state.pending_sanitized_form = {
        "form_question": submission.question,
        "form_workload_name": submission.workload_name or "",
        "form_namespace": submission.namespace or "",
        "form_workload_status": submission.workload_status or "",
        "form_kubernetes_events": submission.kubernetes_events or "",
        "form_container_logs": submission.container_logs or "",
        "form_manifest_yaml": submission.manifest_yaml or "",
    }


def _apply_pending_sanitized_form() -> None:
    pending = st.session_state.pop("pending_sanitized_form", None)
    if isinstance(pending, Mapping):
        for key, value in pending.items():
            if key in FORM_KEYS and isinstance(value, str):
                st.session_state[key] = value


def _load_demonstration(label: str) -> None:
    """Select one corpus case; its evidence is read from disk when inspected."""

    case_id, description = DEMONSTRATIONS[label]
    metadata = load_case_metadata(case_id)
    st.session_state.form_question = metadata.initial_symptom
    st.session_state.form_workload_name = metadata.workload_name
    st.session_state.form_namespace = metadata.namespace
    st.session_state.form_workload_status = ""
    st.session_state.form_kubernetes_events = ""
    st.session_state.form_container_logs = ""
    st.session_state.form_manifest_yaml = ""
    st.session_state.form_demo_id = case_id
    st.session_state.form_demo_label = f"{label}: {description}"
    st.session_state.investigation_error = None


def _clear_demo_on_edit() -> None:
    st.session_state.form_demo_id = None
    st.session_state.form_demo_label = None


def _form_payload() -> dict[str, Any]:
    return {
        "question": st.session_state.get("form_question", ""),
        "workload_name": st.session_state.get("form_workload_name") or None,
        "namespace": st.session_state.get("form_namespace") or None,
        "workload_status": st.session_state.get("form_workload_status") or None,
        "kubernetes_events": st.session_state.get("form_kubernetes_events") or None,
        "container_logs": st.session_state.get("form_container_logs") or None,
        "manifest_yaml": st.session_state.get("form_manifest_yaml") or None,
        "demo_id": st.session_state.get("form_demo_id") or None,
    }


def _run_investigation(payload: dict[str, Any]) -> None:
    """Validate input, create one checkpointed graph, and stream public progress."""

    st.session_state.investigation_error = None
    st.session_state.investigation_processing = True
    try:
        validated_input = graph_input(payload)
        submission = validated_input["submission"]
        fingerprint = _submission_fingerprint(submission)
        if (
            fingerprint == st.session_state.get("last_submission_fingerprint")
            and st.session_state.get("investigation_result") is not None
        ):
            return

        _schedule_sanitized_form(submission)
        _apply_streamlit_provider_secrets()
        thread_id = str(uuid4())
        graph = build_investigation_graph()
        config = investigation_config(thread_id)
        st.session_state.investigation_thread_id = thread_id
        st.session_state.investigation_graph = graph

        with st.status("Analyzing deployment failure…", expanded=True) as status:
            progress = st.progress(0, text="Starting the guided investigation")
            completed_updates = 0
            for update in graph.stream(
                validated_input,
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_update in update.items():
                    if node_name == "__interrupt__":
                        continue
                    completed_updates += 1
                    message = PROGRESS_LABELS.get(
                        node_name, "Completed a safe investigation step"
                    )
                    if node_name == "execute_tool" and isinstance(
                        node_update, Mapping
                    ):
                        message = "Inspected " + _tool_label(
                            str(node_update.get("last_tool_name", ""))
                        )
                    st.write(message)
                    progress.progress(
                        min(0.95, completed_updates / 14), text=message
                    )

            _set_result_from_checkpoint(graph, config)
            st.session_state.last_submission_fingerprint = fingerprint
            result = st.session_state.investigation_result
            progress.progress(1.0, text="Investigation ready")
            status_label = {
                "awaiting_human": "Paused for engineer review",
                "needs_clarification": "More evidence is needed",
            }.get(result.get("status"), "Investigation complete")
            status.update(label=status_label, state="complete", expanded=False)
    except SubmissionValidationError as error:
        st.session_state.investigation_error = " ".join(error.issues)
        st.session_state.pop("investigation_graph", None)
        st.session_state.pop("investigation_result", None)
    except NebiusConfigurationError as error:
        st.session_state.investigation_error = str(error)
        st.session_state.pop("investigation_graph", None)
        st.session_state.pop("investigation_result", None)
    except Exception:
        st.session_state.investigation_error = (
            "The investigation could not be completed. Verify the model "
            "configuration and try again."
        )
        st.session_state.pop("investigation_graph", None)
        st.session_state.pop("investigation_result", None)
    finally:
        st.session_state.investigation_processing = False


def _acknowledge_handoff() -> None:
    graph = st.session_state.get("investigation_graph")
    thread_id = st.session_state.get("investigation_thread_id")
    if graph is None or not isinstance(thread_id, str):
        st.session_state.investigation_error = (
            "The saved investigation is unavailable. Start a new investigation."
        )
        return
    try:
        with st.status("Recording engineer acknowledgment…") as status:
            config = investigation_config(thread_id)
            graph.invoke(Command(resume={"acknowledged": True}), config=config)
            _set_result_from_checkpoint(graph, config)
            status.update(
                label="Escalation recorded", state="complete", expanded=False
            )
    except Exception:
        st.session_state.investigation_error = (
            "The acknowledgment could not be recorded. Please try again."
        )


def _reset_investigation() -> None:
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)


def _clear_result_keep_form() -> None:
    """Clear one result while preserving the sanitized user-facing form."""

    result = st.session_state.get("investigation_result")
    if isinstance(result, Mapping):
        try:
            submission = SubmittedInvestigation.model_validate(
                result.get("submission")
            )
        except (TypeError, ValueError):
            pass
        else:
            _schedule_sanitized_form(submission)

    for key in (
        "investigation_graph",
        "investigation_result",
        "investigation_thread_id",
        "investigation_error",
        "last_submission_fingerprint",
    ):
        st.session_state.pop(key, None)


def _retry_analysis() -> None:
    """Retry the sanitized submission without making the user enter it again."""

    result = st.session_state.get("investigation_result")
    submission = None
    if isinstance(result, Mapping):
        try:
            submission = SubmittedInvestigation.model_validate(
                result.get("submission")
            )
        except (TypeError, ValueError):
            pass
    _clear_result_keep_form()
    if submission is None:
        _run_investigation(_form_payload())
    else:
        _run_investigation(submission.model_dump(exclude={"redaction_markers"}))


def _render_badge(status: str) -> None:
    label, css_class = {
        "diagnosed": ("Diagnosis supported", "badge-success"),
        "needs_clarification": ("More evidence needed", "badge-info"),
        "awaiting_human": ("Needs engineer review", "badge-warning"),
        "escalated": ("Escalated for engineer review", "badge-warning"),
        "failed": ("Investigation incomplete", "badge-neutral"),
    }.get(status, ("Investigation in progress", "badge-neutral"))
    st.markdown(
        f'<span class="outcome-badge {css_class}">{label}</span>',
        unsafe_allow_html=True,
    )


def _render_reported_problem(state: dict[str, Any]) -> None:
    """Keep the sanitized question visible alongside its outcome."""

    question = state.get("question")
    if not isinstance(question, str) or not question:
        return
    st.subheader("Reported problem")
    with st.container(border=True):
        st.write(question)


def _render_timeline(state: dict[str, Any]) -> None:
    st.subheader("Progress timeline")
    rows = _trace_rows(state.get("public_decisions", []))
    if not rows:
        st.caption("No investigation steps were recorded.")
        return
    with st.container(border=True):
        for index, row in enumerate(rows, start=1):
            st.markdown(f"**{index}.** {row}")


def _render_selected_sources(state: dict[str, Any]) -> None:
    st.subheader("Sources selected by the agent")
    tools = state.get("tools_called", [])
    if not tools:
        st.caption("No source was inspected yet.")
        return
    for tool_name in tools:
        attempts = state.get("attempts_by_tool", {}).get(tool_name, 1)
        suffix = f" · {attempts} attempts" if attempts > 1 else ""
        st.markdown(f"- {_tool_label(str(tool_name))}{suffix}")


def _render_evidence(
    evidence: list[EvidenceItem] | list[dict[str, Any]],
    evidence_ids: set[str] | None = None,
) -> None:
    selected = [EvidenceItem.model_validate(item) for item in evidence]
    if evidence_ids is not None:
        selected = [item for item in selected if item.evidence_id in evidence_ids]
    if not selected:
        st.caption("No supporting incident evidence was available.")
        return
    for item in selected:
        with st.container(border=True):
            st.caption(SOURCE_LABELS.get(item.source_tool, "Submitted evidence"))
            st.write(item.summary)


def _render_diagnosis(state: dict[str, Any]) -> None:
    diagnosis = DiagnosisResult.model_validate(state["diagnosis"])
    _render_badge("diagnosed")
    st.subheader("Likely root cause")
    st.write(diagnosis.root_cause)
    st.metric("Confidence", f"{diagnosis.confidence:.0%}")
    st.progress(diagnosis.confidence, text="Evidence-supported confidence")

    st.subheader("Supporting evidence")
    _render_evidence(state.get("evidence", []), set(diagnosis.evidence_ids))
    st.subheader("Recommended manual action")
    st.info(diagnosis.recommended_action)
    st.subheader("Limitations")
    if diagnosis.limitations:
        for limitation in diagnosis.limitations:
            st.markdown(f"- {limitation}")
    else:
        st.caption("No additional limitations were reported.")


def _render_clarification(state: dict[str, Any]) -> None:
    request = ClarificationRequest.model_validate(
        state["clarification_request"]
    )
    _render_badge("needs_clarification")
    st.subheader("Clarification request")
    st.info(request.question)
    st.caption(request.reason)
    st.subheader("Requested evidence")
    for source in request.requested_sources:
        label = {
            "workload_status": "Workload and pod status",
            "kubernetes_events": "Recent Kubernetes warning events",
            "container_logs": "Recent container logs",
            "manifest_yaml": "Relevant workload manifest YAML",
        }[source]
        st.markdown(f"- {label}")
    if state.get("evidence"):
        st.subheader("Evidence already preserved")
        _render_evidence(state["evidence"])
    if st.button(
        "Add Requested Evidence",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.get("investigation_processing", False),
    ):
        _clear_result_keep_form()
        st.rerun()


def _render_tool_failures(errors: list[ToolError] | list[dict[str, Any]]) -> None:
    validated = [ToolError.model_validate(error) for error in errors]
    if not validated:
        st.caption("No read-only inspection failures were recorded.")
        return
    for error in validated:
        retry_note = "retryable" if error.retryable else "not retryable"
        st.markdown(
            f"- **{SOURCE_LABELS.get(error.tool_name, 'Diagnostic source')}**, "
            f"attempt {error.attempt} ({retry_note}): {error.message}"
        )


def _render_handoff(state: dict[str, Any]) -> None:
    handoff = HandoffPayload.model_validate(state["handoff"])
    analysis_retry_available = handoff.handoff_reason.startswith(
        "The analysis model could not return a valid evidence-cited diagnosis"
    )
    if analysis_retry_available:
        st.markdown(
            '<span class="outcome-badge badge-info">Analysis needs retry</span>',
            unsafe_allow_html=True,
        )
        card_title = "Analysis Could Not Be Finalized"
        card_message = (
            "The available evidence was preserved, but the model response did "
            "not meet the required citation format. You can retry without "
            "re-entering the submission."
        )
    else:
        _render_badge(str(state.get("status", "awaiting_human")))
        card_title = "Needs Engineer Review"
        card_message = (
            "The evidence is incomplete, contradictory, or unsafe to interpret "
            "without an engineer."
        )
    st.markdown(
        f'<div class="review-card"><strong>{card_title}</strong><br>'
        f"{card_message}</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Handoff reason")
    st.write(handoff.handoff_reason)
    st.subheader("Preserved evidence")
    _render_evidence(handoff.evidence)
    st.subheader("Tool failures")
    _render_tool_failures(handoff.tool_errors)
    st.subheader("Leading hypotheses")
    if handoff.leading_hypotheses:
        for hypothesis in handoff.leading_hypotheses:
            st.markdown(f"- {hypothesis}")
    else:
        st.caption("No hypothesis is strong enough to present.")
    st.subheader("Unresolved question")
    st.write(handoff.unresolved_question)
    st.subheader("Recommended manual check")
    st.info(handoff.recommended_manual_check)

    if state.get("status") == "awaiting_human":
        if analysis_retry_available and st.button(
            "Retry Analysis",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("investigation_processing", False),
        ):
            _retry_analysis()
            st.rerun()
        if st.button(
            "Acknowledge and End as Escalated",
            type="secondary" if analysis_retry_available else "primary",
            use_container_width=True,
            disabled=st.session_state.get("investigation_processing", False),
        ):
            _acknowledge_handoff()
            st.rerun()
    elif state.get("human_acknowledged") is True:
        st.success("Engineer acknowledgment recorded. Investigation ended as escalated.")


def _render_demo_loaders(disabled: bool) -> None:
    with st.container(border=True):
        st.markdown("#### Try a demonstration")
        st.caption(
            "Each option selects a bundled synthetic case. The agent reads its "
            "evidence from disk as it inspects each source."
        )
        for label in DEMONSTRATIONS:
            if st.button(
                label,
                key=f"demo_{label.casefold().replace(' ', '_')}",
                use_container_width=True,
                disabled=disabled,
            ):
                _load_demonstration(label)
                st.rerun()


def _render_investigation_form() -> None:
    processing = bool(st.session_state.get("investigation_processing", False))
    _render_demo_loaders(processing)

    demo_label = st.session_state.get("form_demo_label")
    if isinstance(demo_label, str):
        st.success(f"Loaded demonstration — {demo_label}")

    st.markdown("#### Describe the failure")
    st.caption(
        "Example: “My checkout-api pods keep restarting with CrashLoopBackOff.”"
    )
    st.text_area(
        "Deployment problem",
        key="form_question",
        height=150,
        placeholder=(
            "Describe what failed, when it started, and any status such as "
            "ImagePullBackOff, CrashLoopBackOff, Pending, or Not Ready."
        ),
        disabled=processing,
        on_change=_clear_demo_on_edit,
    )
    st.text_input(
        "Workload name (optional)",
        key="form_workload_name",
        placeholder="for example, checkout-api",
        disabled=processing,
        on_change=_clear_demo_on_edit,
    )
    st.text_input(
        "Namespace (optional)",
        key="form_namespace",
        placeholder="for example, checkout",
        disabled=processing,
        on_change=_clear_demo_on_edit,
    )

    st.markdown("#### Add available evidence")
    st.caption(
        "Leave unavailable sections blank. Sensitive values are redacted before analysis."
    )
    evidence_fields = (
        (
            "Workload and pod status",
            "form_workload_status",
            "Paste a bounded status or describe output.",
            180,
        ),
        (
            "Kubernetes events",
            "form_kubernetes_events",
            "Paste recent warning events in text or JSON form.",
            200,
        ),
        (
            "Container logs",
            "form_container_logs",
            "Paste the most recent startup or failure log excerpt.",
            220,
        ),
        (
            "Manifest YAML",
            "form_manifest_yaml",
            "Paste the relevant sanitized Deployment, StatefulSet, or Pod YAML.",
            260,
        ),
    )
    for label, key, placeholder, height in evidence_fields:
        with st.expander(label, expanded=False):
            st.text_area(
                label,
                key=key,
                height=height,
                placeholder=placeholder,
                label_visibility="collapsed",
                disabled=processing,
                on_change=_clear_demo_on_edit,
            )

    if st.button(
        "Analyze Deployment Failure",
        type="primary",
        use_container_width=True,
        disabled=processing,
    ):
        _run_investigation(_form_payload())
        st.rerun()


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 860px;
            padding-top: 3.5rem;
            padding-bottom: 3.5rem;
        }
        .block-container h1 {
            font-size: 1.9rem !important;
            line-height: 1.3;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .hero-kicker {
            color: #2563eb;
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .outcome-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.38rem 0.82rem;
            font-size: 0.84rem;
            font-weight: 750;
            margin: 0.15rem 0 1rem 0;
        }
        .badge-success {background: #dcfce7; color: #166534; border: 1px solid #86efac;}
        .badge-info {background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd;}
        .badge-warning {background: #fff7ed; color: #9a3412; border: 1px solid #fdba74;}
        .badge-neutral {background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1;}
        .review-card {
            border: 1px solid #f59e0b;
            border-left: 6px solid #f59e0b;
            border-radius: 0.7rem;
            padding: 1rem 1.1rem;
            margin: 0 0 1.25rem 0;
            background: rgba(245, 158, 11, 0.08);
        }
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input {border-radius: 0.55rem;}
        [data-testid="stButton"] button {border-radius: 0.55rem; font-weight: 650;}
        @media (max-width: 640px) {
            .block-container {padding-top: 3rem; padding-left: 1rem; padding-right: 1rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Kubernetes Deployment Failure Investigator",
        page_icon="🔎",
        layout="centered",
    )
    _apply_pending_sanitized_form()
    _apply_styles()

    st.markdown(
        '<div class="hero-kicker">Guided read-only analysis</div>',
        unsafe_allow_html=True,
    )
    st.title("Kubernetes Deployment Failure Investigator")
    st.write(
        "Describe a deployment problem and add whatever evidence you have. "
        "The agent selects only relevant read-only sources, then diagnoses, "
        "asks for one missing detail, or prepares an engineer handoff."
    )
    st.caption(
        "No live cluster access · No kubectl · No automatic remediation · Engineer review required"
    )

    error_message = st.session_state.get("investigation_error")
    if isinstance(error_message, str) and error_message:
        st.error(error_message)

    result = st.session_state.get("investigation_result")
    if not isinstance(result, dict):
        _render_investigation_form()
        return

    st.divider()
    st.caption("INVESTIGATION OUTCOME")
    _render_reported_problem(result)
    _render_timeline(result)
    _render_selected_sources(result)
    try:
        if result.get("status") == "diagnosed" and result.get("diagnosis") is not None:
            _render_diagnosis(result)
        elif result.get("status") == "needs_clarification":
            _render_clarification(result)
        elif result.get("handoff") is not None:
            _render_handoff(result)
        else:
            _render_badge(str(result.get("status", "failed")))
            st.warning("The investigation ended without a supported result.")
    except Exception:
        st.error(
            "The saved result could not be displayed safely. Start a new investigation."
        )

    st.divider()
    if st.button("Start New Investigation", use_container_width=True):
        _reset_investigation()
        st.rerun()


if __name__ == "__main__":
    main()
