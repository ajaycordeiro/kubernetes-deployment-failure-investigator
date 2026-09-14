"""Five read-only LangChain tools for sanitized deployment evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from threading import Lock
from typing import Annotated, Any, Final, Literal, TypeAlias

import yaml
from langchain_core.tools import InjectedToolArg, tool
from pydantic import ValidationError, field_validator

from agent.input import redact_sensitive_text
from agent.schemas import StrictModel, SubmittedInvestigation, ToolResult
from tools.base import FixtureLoadError, load_fixture
from tools.manifest import select_diagnostic_config
from tools.safety import (
    bound_text,
    neutralize_untrusted_text,
    neutralize_untrusted_value,
)


DiagnosticToolName: TypeAlias = Literal[
    "inspect_workload_status",
    "inspect_kubernetes_events",
    "inspect_container_logs",
    "inspect_manifest_config",
    "search_runbook",
]

DIAGNOSTIC_TOOL_ALLOWLIST: Final[tuple[DiagnosticToolName, ...]] = (
    "inspect_workload_status",
    "inspect_kubernetes_events",
    "inspect_container_logs",
    "inspect_manifest_config",
    "search_runbook",
)
MAX_STATUS_RESULT_CHARS: Final[int] = 6_000
MAX_EVENTS_RESULT_CHARS: Final[int] = 8_000
MAX_LOG_RESULT_CHARS: Final[int] = 4_000
MAX_MANIFEST_RESULT_CHARS: Final[int] = 8_000
MAX_RESULT_LINES: Final[int] = 120
_DEMO_EVENT_CALL_COUNTS: dict[str, int] = {}
_DEMO_EVENT_CALL_LOCK = Lock()


class SubmittedInspectionInput(StrictModel):
    """Graph-injected input hidden from the model-facing tool schema."""

    submission: Annotated[SubmittedInvestigation, InjectedToolArg]

    @field_validator("submission", mode="before")
    @classmethod
    def require_validated_submission(cls, value: object) -> object:
        """Reject model-generated dictionaries at the final tool boundary."""

        if not isinstance(value, SubmittedInvestigation):
            raise ValueError(
                "A validated graph-state submission is required."
            )
        return value


def _canonical_submission(
    submission: SubmittedInvestigation,
) -> SubmittedInvestigation:
    """Revalidate a copied payload so unsafe model construction cannot bypass guards."""

    payload = submission.model_dump(exclude={"redaction_markers"})
    return SubmittedInvestigation.model_validate(payload)


def _invalid_submission_result(capability: str) -> ToolResult:
    return ToolResult(
        ok=False,
        error="invalid_submission: Validated sanitized evidence is required.",
        retryable=False,
        source=f"submitted_evidence/{capability}",
    )


def _absent_result(capability: str) -> ToolResult:
    """Represent missing optional evidence as data, not as a tool failure."""

    return ToolResult(
        ok=True,
        data={
            "availability": "absent",
            "clarification_needed": True,
            "requested_evidence": capability,
            "content": "",
            "truncated": False,
            "untrusted_instructions_removed": 0,
        },
        retryable=False,
        source=f"submitted_evidence/{capability}",
    )


def _unavailable_result(capability: str, source: str) -> ToolResult:
    """Represent explicitly unavailable evidence as a valid clarification signal."""

    return ToolResult(
        ok=True,
        data={
            "availability": "unavailable",
            "clarification_needed": True,
            "requested_evidence": capability,
            "content": "",
            "truncated": False,
            "untrusted_instructions_removed": 0,
        },
        retryable=False,
        source=source,
    )


def _malformed_result(capability: str, detail: str) -> ToolResult:
    return ToolResult(
        ok=False,
        error=f"malformed_content: Submitted {capability} {detail}.",
        retryable=False,
        source=f"submitted_evidence/{capability}",
    )


def _drop_internal_behavior(value: object) -> object:
    """Remove fixture-control fields if they appear in submitted JSON."""

    if isinstance(value, Mapping):
        return {
            str(key)[:253]: _drop_internal_behavior(item)
            for key, item in value.items()
            if str(key).casefold() != "tool_behavior"
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_drop_internal_behavior(item) for item in value]
    return value


def _safe_error_message(message: str | None) -> str:
    sanitized = redact_sensitive_text(message or "Read-only source failed.")
    neutralized, _ = neutralize_untrusted_text(sanitized)
    bounded, _ = bound_text(
        neutralized,
        maximum_chars=500,
        maximum_lines=4,
    )
    return bounded or "Read-only source failed."


def _reset_demo_event_call_counts() -> None:
    """Reset the approved retry demonstration between isolated runs."""

    with _DEMO_EVENT_CALL_LOCK:
        _DEMO_EVENT_CALL_COUNTS.clear()


def _next_demo_event_call(demo_id: str) -> int:
    with _DEMO_EVENT_CALL_LOCK:
        call_number = _DEMO_EVENT_CALL_COUNTS.get(demo_id, 0) + 1
        _DEMO_EVENT_CALL_COUNTS[demo_id] = call_number
        return call_number


def _demo_source_error(demo_id: str, fixture_name: str) -> ToolResult:
    return ToolResult(
        ok=False,
        error="demo_source_error: Approved demonstration evidence is unavailable.",
        retryable=False,
        source=f"data/cases/{demo_id}/{fixture_name}",
    )


def _load_demo_fixture(
    submission: SubmittedInvestigation,
    fixture_name: str,
) -> object | ToolResult:
    """Load only an approved synthetic demonstration selected in the UI."""

    demo_id = submission.demo_id
    if demo_id is None:
        raise ValueError("An approved demonstration ID is required.")
    try:
        return load_fixture(demo_id, fixture_name)
    except FixtureLoadError:
        return _demo_source_error(demo_id, fixture_name)


def _corpus_declared_fault(
    fixture: Mapping[str, Any],
    *,
    demo_id: str,
    fixture_name: str,
    call_number: int,
) -> ToolResult | None:
    """Honor a fault that a synthetic corpus fixture declares for itself.

    Fault injection is data-driven so that no case identifier appears in
    runtime code; a fixture opts in by declaring a `tool_behavior` block.
    """

    behavior = fixture.get("tool_behavior")
    if not isinstance(behavior, Mapping):
        return None
    failures = behavior.get("fail_first_attempts", 0)
    if (
        not isinstance(failures, int)
        or isinstance(failures, bool)
        or call_number > failures
    ):
        return None
    return ToolResult(
        ok=False,
        error=_safe_error_message(
            str(behavior.get("error_message", "Synthetic source timeout."))
        ),
        retryable=behavior.get("retryable") is True,
        source=f"data/cases/{demo_id}/{fixture_name}",
    )


def _demo_value_result(
    value: object,
    *,
    capability: str,
    source: str,
    maximum_chars: int,
    keep_recent: bool = False,
) -> ToolResult:
    """Convert a loaded fixture into the same bounded result as pasted evidence."""

    if isinstance(value, str):
        sanitized = redact_sensitive_text(value)
        neutralized, removed = neutralize_untrusted_text(sanitized)
        content = neutralized
        content_format = "text"
    else:
        safe_value = _drop_internal_behavior(value)
        neutralized_value, removed = neutralize_untrusted_value(safe_value)
        content = json.dumps(
            neutralized_value,
            ensure_ascii=True,
            sort_keys=True,
        )
        content_format = "json"
    return _provided_result(
        capability=capability,
        content=content,
        content_format=content_format,
        maximum_chars=maximum_chars,
        source=source,
        keep_recent=keep_recent,
        removed_instructions=removed,
        mode="demo",
    )


def _provided_result(
    *,
    capability: str,
    content: str,
    content_format: str,
    maximum_chars: int,
    source: str | None = None,
    keep_recent: bool = False,
    removed_instructions: int = 0,
    mode: Literal["submitted", "demo"] = "submitted",
) -> ToolResult:
    bounded, truncated = bound_text(
        content,
        maximum_chars=maximum_chars,
        maximum_lines=MAX_RESULT_LINES,
        keep_recent=keep_recent,
    )
    return ToolResult(
        ok=True,
        data={
            "availability": "provided",
            "mode": mode,
            "format": content_format,
            "content": bounded,
            "truncated": truncated,
            "untrusted_instructions_removed": removed_instructions,
        },
        retryable=False,
        source=source or f"submitted_evidence/{capability}",
    )


def _parse_optional_json(
    content: str,
    *,
    capability: str,
) -> tuple[object | None, ToolResult | None]:
    stripped = content.lstrip()
    if not stripped.startswith(("{", "[")):
        return None, None
    try:
        return json.loads(content), None
    except (ValueError, RecursionError):
        return None, _malformed_result(
            capability, "looks like JSON but could not be parsed"
        )


def _inspect_submitted_text(
    content: str,
    *,
    capability: str,
    maximum_chars: int,
    keep_recent: bool = False,
    expected_json: Literal["mapping", "events"] | None = None,
) -> ToolResult:
    parsed, parse_error = _parse_optional_json(
        content,
        capability=capability,
    )
    if parse_error is not None:
        return parse_error

    if parsed is not None:
        if expected_json == "mapping" and not isinstance(parsed, Mapping):
            return _malformed_result(capability, "must contain a JSON object")
        if expected_json == "events":
            if isinstance(parsed, Mapping):
                items = parsed.get("items")
                if not isinstance(items, list):
                    return _malformed_result(
                        capability, "JSON must contain an items list"
                    )
            elif not isinstance(parsed, list):
                return _malformed_result(
                    capability, "must contain an event list"
                )
        safe_value = _drop_internal_behavior(parsed)
        neutralized, removed = neutralize_untrusted_value(safe_value)
        rendered = json.dumps(
            neutralized,
            ensure_ascii=True,
            sort_keys=True,
        )
        return _provided_result(
            capability=capability,
            content=rendered,
            content_format="json",
            maximum_chars=maximum_chars,
            keep_recent=keep_recent,
            removed_instructions=removed,
        )

    sanitized = redact_sensitive_text(content)
    neutralized_text, removed = neutralize_untrusted_text(sanitized)
    return _provided_result(
        capability=capability,
        content=neutralized_text,
        content_format="text",
        maximum_chars=maximum_chars,
        keep_recent=keep_recent,
        removed_instructions=removed,
    )


@tool("inspect_workload_status", args_schema=SubmittedInspectionInput)
def inspect_workload_status(
    submission: Annotated[SubmittedInvestigation, InjectedToolArg],
) -> ToolResult:
    """Inspect bounded workload status injected from validated graph state."""

    try:
        canonical = _canonical_submission(submission)
    except (ValidationError, ValueError):
        return _invalid_submission_result("workload_status")
    if canonical.workload_status is not None:
        return _inspect_submitted_text(
            canonical.workload_status,
            capability="workload_status",
            maximum_chars=MAX_STATUS_RESULT_CHARS,
            expected_json="mapping",
        )
    if canonical.demo_id is not None:
        fixture = _load_demo_fixture(canonical, "workload_status.json")
        if isinstance(fixture, ToolResult):
            return fixture
        return _demo_value_result(
            fixture,
            capability="workload_status",
            source=f"data/cases/{canonical.demo_id}/workload_status.json",
            maximum_chars=MAX_STATUS_RESULT_CHARS,
        )
    return _absent_result("workload_status")


@tool("inspect_kubernetes_events", args_schema=SubmittedInspectionInput)
def inspect_kubernetes_events(
    submission: Annotated[SubmittedInvestigation, InjectedToolArg],
) -> ToolResult:
    """Inspect bounded Kubernetes events injected from validated graph state."""

    try:
        canonical = _canonical_submission(submission)
    except (ValidationError, ValueError):
        return _invalid_submission_result("kubernetes_events")
    if canonical.kubernetes_events is not None:
        return _inspect_submitted_text(
            canonical.kubernetes_events,
            capability="kubernetes_events",
            maximum_chars=MAX_EVENTS_RESULT_CHARS,
            expected_json="events",
        )
    if canonical.demo_id is not None:
        fixture = _load_demo_fixture(canonical, "events.json")
        if isinstance(fixture, ToolResult):
            return fixture
        if not isinstance(fixture, Mapping) or not isinstance(
            fixture.get("items"), list
        ):
            return _malformed_result(
                "kubernetes_events", "must contain an items list"
            )
        fault = _corpus_declared_fault(
            fixture,
            demo_id=canonical.demo_id,
            fixture_name="events.json",
            call_number=_next_demo_event_call(canonical.demo_id),
        )
        if fault is not None:
            return fault
        return _demo_value_result(
            fixture,
            capability="kubernetes_events",
            source=f"data/cases/{canonical.demo_id}/events.json",
            maximum_chars=MAX_EVENTS_RESULT_CHARS,
        )
    return _absent_result("kubernetes_events")


@tool("inspect_container_logs", args_schema=SubmittedInspectionInput)
def inspect_container_logs(
    submission: Annotated[SubmittedInvestigation, InjectedToolArg],
) -> ToolResult:
    """Inspect a bounded recent log excerpt injected from validated graph state."""

    try:
        canonical = _canonical_submission(submission)
    except (ValidationError, ValueError):
        return _invalid_submission_result("container_logs")
    if canonical.container_logs is not None:
        if canonical.container_logs.strip().casefold().startswith(
            "[logs unavailable:"
        ):
            return _unavailable_result(
                "container_logs", "submitted_evidence/container_logs"
            )
        return _inspect_submitted_text(
            canonical.container_logs,
            capability="container_logs",
            maximum_chars=MAX_LOG_RESULT_CHARS,
            keep_recent=True,
        )
    if canonical.demo_id is not None:
        fixture = _load_demo_fixture(canonical, "logs.txt")
        if isinstance(fixture, ToolResult):
            return fixture
        if not isinstance(fixture, str):
            return _malformed_result("container_logs", "must contain text")
        if fixture.strip().casefold().startswith("[logs unavailable:"):
            return _unavailable_result(
                "container_logs",
                f"data/cases/{canonical.demo_id}/logs.txt",
            )
        return _demo_value_result(
            fixture,
            capability="container_logs",
            source=f"data/cases/{canonical.demo_id}/logs.txt",
            maximum_chars=MAX_LOG_RESULT_CHARS,
            keep_recent=True,
        )
    return _absent_result("container_logs")


@tool("inspect_manifest_config", args_schema=SubmittedInspectionInput)
def inspect_manifest_config(
    submission: Annotated[SubmittedInvestigation, InjectedToolArg],
) -> ToolResult:
    """Inspect diagnosis-relevant configuration from validated manifest YAML."""

    try:
        canonical = _canonical_submission(submission)
    except (ValidationError, ValueError):
        return _invalid_submission_result("manifest_yaml")
    if canonical.manifest_yaml is not None:
        try:
            documents = list(yaml.safe_load_all(canonical.manifest_yaml))
        except (yaml.YAMLError, RecursionError):
            return _malformed_result(
                "manifest_yaml", "could not be parsed as YAML"
            )
        if not documents or any(
            not isinstance(document, Mapping) for document in documents
        ):
            return _malformed_result(
                "manifest_yaml", "must contain mapping documents"
            )
        selected = [
            select_diagnostic_config(document) for document in documents
        ]
        neutralized, removed = neutralize_untrusted_value(selected)
        return _provided_result(
            capability="manifest_yaml",
            content=json.dumps(
                neutralized,
                ensure_ascii=True,
                sort_keys=True,
            ),
            content_format="json",
            maximum_chars=MAX_MANIFEST_RESULT_CHARS,
            removed_instructions=removed,
        )
    if canonical.demo_id is not None:
        fixture = _load_demo_fixture(canonical, "manifest.yaml")
        if isinstance(fixture, ToolResult):
            return fixture
        if not isinstance(fixture, Mapping):
            return _malformed_result(
                "manifest_yaml", "must contain a mapping document"
            )
        selected = select_diagnostic_config(fixture)
        return _demo_value_result(
            selected,
            capability="manifest_yaml",
            source=f"data/cases/{canonical.demo_id}/manifest.yaml",
            maximum_chars=MAX_MANIFEST_RESULT_CHARS,
        )
    return _absent_result("manifest_yaml")


__all__ = [
    "MAX_EVENTS_RESULT_CHARS",
    "MAX_LOG_RESULT_CHARS",
    "MAX_MANIFEST_RESULT_CHARS",
    "MAX_RESULT_LINES",
    "MAX_STATUS_RESULT_CHARS",
    "DIAGNOSTIC_TOOL_ALLOWLIST",
    "SubmittedInspectionInput",
    "DiagnosticToolName",
    "_reset_demo_event_call_counts",
    "inspect_container_logs",
    "inspect_kubernetes_events",
    "inspect_manifest_config",
    "inspect_workload_status",
]
