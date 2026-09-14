"""Validated data contracts shared by the investigation workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)

from agent.input import (
    MAX_CONTAINER_LOGS_CHARS,
    MAX_KUBERNETES_EVENTS_CHARS,
    MAX_MANIFEST_YAML_CHARS,
    MAX_NAMESPACE_CHARS,
    MAX_QUESTION_CHARS,
    MAX_WORKLOAD_NAME_CHARS,
    MAX_WORKLOAD_STATUS_CHARS,
    ManifestInputError,
    is_meaningful_question,
    normalize_inline_text,
    normalize_multiline_text,
    redact_sensitive_text,
    redaction_markers,
    validate_and_redact_manifest,
)


CaseId: TypeAlias = Literal[
    "case_001",
    "case_002",
    "case_003",
    "case_004",
    "case_005",
    "case_006",
    "case_007",
    "case_008",
    "case_009",
    "case_010",
]

# Demonstrations are intentionally limited to the existing, reviewed corpus.
DemoId: TypeAlias = CaseId

EvidenceSourceName: TypeAlias = Literal[
    "workload_status",
    "kubernetes_events",
    "container_logs",
    "manifest_yaml",
]

SymptomCategory: TypeAlias = Literal[
    "image_pull",
    "startup_crash",
    "configuration",
    "readiness",
    "scheduling",
    "storage",
    "unknown",
]

ToolName: TypeAlias = Literal[
    "inspect_workload_status",
    "inspect_kubernetes_events",
    "inspect_container_logs",
    "inspect_manifest_config",
    "search_runbook",
]

ActionName: TypeAlias = Literal[
    "inspect_workload_status",
    "inspect_kubernetes_events",
    "inspect_container_logs",
    "inspect_manifest_config",
    "search_runbook",
    "diagnose",
    "request_clarification",
    "handoff",
]

InvestigationStatus: TypeAlias = Literal[
    "new",
    "investigating",
    "diagnosed",
    "needs_clarification",
    "awaiting_human",
    "escalated",
    "failed",
]

StructuredData: TypeAlias = dict[str, Any] | list[Any] | str | None


class StrictModel(BaseModel):
    """Base model that rejects unexpected fields and trims strings."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CaseMetadata(StrictModel):
    """Validated user-visible metadata for one synthetic case."""

    case_id: CaseId
    workload_name: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    container_name: str = Field(min_length=1)
    initial_symptom: str = Field(min_length=1)


def _bounded_inline_text(
    value: object,
    *,
    maximum: int,
    optional: bool,
) -> object:
    """Bound and sanitize one inline user-controlled field."""

    if not isinstance(value, str):
        return value
    if len(value) > maximum:
        raise ValueError("Input exceeds the permitted maximum length.")
    normalized = normalize_inline_text(value)
    if not normalized and optional:
        return None
    return redact_sensitive_text(normalized)


def _bounded_multiline_text(
    value: object,
    *,
    maximum: int,
    optional: bool,
) -> object:
    """Bound and sanitize one multiline user-controlled field."""

    if not isinstance(value, str):
        return value
    if len(value) > maximum:
        raise ValueError("Input exceeds the permitted maximum length.")
    normalized = normalize_multiline_text(value)
    if not normalized and optional:
        return None
    return redact_sensitive_text(normalized)


def _field_redaction_markers(
    fields: Mapping[str, str | None],
) -> tuple[str, ...]:
    """Describe sanitized fields without retaining any removed values."""

    markers: list[str] = []
    for field_name, value in fields.items():
        markers.extend(
            f"{field_name}:{marker}" for marker in redaction_markers(value)
        )
    return tuple(dict.fromkeys(markers))


class EvidenceAvailability(StrictModel):
    """Which optional evidence sources were included in a submission."""

    workload_status: bool = False
    kubernetes_events: bool = False
    container_logs: bool = False
    manifest_yaml: bool = False


class SubmittedEvidence(StrictModel):
    """Bounded, redacted evidence supplied by an engineer."""

    workload_status: str | None = Field(
        default=None, max_length=MAX_WORKLOAD_STATUS_CHARS
    )
    kubernetes_events: str | None = Field(
        default=None, max_length=MAX_KUBERNETES_EVENTS_CHARS
    )
    container_logs: str | None = Field(
        default=None, max_length=MAX_CONTAINER_LOGS_CHARS
    )
    manifest_yaml: str | None = Field(
        default=None, max_length=MAX_MANIFEST_YAML_CHARS
    )

    @field_validator(
        "workload_status",
        "kubernetes_events",
        "container_logs",
        mode="before",
    )
    @classmethod
    def sanitize_evidence_text(cls, value: object, info: Any) -> object:
        """Normalize and redact evidence before it reaches model storage."""

        limits = {
            "workload_status": MAX_WORKLOAD_STATUS_CHARS,
            "kubernetes_events": MAX_KUBERNETES_EVENTS_CHARS,
            "container_logs": MAX_CONTAINER_LOGS_CHARS,
        }
        return _bounded_multiline_text(
            value,
            maximum=limits[info.field_name],
            optional=True,
        )

    @field_validator("manifest_yaml", mode="before")
    @classmethod
    def sanitize_manifest(cls, value: object) -> object:
        """Validate YAML and redact credentials and Secret values."""

        if not isinstance(value, str):
            return value
        if len(value) > MAX_MANIFEST_YAML_CHARS:
            raise ValueError("Input exceeds the permitted maximum length.")
        if not normalize_multiline_text(value):
            return None
        return validate_and_redact_manifest(value)

    @computed_field(return_type=tuple[str, ...])
    @property
    def redaction_markers(self) -> tuple[str, ...]:
        """List fields where visible redaction markers are present."""

        return _field_redaction_markers(
            {
                "workload_status": self.workload_status,
                "kubernetes_events": self.kubernetes_events,
                "container_logs": self.container_logs,
                "manifest_yaml": self.manifest_yaml,
            }
        )

    def availability(self) -> EvidenceAvailability:
        """Return a compact availability summary for graph initialization."""

        return EvidenceAvailability(
            workload_status=self.workload_status is not None,
            kubernetes_events=self.kubernetes_events is not None,
            container_logs=self.container_logs is not None,
            manifest_yaml=self.manifest_yaml is not None,
        )


class SubmittedInvestigation(StrictModel):
    """Validated natural-language investigation request and optional evidence."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    workload_name: str | None = Field(
        default=None, max_length=MAX_WORKLOAD_NAME_CHARS
    )
    namespace: str | None = Field(default=None, max_length=MAX_NAMESPACE_CHARS)
    workload_status: str | None = Field(
        default=None, max_length=MAX_WORKLOAD_STATUS_CHARS
    )
    kubernetes_events: str | None = Field(
        default=None, max_length=MAX_KUBERNETES_EVENTS_CHARS
    )
    container_logs: str | None = Field(
        default=None, max_length=MAX_CONTAINER_LOGS_CHARS
    )
    manifest_yaml: str | None = Field(
        default=None, max_length=MAX_MANIFEST_YAML_CHARS
    )
    demo_id: DemoId | None = None

    @field_validator("question", mode="before")
    @classmethod
    def sanitize_question(cls, value: object) -> object:
        """Normalize and redact the required question before storing it."""

        return _bounded_inline_text(
            value,
            maximum=MAX_QUESTION_CHARS,
            optional=False,
        )

    @field_validator("question")
    @classmethod
    def require_meaningful_question(cls, value: str) -> str:
        """Reject blank or generic questions that contain no failure symptom."""

        if not is_meaningful_question(value):
            raise ValueError(
                "Question must describe a Kubernetes deployment failure symptom."
            )
        return value

    @field_validator("workload_name", "namespace", mode="before")
    @classmethod
    def sanitize_identifiers(cls, value: object, info: Any) -> object:
        """Normalize optional identifiers and discard whitespace-only values."""

        maximum = (
            MAX_WORKLOAD_NAME_CHARS
            if info.field_name == "workload_name"
            else MAX_NAMESPACE_CHARS
        )
        return _bounded_inline_text(value, maximum=maximum, optional=True)

    @field_validator(
        "workload_status",
        "kubernetes_events",
        "container_logs",
        mode="before",
    )
    @classmethod
    def sanitize_evidence_text(cls, value: object, info: Any) -> object:
        """Normalize and redact supplied evidence before model storage."""

        limits = {
            "workload_status": MAX_WORKLOAD_STATUS_CHARS,
            "kubernetes_events": MAX_KUBERNETES_EVENTS_CHARS,
            "container_logs": MAX_CONTAINER_LOGS_CHARS,
        }
        return _bounded_multiline_text(
            value,
            maximum=limits[info.field_name],
            optional=True,
        )

    @field_validator("manifest_yaml", mode="before")
    @classmethod
    def sanitize_manifest(cls, value: object) -> object:
        """Validate YAML and redact credentials and Secret values."""

        if not isinstance(value, str):
            return value
        if len(value) > MAX_MANIFEST_YAML_CHARS:
            raise ValueError("Input exceeds the permitted maximum length.")
        if not normalize_multiline_text(value):
            return None
        return validate_and_redact_manifest(value)

    @computed_field(return_type=tuple[str, ...])
    @property
    def redaction_markers(self) -> tuple[str, ...]:
        """List each field and marker retained after sanitization."""

        return _field_redaction_markers(
            {
                "question": self.question,
                "workload_name": self.workload_name,
                "namespace": self.namespace,
                "workload_status": self.workload_status,
                "kubernetes_events": self.kubernetes_events,
                "container_logs": self.container_logs,
                "manifest_yaml": self.manifest_yaml,
            }
        )

    def submitted_evidence(self) -> SubmittedEvidence:
        """Create the sanitized evidence object used by later graph work."""

        return SubmittedEvidence(
            workload_status=self.workload_status,
            kubernetes_events=self.kubernetes_events,
            container_logs=self.container_logs,
            manifest_yaml=self.manifest_yaml,
        )

    def evidence_availability(self) -> EvidenceAvailability:
        """Report which evidence sources are currently available."""

        return self.submitted_evidence().availability()


class SymptomAssessment(StrictModel):
    """Structured interpretation of the user-reported deployment symptom."""

    category: SymptomCategory
    summary: str = Field(min_length=1, max_length=1_000)
    observable_signals: list[str] = Field(default_factory=list, max_length=12)
    relevant_sources: list[EvidenceSourceName] = Field(
        default_factory=list, max_length=4
    )
    missing_sources: list[EvidenceSourceName] = Field(
        default_factory=list, max_length=4
    )

    @field_validator("summary", mode="before")
    @classmethod
    def normalize_summary(cls, value: object) -> object:
        return _bounded_inline_text(value, maximum=1_000, optional=False)

    @field_validator("observable_signals", mode="before")
    @classmethod
    def normalize_signals(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [
            _bounded_inline_text(item, maximum=300, optional=False)
            for item in value
        ]

    @field_validator("observable_signals")
    @classmethod
    def unique_signals(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("Observable signals cannot be blank.")
        if len(value) != len(set(value)):
            raise ValueError("Observable signals must be unique.")
        return value

    @field_validator("relevant_sources", "missing_sources")
    @classmethod
    def unique_sources(
        cls, value: list[EvidenceSourceName]
    ) -> list[EvidenceSourceName]:
        if len(value) != len(set(value)):
            raise ValueError("Evidence sources must be unique.")
        return value


class ClarificationRequest(StrictModel):
    """A safe, bounded request for evidence needed to continue."""

    question: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=1_000)
    requested_sources: list[EvidenceSourceName] = Field(
        min_length=1, max_length=4
    )

    @field_validator("question", "reason", mode="before")
    @classmethod
    def normalize_text(cls, value: object, info: Any) -> object:
        maximum = 500 if info.field_name == "question" else 1_000
        return _bounded_inline_text(value, maximum=maximum, optional=False)

    @field_validator("requested_sources")
    @classmethod
    def unique_requested_sources(
        cls, value: list[EvidenceSourceName]
    ) -> list[EvidenceSourceName]:
        if len(value) != len(set(value)):
            raise ValueError("Requested evidence sources must be unique.")
        return value


class SubmissionValidationError(ValueError):
    """Safe validation failure suitable for display at an input boundary."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = tuple(dict.fromkeys(issues))
        super().__init__(" ".join(self.issues))


def _safe_submission_issue(error: Mapping[str, Any]) -> str:
    """Map Pydantic details to a message that never includes submitted text."""

    location = tuple(str(part) for part in error.get("loc", ()))
    field_name = location[0] if location else "submission"
    label = field_name.replace("_", " ").capitalize()
    error_type = str(error.get("type", ""))

    if error_type == "extra_forbidden":
        return "Submission contains an unexpected field."
    if error_type in {"string_too_long", "too_long"} or "maximum length" in str(
        error.get("msg", "")
    ).casefold():
        return f"{label} exceeds the permitted maximum length."
    if field_name == "question":
        return "Question must describe a Kubernetes deployment failure symptom."
    if field_name == "manifest_yaml":
        return "Manifest YAML could not be validated."
    if field_name == "demo_id":
        return "Demo ID is not approved."
    return f"{label} is invalid."


def validate_submitted_investigation(
    payload: Mapping[str, Any] | object,
) -> SubmittedInvestigation:
    """Validate a submission and expose only stable, non-sensitive errors."""

    try:
        return SubmittedInvestigation.model_validate(payload)
    except (ValidationError, ManifestInputError) as error:
        if isinstance(error, ValidationError):
            issues = [_safe_submission_issue(item) for item in error.errors()]
        else:
            issues = ["Manifest YAML could not be validated."]
        raise SubmissionValidationError(issues) from None


class EvidenceItem(StrictModel):
    """One evidence statement collected from an allowlisted diagnostic tool."""

    evidence_id: str = Field(min_length=1)
    source_tool: ToolName
    summary: str = Field(min_length=1)
    raw_reference: str = Field(min_length=1)
    decisive: bool = False


class ToolError(StrictModel):
    """A normalized diagnostic-tool failure stored in graph state."""

    tool_name: ToolName
    message: str = Field(min_length=1)
    retryable: bool
    attempt: int = Field(ge=1, le=6)


class ToolResult(StrictModel):
    """Structured result returned by every diagnostic tool."""

    ok: bool
    data: StructuredData = None
    error: str | None = None
    retryable: bool = False
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_success_and_error_fields(self) -> "ToolResult":
        """Require successful results to have data and failures to have errors."""

        if self.ok:
            if self.data is None:
                raise ValueError("Successful tool results must include data.")
            if self.error is not None:
                raise ValueError("Successful tool results cannot include an error.")
            if self.retryable:
                raise ValueError("Successful tool results cannot be retryable.")
        elif not self.error:
            raise ValueError("Failed tool results must include an error message.")
        return self


class ActionSelection(StrictModel):
    """The model's validated choice of one next action."""

    action: ActionName
    reason: str = Field(min_length=1)
    tool_input: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_terminal_action_input(self) -> "ActionSelection":
        """Terminal actions cannot carry tool arguments."""

        if self.action in {
            "diagnose",
            "request_clarification",
            "handoff",
        } and self.tool_input:
            raise ValueError("Terminal actions cannot include tool input.")
        return self


class DiagnosisResult(StrictModel):
    """Evidence-grounded diagnosis shown to the engineer."""

    root_cause: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    human_review_required: Literal[True] = True

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        """Prevent a diagnosis from inflating support with duplicate evidence."""

        if len(value) != len(set(value)):
            raise ValueError("Diagnosis evidence IDs must be unique.")
        if any(not evidence_id.strip() for evidence_id in value):
            raise ValueError("Diagnosis evidence IDs cannot be blank.")
        return value


class HandoffPayload(StrictModel):
    """Structured context supplied when the graph pauses for engineer review."""

    demo_id: DemoId | None = None
    question: str | None = Field(default=None, max_length=MAX_QUESTION_CHARS)
    workload_name: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    initial_symptom: str = Field(min_length=1)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    tools_attempted: list[ToolName] = Field(default_factory=list)
    tool_errors: list[ToolError] = Field(default_factory=list)
    leading_hypotheses: list[str] = Field(default_factory=list)
    handoff_reason: str = Field(min_length=1)
    unresolved_question: str = Field(min_length=1)
    recommended_manual_check: str = Field(min_length=1)
    public_decisions: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("tools_attempted")
    @classmethod
    def tools_attempted_must_be_unique(
        cls, value: list[ToolName]
    ) -> list[ToolName]:
        """Keep the handoff trace concise and chronologically unique."""

        if len(value) != len(set(value)):
            raise ValueError("Handoff tools attempted must be unique.")
        return value


__all__ = [
    "ActionName",
    "ActionSelection",
    "CaseId",
    "CaseMetadata",
    "ClarificationRequest",
    "DemoId",
    "DiagnosisResult",
    "EvidenceAvailability",
    "EvidenceItem",
    "EvidenceSourceName",
    "HandoffPayload",
    "InvestigationStatus",
    "StructuredData",
    "SubmissionValidationError",
    "SubmittedEvidence",
    "SubmittedInvestigation",
    "SymptomAssessment",
    "SymptomCategory",
    "ToolError",
    "ToolName",
    "ToolResult",
    "validate_submitted_investigation",
]
