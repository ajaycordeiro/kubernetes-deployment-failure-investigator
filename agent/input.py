"""Validation helpers for bounded, sanitized user-submitted evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, MutableSequence
from typing import Any, Final

import yaml


MAX_QUESTION_CHARS: Final[int] = 2_000
MAX_WORKLOAD_NAME_CHARS: Final[int] = 253
MAX_NAMESPACE_CHARS: Final[int] = 63
MAX_WORKLOAD_STATUS_CHARS: Final[int] = 12_000
MAX_KUBERNETES_EVENTS_CHARS: Final[int] = 20_000
MAX_CONTAINER_LOGS_CHARS: Final[int] = 20_000
MAX_MANIFEST_YAML_CHARS: Final[int] = 24_000
MAX_MANIFEST_DOCUMENTS: Final[int] = 4
MAX_MANIFEST_NODES: Final[int] = 2_000

REDACTED_AUTHORIZATION: Final[str] = "__REDACTED_AUTHORIZATION__"
REDACTED_BEARER_TOKEN: Final[str] = "__REDACTED_BEARER_TOKEN__"
REDACTED_PASSWORD: Final[str] = "__REDACTED_PASSWORD__"
REDACTED_API_TOKEN: Final[str] = "__REDACTED_API_TOKEN__"
REDACTED_KUBERNETES_SECRET: Final[str] = (
    "__REDACTED_KUBERNETES_SECRET_VALUE__"
)

_REDACTION_PATTERN = re.compile(r"__REDACTED_[A-Z_]+__")
_AUTHORIZATION_LINE_PATTERN = re.compile(
    r"(?im)^(?P<prefix>\s*authorization\s*[:=]\s*).+$"
)
_JSON_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)(?P<prefix>[\"']authorization[\"']\s*:\s*)"
    r"(?P<quote>[\"']).*?(?P=quote)"
)
_INLINE_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)(?P<prefix>\bauthorization\s*[:=]\s*)(?:bearer\s+)?\S+"
)
_BEARER_PATTERN = re.compile(
    r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}"
)
_JSON_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?P<prefix>[\"'](?P<key>(?:[a-z0-9]+[_-])*(?:api[_-]?key|"
    r"api[_-]?token|access[_-]?token|token|password|passwd|pwd|"
    r"client[_-]?secret|secret[_-]?key))[\"']"
    r"\s*:\s*)(?P<quote>[\"'])(?P<value>.*?)(?P=quote)"
)
_LINE_CREDENTIAL_PATTERN = re.compile(
    r"(?im)^(?P<prefix>\s*(?P<key>(?:[a-z0-9]+[_-])*(?:api[_-]?key|"
    r"api[_-]?token|access[_-]?token|token|password|passwd|pwd|"
    r"client[_-]?secret|secret[_-]?key))\s*[:=]\s*)"
    r"(?P<value>[^#\r\n]+)"
)
_INLINE_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?P<prefix>\b(?P<key>(?:[a-z0-9]+[_-])*(?:api[_-]?key|"
    r"api[_-]?token|access[_-]?token|token|password|passwd|pwd|"
    r"client[_-]?secret|secret[_-]?key))\s*=\s*)"
    r"(?P<value>[^\s,;\]}]+)"
)
_COMMAND_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?P<prefix>--(?:api-key|api-token|access-token|token|password|"
    r"client-secret)\s+)(?P<value>\S+)"
)
_PREFIXED_TOKEN_PATTERN = re.compile(
    r"(?i)\b(?:sk|nvapi|ghp|glpat)-[a-z0-9_-]{12,}\b"
)
_AWS_ACCESS_KEY_PATTERN = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_JWT_PATTERN = re.compile(
    r"\beyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\."
    r"[a-zA-Z0-9_-]{8,}\b"
)
_WORD_PATTERN = re.compile(r"[a-z][a-z0-9_-]+", re.IGNORECASE)
_KNOWN_FAILURE_PATTERN = re.compile(
    r"(?i)(imagepullbackoff|errimagepull|crashloopbackoff|"
    r"createcontainerconfigerror|failedscheduling|failedmount)"
)
_SYMPTOM_WORDS: Final[frozenset[str]] = frozenset(
    {
        "backoff",
        "cannot",
        "crash",
        "crashing",
        "denied",
        "error",
        "fail",
        "failed",
        "failing",
        "failure",
        "missing",
        "mount",
        "pending",
        "progressing",
        "probe",
        "pull",
        "ready",
        "restart",
        "restarting",
        "schedule",
        "start",
        "starting",
        "stuck",
        "timeout",
        "unauthorized",
        "unavailable",
        "unhealthy",
    }
)
_GENERIC_QUESTIONS: Final[frozenset[str]] = frozenset(
    {
        "can you help me",
        "help me with this",
        "please help me",
        "something is wrong",
        "what is the problem",
    }
)
_SENSITIVE_ENV_NAME_PATTERN = re.compile(
    r"(?i)(x[_-]?api[_-]?key|api[_-]?key|api[_-]?token|access[_-]?token|token|password|passwd|"
    r"pwd|client[_-]?secret|secret[_-]?key)"
)


class ManifestInputError(ValueError):
    """Raised with a safe message when submitted manifest YAML is invalid."""


def normalize_inline_text(value: str) -> str:
    """Collapse user-facing inline text to stable single spacing."""

    return " ".join(value.split())


def normalize_multiline_text(value: str) -> str:
    """Normalize line endings and trailing whitespace without changing indentation."""

    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    compacted: list[str] = []
    blank_count = 0
    for line in lines:
        if line:
            blank_count = 0
            compacted.append(line)
        else:
            blank_count += 1
            if blank_count <= 2:
                compacted.append(line)
    return "\n".join(compacted)


def _credential_marker(key: str) -> str:
    normalized = key.casefold().replace("-", "_")
    if normalized.endswith(("password", "passwd", "pwd")):
        return REDACTED_PASSWORD
    return REDACTED_API_TOKEN


def redact_sensitive_text(value: str) -> str:
    """Replace obvious credential material with explicit stable markers."""

    redacted = _AUTHORIZATION_LINE_PATTERN.sub(
        lambda match: f"{match.group('prefix')}{REDACTED_AUTHORIZATION}",
        value,
    )
    redacted = _JSON_AUTHORIZATION_PATTERN.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}"
            f"{REDACTED_AUTHORIZATION}{match.group('quote')}"
        ),
        redacted,
    )
    redacted = _INLINE_AUTHORIZATION_PATTERN.sub(
        lambda match: f"{match.group('prefix')}{REDACTED_AUTHORIZATION}",
        redacted,
    )
    redacted = _JSON_CREDENTIAL_PATTERN.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}"
            f"{_credential_marker(match.group('key'))}{match.group('quote')}"
        ),
        redacted,
    )
    redacted = _LINE_CREDENTIAL_PATTERN.sub(
        lambda match: (
            f"{match.group('prefix')}{_credential_marker(match.group('key'))}"
        ),
        redacted,
    )
    redacted = _INLINE_CREDENTIAL_PATTERN.sub(
        lambda match: (
            f"{match.group('prefix')}{_credential_marker(match.group('key'))}"
        ),
        redacted,
    )
    redacted = _COMMAND_CREDENTIAL_PATTERN.sub(
        lambda match: f"{match.group('prefix')}{REDACTED_API_TOKEN}",
        redacted,
    )
    redacted = _BEARER_PATTERN.sub(
        f"Bearer {REDACTED_BEARER_TOKEN}", redacted
    )
    redacted = _PREFIXED_TOKEN_PATTERN.sub(REDACTED_API_TOKEN, redacted)
    redacted = _AWS_ACCESS_KEY_PATTERN.sub(REDACTED_API_TOKEN, redacted)
    return _JWT_PATTERN.sub(REDACTED_API_TOKEN, redacted)


def redaction_markers(value: str | None) -> tuple[str, ...]:
    """Return the distinct redaction markers present in sanitized text."""

    if not value:
        return ()
    return tuple(dict.fromkeys(_REDACTION_PATTERN.findall(value)))


def is_meaningful_question(value: str) -> bool:
    """Require enough symptom information to begin a bounded investigation."""

    normalized = normalize_inline_text(value).casefold()
    if normalized in _GENERIC_QUESTIONS:
        return False
    words = _WORD_PATTERN.findall(normalized)
    if len(words) < 3:
        return False
    return bool(_KNOWN_FAILURE_PATTERN.search(normalized)) or bool(
        set(words) & _SYMPTOM_WORDS
    )


def _count_yaml_nodes(value: object, seen: set[int] | None = None) -> int:
    visited = seen if seen is not None else set()
    if isinstance(value, (Mapping, list, tuple)):
        identity = id(value)
        if identity in visited:
            return 0
        visited.add(identity)
    if isinstance(value, Mapping):
        return 1 + sum(
            _count_yaml_nodes(key, visited) + _count_yaml_nodes(item, visited)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return 1 + sum(_count_yaml_nodes(item, visited) for item in value)
    return 1


def _redact_manifest_node(
    value: object,
    *,
    secret_document: bool,
    seen: set[int] | None = None,
) -> bool:
    visited = seen if seen is not None else set()
    if isinstance(value, (MutableMapping, MutableSequence)):
        identity = id(value)
        if identity in visited:
            return False
        visited.add(identity)

    changed = False
    if isinstance(value, MutableMapping):
        current_secret_document = secret_document or (
            str(value.get("kind", "")).casefold() == "secret"
        )
        if current_secret_document:
            for field in ("data", "stringData", "binaryData"):
                secret_values = value.get(field)
                if isinstance(secret_values, MutableMapping):
                    for key in tuple(secret_values):
                        secret_values[key] = REDACTED_KUBERNETES_SECRET
                        changed = True

        env_name = value.get("name")
        if (
            isinstance(env_name, str)
            and _SENSITIVE_ENV_NAME_PATTERN.search(env_name)
            and "value" in value
        ):
            value["value"] = REDACTED_KUBERNETES_SECRET
            changed = True

        for item in value.values():
            changed = (
                _redact_manifest_node(
                    item,
                    secret_document=current_secret_document,
                    seen=visited,
                )
                or changed
            )
    elif isinstance(value, MutableSequence):
        for item in value:
            changed = (
                _redact_manifest_node(
                    item,
                    secret_document=secret_document,
                    seen=visited,
                )
                or changed
            )
    return changed


def validate_and_redact_manifest(value: str) -> str:
    """Safely parse YAML and remove literal Kubernetes Secret values."""

    normalized = normalize_multiline_text(value)
    try:
        documents = list(yaml.safe_load_all(normalized))
    except (yaml.YAMLError, RecursionError) as error:
        raise ManifestInputError(
            "Manifest YAML could not be parsed safely."
        ) from error

    if not documents or len(documents) > MAX_MANIFEST_DOCUMENTS:
        raise ManifestInputError(
            "Manifest YAML must contain between one and four documents."
        )
    if any(not isinstance(document, MutableMapping) for document in documents):
        raise ManifestInputError(
            "Each manifest YAML document must be a mapping."
        )
    if sum(_count_yaml_nodes(document) for document in documents) > MAX_MANIFEST_NODES:
        raise ManifestInputError("Manifest YAML is too structurally complex.")

    changed = False
    for document in documents:
        changed = (
            _redact_manifest_node(
                document,
                secret_document=False,
            )
            or changed
        )
    if not changed:
        return redact_sensitive_text(normalized)
    dumped = normalize_multiline_text(
        yaml.safe_dump_all(documents, sort_keys=False, explicit_start=False)
    )
    return redact_sensitive_text(dumped)


__all__ = [
    "MAX_CONTAINER_LOGS_CHARS",
    "MAX_KUBERNETES_EVENTS_CHARS",
    "MAX_MANIFEST_DOCUMENTS",
    "MAX_MANIFEST_NODES",
    "MAX_MANIFEST_YAML_CHARS",
    "MAX_NAMESPACE_CHARS",
    "MAX_QUESTION_CHARS",
    "MAX_WORKLOAD_NAME_CHARS",
    "MAX_WORKLOAD_STATUS_CHARS",
    "ManifestInputError",
    "REDACTED_API_TOKEN",
    "REDACTED_AUTHORIZATION",
    "REDACTED_BEARER_TOKEN",
    "REDACTED_KUBERNETES_SECRET",
    "REDACTED_PASSWORD",
    "is_meaningful_question",
    "normalize_inline_text",
    "normalize_multiline_text",
    "redact_sensitive_text",
    "redaction_markers",
    "validate_and_redact_manifest",
]
