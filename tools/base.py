"""Safe, read-only access to synthetic deployment-failure fixtures."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

from agent.schemas import CaseId, CaseMetadata, ToolResult

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only before setup
    yaml = None


JsonFixtureName: TypeAlias = Literal[
    "metadata.json",
    "workload_status.json",
    "events.json",
]
TextFixtureName: TypeAlias = Literal["logs.txt"]
YamlFixtureName: TypeAlias = Literal["manifest.yaml"]
FixtureName: TypeAlias = JsonFixtureName | TextFixtureName | YamlFixtureName
FixtureData: TypeAlias = dict[str, Any] | str

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
CASES_ROOT: Final[Path] = (PROJECT_ROOT / "data" / "cases").resolve()
RUNBOOKS_ROOT: Final[Path] = (PROJECT_ROOT / "data" / "runbooks").resolve()
RUNBOOK_FILENAME: Final[str] = "kubernetes_failures.md"

ALLOWED_CASE_IDS: Final[frozenset[str]] = frozenset(
    f"case_{number:03d}" for number in range(1, 11)
)
ALLOWED_FIXTURE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "metadata.json",
        "workload_status.json",
        "events.json",
        "logs.txt",
        "manifest.yaml",
    }
)


class FixtureErrorCode(StrEnum):
    """Stable codes for loader failures."""

    INVALID_CASE_ID = "invalid_case_id"
    INVALID_FIXTURE_NAME = "invalid_fixture_name"
    ACCESS_DENIED = "access_denied"
    NOT_FOUND = "not_found"
    READ_ERROR = "read_error"
    DECODE_ERROR = "decode_error"
    PARSE_ERROR = "parse_error"
    INVALID_DOCUMENT = "invalid_document"
    DEPENDENCY_MISSING = "dependency_missing"


class FixtureLoadError(Exception):
    """Base class for all expected fixture-loading failures."""

    code: FixtureErrorCode

    def __init__(
        self,
        message: str,
        *,
        case_id: object | None = None,
        fixture_name: object | None = None,
    ) -> None:
        super().__init__(message)
        self.case_id = case_id
        self.fixture_name = fixture_name


class InvalidCaseIdError(FixtureLoadError):
    """Raised when a case ID is not one of the ten approved IDs."""

    code = FixtureErrorCode.INVALID_CASE_ID


class InvalidFixtureNameError(FixtureLoadError):
    """Raised when a simple filename is not in the fixture allowlist."""

    code = FixtureErrorCode.INVALID_FIXTURE_NAME


class FixtureAccessError(FixtureLoadError):
    """Raised when input attempts to address a path outside a case folder."""

    code = FixtureErrorCode.ACCESS_DENIED


class FixtureNotFoundError(FixtureLoadError):
    """Raised when an allowlisted fixture is absent."""

    code = FixtureErrorCode.NOT_FOUND


class FixtureReadError(FixtureLoadError):
    """Raised when an allowlisted fixture cannot be read."""

    code = FixtureErrorCode.READ_ERROR


class FixtureDecodeError(FixtureLoadError):
    """Raised when an allowlisted fixture is not valid UTF-8 text."""

    code = FixtureErrorCode.DECODE_ERROR


class FixtureParseError(FixtureLoadError):
    """Raised when JSON or YAML syntax cannot be parsed."""

    code = FixtureErrorCode.PARSE_ERROR


class InvalidFixtureDocumentError(FixtureLoadError):
    """Raised when a parsed fixture is not a mapping document."""

    code = FixtureErrorCode.INVALID_DOCUMENT


class FixtureDependencyError(FixtureLoadError):
    """Raised when an optional parser dependency is unavailable."""

    code = FixtureErrorCode.DEPENDENCY_MISSING


class FixtureLoader:
    """Load only approved synthetic fixture files from approved case folders."""

    def list_case_ids(self) -> tuple[str, ...]:
        """Return the ten supported IDs in stable display order."""

        return tuple(sorted(ALLOWED_CASE_IDS))

    def load_fixture(
        self,
        case_id: object,
        fixture_name: object,
    ) -> FixtureData:
        """Load an allowlisted JSON, text, or YAML fixture."""

        validated_case_id = self._validate_case_id(case_id)
        validated_fixture_name = self._validate_fixture_name(fixture_name)
        fixture_path = self._resolve_fixture_path(
            validated_case_id, validated_fixture_name
        )

        if validated_fixture_name.endswith(".json"):
            return self._load_mapping_document(
                fixture_path,
                validated_case_id,
                validated_fixture_name,
                document_type="JSON",
            )
        if validated_fixture_name.endswith(".yaml"):
            return self._load_mapping_document(
                fixture_path,
                validated_case_id,
                validated_fixture_name,
                document_type="YAML",
            )
        return self._read_text(
            fixture_path, validated_case_id, validated_fixture_name
        )

    def load_case_metadata(self, case_id: object) -> CaseMetadata:
        """Load and validate the user-visible metadata for one case."""

        validated_case_id = self._validate_case_id(case_id)
        raw_metadata = self.load_fixture(validated_case_id, "metadata.json")
        if not isinstance(raw_metadata, dict):
            raise InvalidFixtureDocumentError(
                "Case metadata must be a mapping document.",
                case_id=validated_case_id,
                fixture_name="metadata.json",
            )
        return CaseMetadata.model_validate(raw_metadata)

    def load_runbook(self) -> str:
        """Load the single approved local troubleshooting runbook."""

        runbook_path = (RUNBOOKS_ROOT / RUNBOOK_FILENAME).resolve()
        if (
            not runbook_path.is_relative_to(RUNBOOKS_ROOT)
            or runbook_path.parent != RUNBOOKS_ROOT
        ):
            raise FixtureAccessError(
                "Resolved runbook path is outside the approved directory.",
                fixture_name=RUNBOOK_FILENAME,
            )
        if not runbook_path.is_file():
            raise FixtureNotFoundError(
                "The approved troubleshooting runbook does not exist.",
                fixture_name=RUNBOOK_FILENAME,
            )
        return self._read_text(runbook_path, "runbook", RUNBOOK_FILENAME)

    def _validate_case_id(self, case_id: object) -> CaseId:
        if not isinstance(case_id, str) or case_id not in ALLOWED_CASE_IDS:
            raise InvalidCaseIdError(
                "Case ID must be one of case_001 through case_010.",
                case_id=case_id,
            )
        return cast(CaseId, case_id)

    def _validate_fixture_name(self, fixture_name: object) -> FixtureName:
        if not isinstance(fixture_name, str) or not fixture_name:
            raise InvalidFixtureNameError(
                "Fixture name must be a non-empty string from the allowlist.",
                fixture_name=fixture_name,
            )

        candidate = Path(fixture_name)
        has_directory_component = (
            candidate.is_absolute()
            or candidate.name != fixture_name
            or "/" in fixture_name
            or "\\" in fixture_name
            or ".." in candidate.parts
        )
        if has_directory_component:
            raise FixtureAccessError(
                "Fixture paths cannot contain directory components.",
                fixture_name=fixture_name,
            )

        if fixture_name not in ALLOWED_FIXTURE_NAMES:
            raise InvalidFixtureNameError(
                "Fixture name is not in the read-only allowlist.",
                fixture_name=fixture_name,
            )
        return cast(FixtureName, fixture_name)

    def _resolve_fixture_path(
        self,
        case_id: CaseId,
        fixture_name: FixtureName,
    ) -> Path:
        case_directory = (CASES_ROOT / case_id).resolve()
        fixture_path = (case_directory / fixture_name).resolve()

        if (
            not case_directory.is_relative_to(CASES_ROOT)
            or not fixture_path.is_relative_to(CASES_ROOT)
            or fixture_path.parent != case_directory
        ):
            raise FixtureAccessError(
                "Resolved fixture path is outside the approved case directory.",
                case_id=case_id,
                fixture_name=fixture_name,
            )

        if not fixture_path.is_file():
            raise FixtureNotFoundError(
                "The requested fixture does not exist for this case.",
                case_id=case_id,
                fixture_name=fixture_name,
            )
        return fixture_path

    def _read_text(
        self,
        fixture_path: Path,
        case_id: object,
        fixture_name: object,
    ) -> str:
        try:
            return fixture_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise FixtureDecodeError(
                "Fixture is not valid UTF-8 text.",
                case_id=case_id,
                fixture_name=fixture_name,
            ) from error
        except OSError as error:
            raise FixtureReadError(
                "Fixture could not be read.",
                case_id=case_id,
                fixture_name=fixture_name,
            ) from error

    def _load_mapping_document(
        self,
        fixture_path: Path,
        case_id: CaseId,
        fixture_name: FixtureName,
        *,
        document_type: Literal["JSON", "YAML"],
    ) -> dict[str, Any]:
        raw_text = self._read_text(fixture_path, case_id, fixture_name)

        if document_type == "JSON":
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as error:
                raise FixtureParseError(
                    "Fixture contains invalid JSON.",
                    case_id=case_id,
                    fixture_name=fixture_name,
                ) from error
        else:
            if yaml is None:
                raise FixtureDependencyError(
                    "PyYAML is required to load YAML fixtures.",
                    case_id=case_id,
                    fixture_name=fixture_name,
                )
            try:
                parsed = yaml.safe_load(raw_text)
            except yaml.YAMLError as error:
                raise FixtureParseError(
                    "Fixture contains invalid YAML.",
                    case_id=case_id,
                    fixture_name=fixture_name,
                ) from error

        if not isinstance(parsed, dict):
            raise InvalidFixtureDocumentError(
                f"{document_type} fixture must contain a mapping document.",
                case_id=case_id,
                fixture_name=fixture_name,
            )
        return parsed


DEFAULT_FIXTURE_LOADER: Final[FixtureLoader] = FixtureLoader()


def load_fixture(case_id: object, fixture_name: object) -> FixtureData:
    """Load one fixture through the default safe loader."""

    return DEFAULT_FIXTURE_LOADER.load_fixture(case_id, fixture_name)


def load_case_metadata(case_id: object) -> CaseMetadata:
    """Load one case's validated user-visible metadata."""

    return DEFAULT_FIXTURE_LOADER.load_case_metadata(case_id)


def load_runbook() -> str:
    """Load the approved troubleshooting runbook through the safe loader."""

    return DEFAULT_FIXTURE_LOADER.load_runbook()


def fixture_error_result(
    error: FixtureLoadError,
    *,
    source: str,
) -> ToolResult:
    """Convert an expected local-data failure to the common tool contract."""

    return ToolResult(
        ok=False,
        error=f"{error.code.value}: {error}",
        retryable=False,
        source=source,
    )


__all__ = [
    "ALLOWED_CASE_IDS",
    "ALLOWED_FIXTURE_NAMES",
    "CASES_ROOT",
    "DEFAULT_FIXTURE_LOADER",
    "FixtureAccessError",
    "FixtureData",
    "FixtureDecodeError",
    "FixtureDependencyError",
    "FixtureErrorCode",
    "FixtureLoadError",
    "FixtureLoader",
    "FixtureNotFoundError",
    "FixtureParseError",
    "FixtureReadError",
    "InvalidCaseIdError",
    "InvalidFixtureDocumentError",
    "InvalidFixtureNameError",
    "RUNBOOKS_ROOT",
    "RUNBOOK_FILENAME",
    "fixture_error_result",
    "load_case_metadata",
    "load_fixture",
    "load_runbook",
]
