"""Tests for validated contracts and the safe fixture loader."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from agent.schemas import (
    ActionSelection,
    DiagnosisResult,
    EvidenceItem,
    HandoffPayload,
    ToolError,
    ToolResult,
)
from tools.base import (
    ALLOWED_CASE_IDS,
    ALLOWED_FIXTURE_NAMES,
    FixtureAccessError,
    FixtureErrorCode,
    FixtureLoader,
    InvalidCaseIdError,
    InvalidFixtureNameError,
)


class FixtureLoaderTests(TestCase):
    """Exercise valid loading and all required path boundaries."""

    def setUp(self) -> None:
        self.loader = FixtureLoader()

    def test_lists_exactly_ten_approved_cases(self) -> None:
        self.assertEqual(
            self.loader.list_case_ids(),
            tuple(f"case_{number:03d}" for number in range(1, 11)),
        )
        self.assertEqual(len(ALLOWED_CASE_IDS), 10)

    def test_loads_and_validates_every_case_metadata_file(self) -> None:
        for case_id in self.loader.list_case_ids():
            with self.subTest(case_id=case_id):
                metadata = self.loader.load_case_metadata(case_id)
                self.assertEqual(metadata.case_id, case_id)
                self.assertTrue(metadata.workload_name)
                self.assertTrue(metadata.namespace)
                self.assertTrue(metadata.container_name)
                self.assertTrue(metadata.initial_symptom)

    def test_loads_every_allowlisted_fixture_for_every_case(self) -> None:
        for case_id in self.loader.list_case_ids():
            for fixture_name in sorted(ALLOWED_FIXTURE_NAMES):
                with self.subTest(
                    case_id=case_id, fixture_name=fixture_name
                ):
                    fixture = self.loader.load_fixture(case_id, fixture_name)
                    if fixture_name == "logs.txt":
                        self.assertIsInstance(fixture, str)
                    else:
                        self.assertIsInstance(fixture, dict)

    def test_loads_json_fixture_as_mapping(self) -> None:
        status = self.loader.load_fixture("case_001", "workload_status.json")
        self.assertIsInstance(status, dict)
        self.assertEqual(status["deployment"]["name"], "payment-api")

    def test_loads_text_fixture_as_text(self) -> None:
        logs = self.loader.load_fixture("case_003", "logs.txt")
        self.assertIsInstance(logs, str)
        self.assertIn("ModuleNotFoundError", logs)

    def test_loads_yaml_fixture_as_mapping(self) -> None:
        manifest = self.loader.load_fixture("case_006", "manifest.yaml")
        self.assertIsInstance(manifest, dict)
        self.assertEqual(manifest["kind"], "Deployment")
        self.assertEqual(manifest["metadata"]["name"], "shipping-api")

    def test_rejects_unknown_and_malformed_case_ids(self) -> None:
        invalid_case_ids: tuple[object, ...] = (
            "case_000",
            "case_011",
            "CASE_001",
            "../case_001",
            "case_001/../../tests",
            "",
            None,
            1,
        )
        for case_id in invalid_case_ids:
            with self.subTest(case_id=case_id):
                with self.assertRaises(InvalidCaseIdError) as context:
                    self.loader.load_fixture(case_id, "metadata.json")
                self.assertEqual(
                    context.exception.code, FixtureErrorCode.INVALID_CASE_ID
                )

    def test_rejects_path_traversal_and_absolute_fixture_paths(self) -> None:
        unsafe_names: tuple[object, ...] = (
            "../metadata.json",
            "..\\metadata.json",
            "case_001/metadata.json",
            "/tmp/metadata.json",
            "C:\\temp\\metadata.json",
        )
        for fixture_name in unsafe_names:
            with self.subTest(fixture_name=fixture_name):
                with self.assertRaises(FixtureAccessError) as context:
                    self.loader.load_fixture("case_001", fixture_name)
                self.assertEqual(
                    context.exception.code, FixtureErrorCode.ACCESS_DENIED
                )

    def test_rejects_unknown_simple_fixture_names(self) -> None:
        unknown_names: tuple[object, ...] = (
            "answer.json",
            "expected_result.json",
            "deployment.yaml",
            "",
            None,
        )
        for fixture_name in unknown_names:
            with self.subTest(fixture_name=fixture_name):
                with self.assertRaises(InvalidFixtureNameError) as context:
                    self.loader.load_fixture("case_001", fixture_name)
                self.assertEqual(
                    context.exception.code,
                    FixtureErrorCode.INVALID_FIXTURE_NAME,
                )

    def test_blocks_ground_truth_access_before_reading(self) -> None:
        with patch("pathlib.Path.read_text") as read_text:
            with self.assertRaises(FixtureAccessError):
                self.loader.load_fixture(
                    "case_001", "../../tests/ground_truth.json"
                )
            with self.assertRaises(InvalidFixtureNameError):
                self.loader.load_fixture("case_001", "ground_truth.json")
            read_text.assert_not_called()
        self.assertNotIn("ground_truth.json", ALLOWED_FIXTURE_NAMES)


class SchemaTests(TestCase):
    """Confirm the requested Pydantic contracts enforce key invariants."""

    def test_tool_result_requires_error_for_failure(self) -> None:
        with self.assertRaises(ValueError):
            ToolResult(ok=False, source="events.json")

    def test_tool_result_rejects_error_on_success(self) -> None:
        with self.assertRaises(ValueError):
            ToolResult(
                ok=True,
                data={"items": []},
                error="unexpected",
                source="events.json",
            )

    def test_terminal_action_rejects_tool_input(self) -> None:
        with self.assertRaises(ValueError):
            ActionSelection(
                action="diagnose",
                reason="Evidence is sufficient.",
                tool_input={"query": "unexpected"},
            )

    def test_diagnosis_rejects_duplicate_evidence_ids(self) -> None:
        with self.assertRaises(ValueError):
            DiagnosisResult(
                root_cause="Invalid image tag",
                confidence=0.92,
                evidence_ids=["event-1", "event-1"],
                recommended_action="Verify the reviewed image tag.",
            )

    def test_handoff_payload_accepts_nested_contracts(self) -> None:
        evidence = EvidenceItem(
            evidence_id="status-1",
            source_tool="inspect_workload_status",
            summary="Container is in CrashLoopBackOff.",
            raw_reference="workload_status.json",
        )
        error = ToolError(
            tool_name="inspect_container_logs",
            message="Logs are incomplete.",
            retryable=False,
            attempt=1,
        )
        payload = HandoffPayload(
            demo_id="case_010",
            workload_name="checkout-api",
            namespace="checkout",
            initial_symptom="Container repeatedly restarts.",
            evidence=[evidence],
            tools_attempted=[
                "inspect_workload_status",
                "inspect_container_logs",
            ],
            tool_errors=[error],
            leading_hypotheses=["Missing or invalid provider configuration"],
            handoff_reason="Available evidence is insufficient.",
            unresolved_question="Which provider configuration is unavailable?",
            recommended_manual_check="Inspect the complete startup logs.",
        )
        self.assertEqual(payload.demo_id, "case_010")
        self.assertEqual(payload.evidence[0].evidence_id, "status-1")
