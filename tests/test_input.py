"""Focused tests for bounded and sanitized user-submitted evidence."""

from __future__ import annotations

import unittest

import yaml
from pydantic import ValidationError

from agent.input import (
    MAX_CONTAINER_LOGS_CHARS,
    MAX_KUBERNETES_EVENTS_CHARS,
    MAX_MANIFEST_YAML_CHARS,
    MAX_NAMESPACE_CHARS,
    MAX_QUESTION_CHARS,
    MAX_WORKLOAD_NAME_CHARS,
    MAX_WORKLOAD_STATUS_CHARS,
    REDACTED_API_TOKEN,
    REDACTED_AUTHORIZATION,
    REDACTED_KUBERNETES_SECRET,
    REDACTED_PASSWORD,
)
from agent.schemas import (
    ClarificationRequest,
    EvidenceAvailability,
    SubmissionValidationError,
    SubmittedEvidence,
    SubmittedInvestigation,
    SymptomAssessment,
    validate_submitted_investigation,
)


VALID_QUESTION = "Why is the payment-api deployment failing to become ready?"


class SubmittedInvestigationTests(unittest.TestCase):
    """Validate the public submission boundary without touching graph behavior."""

    def test_accepts_complete_input_and_normalizes_whitespace(self) -> None:
        submission = validate_submitted_investigation(
            {
                "question": "  Why   is the payment-api deployment failing?  ",
                "workload_name": "  payment-api  ",
                "namespace": "  production  ",
                "workload_status": "\r\nAvailable: 0  \r\nReady: 0\r\n",
                "kubernetes_events": "  Failed to pull image  \n",
                "container_logs": "  startup failed  \n",
                "manifest_yaml": (
                    "apiVersion: apps/v1\n"
                    "kind: Deployment\n"
                    "metadata:\n"
                    "  name: payment-api\n"
                ),
                "demo_id": "case_001",
            }
        )

        self.assertEqual(
            submission.question,
            "Why is the payment-api deployment failing?",
        )
        self.assertEqual(submission.workload_name, "payment-api")
        self.assertEqual(submission.namespace, "production")
        self.assertEqual(submission.workload_status, "Available: 0\nReady: 0")
        self.assertEqual(submission.demo_id, "case_001")
        self.assertEqual(yaml.safe_load(submission.manifest_yaml)["kind"], "Deployment")
        self.assertEqual(
            submission.evidence_availability().model_dump(),
            {
                "workload_status": True,
                "kubernetes_events": True,
                "container_logs": True,
                "manifest_yaml": True,
            },
        )

    def test_accepts_question_only_input(self) -> None:
        submission = validate_submitted_investigation(
            {"question": "Why does this Kubernetes workload keep crashing?"}
        )

        self.assertIsInstance(submission, SubmittedInvestigation)
        self.assertEqual(
            submission.evidence_availability().model_dump(),
            {
                "workload_status": False,
                "kubernetes_events": False,
                "container_logs": False,
                "manifest_yaml": False,
            },
        )
        self.assertEqual(submission.redaction_markers, ())

    def test_rejects_every_oversized_text_field(self) -> None:
        limits = {
            "question": MAX_QUESTION_CHARS,
            "workload_name": MAX_WORKLOAD_NAME_CHARS,
            "namespace": MAX_NAMESPACE_CHARS,
            "workload_status": MAX_WORKLOAD_STATUS_CHARS,
            "kubernetes_events": MAX_KUBERNETES_EVENTS_CHARS,
            "container_logs": MAX_CONTAINER_LOGS_CHARS,
            "manifest_yaml": MAX_MANIFEST_YAML_CHARS,
        }
        for field_name, maximum in limits.items():
            with self.subTest(field_name=field_name):
                payload = {"question": VALID_QUESTION, field_name: "x" * (maximum + 1)}
                with self.assertRaises(SubmissionValidationError) as context:
                    validate_submitted_investigation(payload)
                self.assertIn("maximum length", str(context.exception))

    def test_applies_length_limit_before_redaction(self) -> None:
        oversized_secret = "api_key=" + ("s" * MAX_CONTAINER_LOGS_CHARS)
        with self.assertRaises(SubmissionValidationError):
            validate_submitted_investigation(
                {"question": VALID_QUESTION, "container_logs": oversized_secret}
            )

    def test_rejects_malformed_yaml_with_safe_message(self) -> None:
        secret = "highly-sensitive-manifest-value"
        with self.assertRaises(SubmissionValidationError) as context:
            validate_submitted_investigation(
                {
                    "question": VALID_QUESTION,
                    "manifest_yaml": (
                        "apiVersion: v1\nkind: Deployment\nmetadata: [\n"
                        f"password: {secret}"
                    ),
                }
            )

        self.assertEqual(
            str(context.exception), "Manifest YAML could not be validated."
        )
        self.assertNotIn(secret, str(context.exception))

    def test_rejects_unexpected_fields_and_unapproved_demo_ids(self) -> None:
        with self.assertRaises(SubmissionValidationError) as extra_context:
            validate_submitted_investigation(
                {"question": VALID_QUESTION, "case_id": "case_001"}
            )
        self.assertEqual(
            str(extra_context.exception),
            "Submission contains an unexpected field.",
        )

        with self.assertRaises(SubmissionValidationError) as demo_context:
            validate_submitted_investigation(
                {"question": VALID_QUESTION, "demo_id": "case_999"}
            )
        self.assertEqual(str(demo_context.exception), "Demo ID is not approved.")

    def test_redacts_credentials_before_the_model_can_be_serialized(self) -> None:
        secrets = {
            "bearer": "question-bearer-token-12345",
            "authorization": "Basic-YWxhZGRpbjpvcGVuc2VzYW1l",
            "client": "events-client-secret",
            "password": "logs-password-value",
            "api_key": "sk-1234567890abcdef",
            "nebius_api_key": "nebius-api-secret-value",
        }
        submission = validate_submitted_investigation(
            {
                "question": (
                    "Why is the deployment failing with Bearer "
                    f"{secrets['bearer']}?"
                ),
                "workload_status": (
                    "Authorization: " + secrets["authorization"]
                ),
                "kubernetes_events": (
                    "Authentication failed; client_secret=" + secrets["client"]
                ),
                "container_logs": (
                    "password="
                    + secrets["password"]
                    + "\napi_key="
                    + secrets["api_key"]
                    + "\nNEBIUS_API_KEY="
                    + secrets["nebius_api_key"]
                ),
            }
        )

        serialized = submission.model_dump_json()
        for secret in secrets.values():
            self.assertNotIn(secret, serialized)
        self.assertIn(REDACTED_AUTHORIZATION, serialized)
        self.assertIn(REDACTED_PASSWORD, serialized)
        self.assertIn(REDACTED_API_TOKEN, serialized)
        self.assertTrue(submission.redaction_markers)

    def test_redacts_kubernetes_secret_values_and_literal_secret_env_values(
        self,
    ) -> None:
        manifest = """
apiVersion: v1
kind: Secret
metadata:
  name: app-credentials
data:
  password: c3VwZXItc2VjcmV0
stringData:
  API_TOKEN: plain-text-token
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payment-api
spec:
  template:
    spec:
      containers:
        - name: app
          image: example/app:1
          env:
            - name: DATABASE_PASSWORD
              value: literal-database-password
            - name: API_TOKEN
              valueFrom:
                secretKeyRef:
                  name: app-credentials
                  key: API_TOKEN
"""
        submission = validate_submitted_investigation(
            {"question": VALID_QUESTION, "manifest_yaml": manifest}
        )

        serialized = submission.model_dump_json()
        for secret in (
            "c3VwZXItc2VjcmV0",
            "plain-text-token",
            "literal-database-password",
        ):
            self.assertNotIn(secret, serialized)
        self.assertIn(REDACTED_KUBERNETES_SECRET, serialized)
        self.assertIn("app-credentials", serialized)
        self.assertIn(
            f"manifest_yaml:{REDACTED_KUBERNETES_SECRET}",
            submission.redaction_markers,
        )

    def test_submitted_evidence_cannot_bypass_sanitization(self) -> None:
        secret = "direct-evidence-password"
        evidence = SubmittedEvidence(container_logs=f"password={secret}")

        self.assertNotIn(secret, evidence.model_dump_json())
        self.assertIn(REDACTED_PASSWORD, evidence.container_logs)

    def test_rejects_empty_and_meaningless_questions(self) -> None:
        for question in (
            "",
            "   \r\n  ",
            "help",
            "please help me",
            "what is the problem",
            "Tell me about Kubernetes",
        ):
            with self.subTest(question=question):
                with self.assertRaises(SubmissionValidationError) as context:
                    validate_submitted_investigation({"question": question})
                self.assertEqual(
                    str(context.exception),
                    "Question must describe a Kubernetes deployment failure symptom.",
                )

    def test_supporting_contracts_are_strict_and_bounded(self) -> None:
        availability = EvidenceAvailability(workload_status=True)
        self.assertTrue(availability.workload_status)
        self.assertFalse(availability.container_logs)

        assessment = SymptomAssessment(
            category="scheduling",
            summary="  Pods   remain Pending after rollout. ",
            observable_signals=["  FailedScheduling event  "],
            relevant_sources=["kubernetes_events"],
            missing_sources=["manifest_yaml"],
        )
        self.assertEqual(assessment.summary, "Pods remain Pending after rollout.")
        self.assertEqual(
            assessment.observable_signals, ["FailedScheduling event"]
        )

        clarification = ClarificationRequest(
            question="  Can you provide the pod events? ",
            reason="  Scheduling evidence is missing. ",
            requested_sources=["kubernetes_events"],
        )
        self.assertEqual(
            clarification.question, "Can you provide the pod events?"
        )
        with self.assertRaises(ValidationError):
            ClarificationRequest(
                question="q" * 501,
                reason="Evidence is missing.",
                requested_sources=["kubernetes_events"],
            )

    def test_supporting_contracts_reject_unexpected_fields(self) -> None:
        with self.assertRaises(ValidationError):
            EvidenceAvailability(workload_status=True, raw_secret="not-allowed")


if __name__ == "__main__":
    unittest.main()
