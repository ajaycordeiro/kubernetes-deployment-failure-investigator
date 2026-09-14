"""Streamlit AppTest coverage for the guided natural-language interface."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import patch

from langchain_core.messages import BaseMessage
from streamlit.testing.v1 import AppTest

from agent.model import NebiusConfigurationError
from agent.schemas import ActionSelection, DiagnosisResult
from app import _apply_streamlit_provider_secrets, _trace_rows


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class _UiModel:
    def __init__(
        self,
        actions: list[object] | None = None,
        diagnosis: DiagnosisResult | None = None,
    ) -> None:
        self.actions = list(actions or [])
        self.diagnosis = diagnosis

    def select_action(self, messages: Sequence[BaseMessage]) -> Any:
        value = self.actions.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def produce_diagnosis(
        self, messages: Sequence[BaseMessage]
    ) -> DiagnosisResult:
        if self.diagnosis is None:
            raise AssertionError("This scenario must not request a diagnosis.")
        return self.diagnosis


def _action(name: str) -> ActionSelection:
    return ActionSelection(
        action=name,
        reason="Select the next public read-only source.",
        tool_input={},
    )


def _diagnosis(evidence_ids: list[str]) -> DiagnosisResult:
    return DiagnosisResult(
        root_cause="The application image is missing the gunicorn dependency.",
        confidence=0.92,
        evidence_ids=evidence_ids,
        recommended_action=(
            "Have an engineer correct and rebuild the image, validate it, and "
            "use the reviewed deployment process."
        ),
        limitations=["Only supplied read-only evidence was inspected."],
        human_review_required=True,
    )


def _button(app: AppTest, label: str) -> Any:
    return next(button for button in app.button if button.label == label)


def _load_demo(app: AppTest, label: str) -> AppTest:
    _button(app, label).click().run()
    return app


def _analyze_with_model(app: AppTest, model: _UiModel) -> AppTest:
    with patch("agent.graph.NebiusModelAdapter", return_value=model):
        _button(app, "Analyze Deployment Failure").click().run(timeout=15)
    return app


class StreamlitApplicationTests(TestCase):
    def test_streamlit_secrets_fill_only_missing_provider_settings(self) -> None:
        hosted_secrets = {
            "NEBIUS_API_KEY": "hosted-secret-key",
            "NEBIUS_MODEL": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        }
        with patch.dict(
            os.environ,
            {"NEBIUS_API_KEY": "environment-key"},
            clear=True,
        ):
            with patch("app.st.secrets", hosted_secrets):
                _apply_streamlit_provider_secrets()

            self.assertEqual(os.environ["NEBIUS_API_KEY"], "environment-key")
            self.assertEqual(
                os.environ["NEBIUS_MODEL"],
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
            )

    def test_initial_page_has_guided_form_and_three_demo_loaders(self) -> None:
        app = AppTest.from_file(APP_PATH, default_timeout=10).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            app.title[0].value,
            "Kubernetes Deployment Failure Investigator",
        )
        self.assertEqual(len(app.selectbox), 0)
        self.assertEqual(
            [area.label for area in app.text_area],
            [
                "Deployment problem",
                "Workload and pod status",
                "Kubernetes events",
                "Container logs",
                "Manifest YAML",
            ],
        )
        self.assertEqual(
            [item.label for item in app.text_input],
            ["Workload name (optional)", "Namespace (optional)"],
        )
        labels = [button.label for button in app.button]
        for required in (
            "Normal diagnosis",
            "Retry and recovery",
            "Engineer handoff",
            "Analyze Deployment Failure",
        ):
            self.assertIn(required, labels)

    def test_demo_loader_selects_a_case_without_prefilling_its_evidence(
        self,
    ) -> None:
        """A demo names the case; the agent must fetch the evidence itself."""

        app = AppTest.from_file(APP_PATH, default_timeout=10).run()
        _load_demo(app, "Normal diagnosis")

        self.assertEqual(len(app.exception), 0)
        self.assertIn("CrashLoopBackOff", app.text_area(key="form_question").value)
        self.assertEqual(
            app.text_input(key="form_workload_name").value, "order-api"
        )
        self.assertEqual(app.text_input(key="form_namespace").value, "orders")
        self.assertEqual(app.session_state["form_demo_id"], "case_003")
        for evidence_key in (
            "form_workload_status",
            "form_kubernetes_events",
            "form_container_logs",
            "form_manifest_yaml",
        ):
            self.assertEqual(app.text_area(key=evidence_key).value, "")
        rendered = " ".join(
            str(area.value) for area in app.text_area
        ).casefold()
        self.assertNotIn("ground_truth", rendered)
        self.assertNotIn("tool_behavior", rendered)

    def test_diagnosis_flow_displays_required_sections_and_thread_id(self) -> None:
        app = AppTest.from_file(APP_PATH, default_timeout=15).run()
        _load_demo(app, "Normal diagnosis")
        model = _UiModel(
            actions=[
                _action("inspect_container_logs"),
                _action("inspect_workload_status"),
            ],
            diagnosis=_diagnosis(
                [
                    "inspect_container_logs:1",
                    "inspect_workload_status:1",
                ]
            ),
        )
        _analyze_with_model(app, model)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.session_state["investigation_thread_id"])
        self.assertEqual(
            app.session_state["investigation_result"]["status"], "diagnosed"
        )
        subheaders = [item.value for item in app.subheader]
        for required in (
            "Reported problem",
            "Progress timeline",
            "Sources selected by the agent",
            "Likely root cause",
            "Supporting evidence",
            "Recommended manual action",
            "Limitations",
        ):
            self.assertIn(required, subheaders)
        self.assertEqual(app.metric[0].label, "Confidence")
        self.assertEqual(app.metric[0].value, "92%")
        rendered = " ".join(item.value for item in app.markdown)
        self.assertNotIn("model_errors", rendered)
        asked = app.session_state["investigation_result"]["question"]
        self.assertIn(asked, rendered)

    def test_question_only_input_returns_clarification_request(self) -> None:
        app = AppTest.from_file(APP_PATH, default_timeout=15).run()
        app.text_area(key="form_question").set_value(
            "Why is the deployment stuck in ImagePullBackOff?"
        ).run()
        _analyze_with_model(app, _UiModel())

        self.assertEqual(len(app.exception), 0)
        result = app.session_state["investigation_result"]
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(
            result["clarification_request"].requested_sources,
            ["kubernetes_events"],
        )
        self.assertIn(
            "Clarification request", [item.value for item in app.subheader]
        )
        self.assertIn(
            "Requested evidence", [item.value for item in app.subheader]
        )

    def test_clarification_returns_to_the_populated_form(self) -> None:
        question = "Why is the deployment stuck in ImagePullBackOff?"
        app = AppTest.from_file(APP_PATH, default_timeout=15).run()
        app.text_area(key="form_question").set_value(question).run()
        _analyze_with_model(app, _UiModel())

        _button(app, "Add Requested Evidence").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("investigation_result", app.session_state)
        self.assertNotIn("investigation_thread_id", app.session_state)
        self.assertEqual(
            app.text_area(key="form_question").value,
            question,
        )
        self.assertIn(
            "Analyze Deployment Failure", [button.label for button in app.button]
        )

    def test_retry_demo_shows_one_failure_and_one_success(self) -> None:
        app = AppTest.from_file(APP_PATH, default_timeout=15).run()
        _load_demo(app, "Retry and recovery")
        self.assertEqual(app.session_state["form_demo_id"], "case_009")
        self.assertEqual(
            app.text_area(key="form_kubernetes_events").value, ""
        )
        model = _UiModel(
            actions=[
                _action("inspect_kubernetes_events"),
                _action("inspect_workload_status"),
            ],
            diagnosis=_diagnosis(
                [
                    "inspect_kubernetes_events:2",
                    "inspect_workload_status:1",
                ]
            ),
        )
        _analyze_with_model(app, model)

        result = app.session_state["investigation_result"]
        self.assertEqual(result["status"], "diagnosed")
        self.assertEqual(
            result["attempts_by_tool"]["inspect_kubernetes_events"], 2
        )
        timeline = " ".join(
            item.value for item in app.markdown if isinstance(item.value, str)
        )
        self.assertIn("temporary failure; one retry allowed", timeline)
        self.assertIn("2 attempts", timeline)

    def test_handoff_demo_can_be_acknowledged_and_ended_as_escalated(self) -> None:
        app = AppTest.from_file(APP_PATH, default_timeout=15).run()
        _load_demo(app, "Engineer handoff")
        model = _UiModel(
            actions=[
                _action("inspect_workload_status"),
                _action("inspect_container_logs"),
                _action("inspect_kubernetes_events"),
                _action("inspect_manifest_config"),
            ]
        )
        _analyze_with_model(app, model)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            app.session_state["investigation_result"]["status"],
            "awaiting_human",
        )
        for required in (
            "Handoff reason",
            "Preserved evidence",
            "Tool failures",
            "Leading hypotheses",
            "Unresolved question",
            "Recommended manual check",
        ):
            self.assertIn(required, [item.value for item in app.subheader])

        _button(app, "Acknowledge and End as Escalated").click().run()

        self.assertEqual(len(app.exception), 0)
        result = app.session_state["investigation_result"]
        self.assertEqual(result["status"], "escalated")
        self.assertTrue(result["human_acknowledged"])
        self.assertNotIn(
            "Acknowledge and End as Escalated",
            [button.label for button in app.button],
        )

    def test_model_format_failure_can_retry_without_reentering_the_form(
        self,
    ) -> None:
        app = AppTest.from_file(APP_PATH, default_timeout=15).run()
        _load_demo(app, "Normal diagnosis")
        original_question = app.text_area(key="form_question").value
        failing_model = _UiModel(
            actions=[
                _action("inspect_container_logs"),
                _action("inspect_workload_status"),
            ]
        )
        _analyze_with_model(app, failing_model)

        self.assertEqual(
            app.session_state["investigation_result"]["status"],
            "awaiting_human",
        )
        self.assertIn("Retry Analysis", [button.label for button in app.button])
        rendered = " ".join(item.value for item in app.markdown)
        self.assertIn("Analysis Could Not Be Finalized", rendered)

        successful_model = _UiModel(
            actions=[
                _action("inspect_container_logs"),
                _action("inspect_workload_status"),
            ],
            diagnosis=_diagnosis(
                [
                    "inspect_container_logs:1",
                    "inspect_workload_status:1",
                ]
            ),
        )
        with patch(
            "agent.graph.NebiusModelAdapter", return_value=successful_model
        ):
            _button(app, "Retry Analysis").click().run(timeout=15)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            app.session_state["investigation_result"]["status"], "diagnosed"
        )
        self.assertEqual(app.session_state["form_question"], original_question)

    def test_start_new_investigation_resets_result_and_thread(self) -> None:
        app = AppTest.from_file(APP_PATH, default_timeout=15).run()
        app.text_area(key="form_question").set_value(
            "Why is the deployment stuck in ImagePullBackOff?"
        ).run()
        _analyze_with_model(app, _UiModel())
        self.assertIn("investigation_thread_id", app.session_state)

        _button(app, "Start New Investigation").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("investigation_result", app.session_state)
        self.assertNotIn("investigation_thread_id", app.session_state)
        self.assertEqual(app.text_area(key="form_question").value, "")
        self.assertIn(
            "Analyze Deployment Failure", [button.label for button in app.button]
        )

    def test_missing_provider_configuration_is_safe_and_actionable(self) -> None:
        app = AppTest.from_file(APP_PATH, default_timeout=10).run()
        app.text_area(key="form_question").set_value(
            "Why is this Kubernetes deployment failing to become ready?"
        ).run()

        with patch(
            "agent.graph.NebiusModelAdapter",
            side_effect=NebiusConfigurationError("NEBIUS_API_KEY is required."),
        ):
            _button(app, "Analyze Deployment Failure").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertIn("NEBIUS_API_KEY is required", app.error[0].value)
        self.assertNotIn("investigation_graph", app.session_state)

    def test_invalid_input_shows_safe_validation_without_starting_graph(self) -> None:
        app = AppTest.from_file(APP_PATH, default_timeout=10).run()
        secret = "secret-must-not-appear-in-error"
        app.text_area(key="form_question").set_value("please help me").run()
        app.text_area(key="form_container_logs").set_value(
            f"password={secret}"
        ).run()

        _button(app, "Analyze Deployment Failure").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertIn("deployment failure symptom", app.error[0].value)
        self.assertNotIn(secret, app.error[0].value)
        self.assertNotIn("investigation_graph", app.session_state)

    def test_redacted_credentials_never_appear_in_rendered_results(self) -> None:
        raw_api_key = "sk-rendered-secret-1234567890"
        raw_password = "rendered-password-must-disappear"
        app = AppTest.from_file(APP_PATH, default_timeout=15).run()
        app.text_area(key="form_question").set_value(
            "Why does payment-api keep crashing during startup?"
        ).run()
        app.text_area(key="form_workload_status").set_value(
            '{"pods":[{"container":{"state":{"reason":"CrashLoopBackOff"}}}]}'
        ).run()
        app.text_area(key="form_container_logs").set_value(
            "ModuleNotFoundError: No module named 'gunicorn'\n"
            f"api_key={raw_api_key}\npassword={raw_password}"
        ).run()
        model = _UiModel(
            actions=[
                _action("inspect_container_logs"),
                _action("inspect_workload_status"),
            ],
            diagnosis=_diagnosis(
                [
                    "inspect_container_logs:1",
                    "inspect_workload_status:1",
                ]
            ),
        )

        _analyze_with_model(app, model)

        self.assertEqual(len(app.exception), 0)
        state_text = json.dumps(
            app.session_state["investigation_result"], default=str
        )
        rendered_text = " ".join(
            str(item.value)
            for collection in (
                app.markdown,
                app.caption,
                app.info,
                app.success,
                app.warning,
                app.error,
            )
            for item in collection
        )
        for secret in (raw_api_key, raw_password):
            self.assertNotIn(secret, state_text)
            self.assertNotIn(secret, rendered_text)
        self.assertIn("__REDACTED_API_TOKEN__", state_text)
        self.assertIn("__REDACTED_PASSWORD__", state_text)

    def test_public_trace_does_not_expose_model_reasoning_or_errors(self) -> None:
        private_reason = "private chain reasoning must stay hidden"
        internal_error = "raw provider response must stay hidden"
        rows = _trace_rows(
            [
                {
                    "phase": "selection",
                    "action": "inspect_kubernetes_events",
                    "reason": private_reason,
                    "model_error": internal_error,
                },
                {
                    "phase": "tool",
                    "tool": "inspect_kubernetes_events",
                    "attempt": 1,
                    "ok": False,
                    "retryable": True,
                },
            ]
        )

        rendered = " ".join(rows)
        self.assertNotIn(private_reason, rendered)
        self.assertNotIn(internal_error, rendered)
        self.assertIn("Kubernetes events", rendered)
        self.assertIn("retry allowed", rendered)

    def test_every_selection_step_reports_who_decided_it(self) -> None:
        rows = _trace_rows(
            [
                {
                    "phase": "selection",
                    "action": "inspect_workload_status",
                    "reason": "public reason",
                    "decided_by": "model",
                    "deviated_from_default": True,
                },
                {
                    "phase": "selection",
                    "action": "inspect_kubernetes_events",
                    "reason": "public reason",
                    "decided_by": "model",
                    "deviated_from_default": False,
                },
                {
                    "phase": "selection",
                    "action": "inspect_container_logs",
                    "reason": "public reason",
                    "decided_by": "guard",
                    "deviated_from_default": False,
                },
                {
                    "phase": "selection",
                    "action": "handoff",
                    "reason": "public reason",
                    "decided_by": "deterministic",
                    "deviated_from_default": False,
                },
            ]
        )

        self.assertEqual(len(rows), 4)
        self.assertIn("model choice (differs from default order)", rows[0])
        self.assertIn("model choice (matches default order)", rows[1])
        self.assertIn("safety guard replaced the model's proposal", rows[2])
        self.assertIn("deterministic workflow step", rows[3])


if __name__ == "__main__":
    import unittest

    unittest.main()
