"""Unit tests for the Nebius model boundary; no live API calls are made."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from agent.model import (
    NEBIUS_BASE_URL,
    NebiusConfigurationError,
    NebiusModelAdapter,
    create_chat_model,
    load_nebius_settings,
)
from agent.schemas import (
    ActionSelection,
    DiagnosisResult,
    SymptomAssessment,
)


class _StructuredRunnable:
    def __init__(self, result: object) -> None:
        self.result = result

    def invoke(self, messages: object) -> object:
        return self.result


class _FakeChatModel:
    def __init__(self) -> None:
        self.schemas: list[type[object]] = []

    def with_structured_output(
        self, schema: type[object], **kwargs: object
    ) -> _StructuredRunnable:
        self.schemas.append(schema)
        if schema is ActionSelection:
            result: object = {
                "action": "inspect_workload_status",
                "reason": "Status is the safest first source.",
                "tool_input": {},
            }
        elif schema is SymptomAssessment:
            result = {
                "category": "image_pull",
                "summary": "The question reports an image retrieval failure.",
                "observable_signals": [],
                "relevant_sources": [],
                "missing_sources": [],
            }
        else:
            result = {
                "root_cause": "The declared image tag does not exist.",
                "confidence": 0.95,
                "evidence_ids": ["inspect_kubernetes_events:1"],
                "recommended_action": "An engineer should verify the image tag.",
                "human_review_required": True,
            }
        return _StructuredRunnable(result)


class NebiusSettingsTests(TestCase):
    def test_requires_nebius_key_and_model(self) -> None:
        with self.assertRaisesRegex(
            NebiusConfigurationError, "NEBIUS_API_KEY"
        ):
            load_nebius_settings({})
        with self.assertRaisesRegex(NebiusConfigurationError, "NEBIUS_MODEL"):
            load_nebius_settings({"NEBIUS_API_KEY": "secret-value"})

    def test_ignores_openai_variables(self) -> None:
        with self.assertRaisesRegex(
            NebiusConfigurationError, "NEBIUS_API_KEY"
        ):
            load_nebius_settings(
                {
                    "OPENAI_API_KEY": "must-not-be-used",
                    "OPENAI_MODEL": "must-not-be-used",
                }
            )

    def test_validates_and_masks_nebius_settings(self) -> None:
        settings = load_nebius_settings(
            {
                "NEBIUS_API_KEY": "private-nebius-key",
                "NEBIUS_MODEL": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            }
        )
        self.assertEqual(settings.base_url, NEBIUS_BASE_URL)
        self.assertEqual(
            settings.model_id, "Qwen/Qwen3-30B-A3B-Instruct-2507"
        )
        self.assertNotIn("private-nebius-key", repr(settings))

    def test_rejects_malformed_model_id(self) -> None:
        with self.assertRaises(NebiusConfigurationError):
            load_nebius_settings(
                {
                    "NEBIUS_API_KEY": "private-nebius-key",
                    "NEBIUS_MODEL": "bad model id",
                }
            )


class NebiusClientTests(TestCase):
    @patch("agent.model.ChatOpenAI")
    def test_client_uses_only_fixed_nebius_endpoint(self, chat_class: object) -> None:
        settings = load_nebius_settings(
            {
                "NEBIUS_API_KEY": "private-nebius-key",
                "NEBIUS_MODEL": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            }
        )
        create_chat_model(settings)

        kwargs = chat_class.call_args.kwargs
        self.assertEqual(kwargs["base_url"], NEBIUS_BASE_URL)
        self.assertEqual(
            kwargs["api_key"].get_secret_value(), "private-nebius-key"
        )
        self.assertEqual(
            kwargs["model"], "Qwen/Qwen3-30B-A3B-Instruct-2507"
        )
        self.assertEqual(kwargs["temperature"], 0)
        self.assertFalse(kwargs["use_responses_api"])

    def test_adapter_validates_every_structured_output(self) -> None:
        fake_chat = _FakeChatModel()
        adapter = NebiusModelAdapter(fake_chat)

        action = adapter.select_action([])
        diagnosis = adapter.produce_diagnosis([])
        assessment = adapter.assess_symptom([])

        self.assertIsInstance(action, ActionSelection)
        self.assertIsInstance(diagnosis, DiagnosisResult)
        self.assertIsInstance(assessment, SymptomAssessment)
        self.assertEqual(assessment.category, "image_pull")
        self.assertEqual(
            fake_chat.schemas,
            [ActionSelection, DiagnosisResult, SymptomAssessment],
        )
