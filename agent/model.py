"""Nebius Token Factory model configuration and structured-output adapter."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr, field_validator

from agent.schemas import (
    ActionSelection,
    DiagnosisResult,
    StrictModel,
    SymptomAssessment,
)


NEBIUS_BASE_URL: Final[str] = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[float] = 60.0
DEFAULT_PROVIDER_RETRIES: Final[int] = 2


class NebiusConfigurationError(RuntimeError):
    """Raised when required Nebius configuration is absent or invalid."""


class NebiusSettings(StrictModel):
    """Validated provider settings kept outside investigation state."""

    api_key: SecretStr
    model_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$",
    )
    base_url: str = NEBIUS_BASE_URL

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_nebius(cls, value: str) -> str:
        """Prevent a provider key from being redirected to another host."""

        if value != NEBIUS_BASE_URL:
            raise ValueError("The model endpoint must be Nebius Token Factory.")
        return value


class InvestigationModel(Protocol):
    """Small interface used by graph nodes and mocked by unit tests."""

    def select_action(
        self, messages: Sequence[BaseMessage]
    ) -> ActionSelection:
        """Return one validated next action."""

    def produce_diagnosis(
        self, messages: Sequence[BaseMessage]
    ) -> DiagnosisResult:
        """Return one validated final diagnosis."""


class SymptomClassifier(Protocol):
    """Optional capability; the graph falls back to local rules without it."""

    def assess_symptom(
        self, messages: Sequence[BaseMessage]
    ) -> SymptomAssessment:
        """Return one validated symptom classification."""


def load_nebius_settings(
    environment: Mapping[str, str] | None = None,
) -> NebiusSettings:
    """Load validated settings without ever returning or logging a raw key."""

    if environment is None:
        load_dotenv()
        environment = os.environ

    api_key = environment.get("NEBIUS_API_KEY", "").strip()
    model_id = environment.get("NEBIUS_MODEL", "").strip()
    if not api_key:
        raise NebiusConfigurationError("NEBIUS_API_KEY is required.")
    if not model_id:
        raise NebiusConfigurationError("NEBIUS_MODEL is required.")

    try:
        return NebiusSettings(
            api_key=SecretStr(api_key),
            model_id=model_id,
        )
    except ValueError as error:
        raise NebiusConfigurationError(
            "Nebius configuration is invalid. Check NEBIUS_MODEL."
        ) from error


def create_chat_model(
    settings: NebiusSettings | None = None,
) -> ChatOpenAI:
    """Create an OpenAI-compatible client that sends requests only to Nebius."""

    resolved = settings or load_nebius_settings()
    return ChatOpenAI(
        api_key=resolved.api_key,
        base_url=NEBIUS_BASE_URL,
        model=resolved.model_id,
        temperature=0,
        timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        max_retries=DEFAULT_PROVIDER_RETRIES,
        use_responses_api=False,
    )


class NebiusModelAdapter:
    """Expose only the structured model operations required by the graph."""

    def __init__(self, chat_model: ChatOpenAI | None = None) -> None:
        model = chat_model or create_chat_model()
        self._action_model: Runnable[Any, Any] = model.with_structured_output(
            ActionSelection,
            method="json_schema",
        )
        self._diagnosis_model: Runnable[Any, Any] = model.with_structured_output(
            DiagnosisResult,
            method="json_schema",
        )
        self._assessment_model: Runnable[Any, Any] = model.with_structured_output(
            SymptomAssessment,
            method="json_schema",
        )

    def assess_symptom(
        self, messages: Sequence[BaseMessage]
    ) -> SymptomAssessment:
        """Classify and locally validate one reported deployment symptom."""

        raw_result = self._assessment_model.invoke(list(messages))
        return SymptomAssessment.model_validate(raw_result)

    def select_action(
        self, messages: Sequence[BaseMessage]
    ) -> ActionSelection:
        """Select and locally validate exactly one graph action."""

        raw_result = self._action_model.invoke(list(messages))
        return ActionSelection.model_validate(raw_result)

    def produce_diagnosis(
        self, messages: Sequence[BaseMessage]
    ) -> DiagnosisResult:
        """Produce and locally validate an evidence-grounded diagnosis."""

        raw_result = self._diagnosis_model.invoke(list(messages))
        return DiagnosisResult.model_validate(raw_result)


__all__ = [
    "DEFAULT_PROVIDER_RETRIES",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "InvestigationModel",
    "NEBIUS_BASE_URL",
    "NebiusConfigurationError",
    "NebiusModelAdapter",
    "NebiusSettings",
    "SymptomClassifier",
    "create_chat_model",
    "load_nebius_settings",
]
