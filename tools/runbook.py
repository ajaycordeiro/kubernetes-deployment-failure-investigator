"""Small heading-and-keyword search over the approved local runbook."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from langchain_core.tools import tool
from pydantic import Field, field_validator

from agent.input import normalize_inline_text, redact_sensitive_text
from agent.schemas import StrictModel, ToolResult
from tools.base import FixtureLoadError, fixture_error_result, load_runbook
from tools.safety import neutralize_untrusted_text


MAX_RUNBOOK_RESULTS: Final[int] = 5
MAX_SECTION_CHARS: Final[int] = 4_000
RUNBOOK_SOURCE: Final[str] = "data/runbooks/kubernetes_failures.md"
_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_SECTION_PATTERN = re.compile(r"(?m)^##\s+(.+?)\s*$")
_STOP_WORDS: Final[frozenset[str]] = frozenset(
    {"a", "an", "and", "for", "in", "is", "of", "on", "or", "the", "to"}
)


class RunbookSearchInput(StrictModel):
    """Validated arguments for local runbook search."""

    query: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=3, ge=1, le=MAX_RUNBOOK_RESULTS)

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_query(cls, value: object) -> object:
        """Bound and neutralize a canonical query before local matching."""

        if not isinstance(value, str):
            return value
        if len(value) > 200:
            raise ValueError("Runbook query exceeds the maximum length.")
        normalized = normalize_inline_text(value)
        redacted = redact_sensitive_text(normalized)
        neutralized, _ = neutralize_untrusted_text(redacted)
        return neutralized


@dataclass(frozen=True)
class _RunbookSection:
    heading: str
    content: str
    order: int


def _words(value: str) -> set[str]:
    return set(_WORD_PATTERN.findall(value.casefold()))


def _parse_sections(markdown: str) -> list[_RunbookSection]:
    matches = list(_SECTION_PATTERN.finditer(markdown))
    return [
        _RunbookSection(
            heading=match.group(1).strip(),
            content=markdown[
                match.end() : (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(markdown)
                )
            ].strip(),
            order=index,
        )
        for index, match in enumerate(matches)
    ]


def _score_section(section: _RunbookSection, query: str) -> int:
    query_words = _words(query) - _STOP_WORDS
    heading_words = _words(section.heading)
    content_words = _words(section.content)
    normalized_query = " ".join(query.casefold().split())
    normalized_heading = " ".join(section.heading.casefold().split())
    normalized_content = " ".join(section.content.casefold().split())

    score = 0
    if normalized_query in normalized_heading:
        score += 8
    elif normalized_query in normalized_content:
        score += 3
    for keyword in query_words:
        if keyword in heading_words:
            score += 4
        elif keyword in normalized_heading:
            score += 3
        if keyword in content_words:
            score += 1
    return score


@tool("search_runbook", args_schema=RunbookSearchInput)
def search_runbook(query: str, top_k: int = 3) -> ToolResult:
    """Find relevant troubleshooting sections by heading and keyword overlap."""

    try:
        markdown = load_runbook()
    except FixtureLoadError as error:
        return fixture_error_result(error, source=RUNBOOK_SOURCE)

    scored_sections = [
        (score, section)
        for section in _parse_sections(markdown)
        if (score := _score_section(section, query)) > 0
    ]
    scored_sections.sort(key=lambda item: (-item[0], item[1].order))
    matches = []
    for score, section in scored_sections[:top_k]:
        safe_content, removed = neutralize_untrusted_text(section.content)
        matches.append(
            {
                "heading": section.heading[:200],
                "content": safe_content[:MAX_SECTION_CHARS],
                "score": score,
                "truncated": len(safe_content) > MAX_SECTION_CHARS,
                "untrusted_instructions_removed": removed,
            }
        )
    return ToolResult(
        ok=True,
        data={
            "query": query,
            "matches": matches,
            "reference_context": True,
            "incident_evidence": False,
        },
        source=RUNBOOK_SOURCE,
    )


__all__ = [
    "MAX_RUNBOOK_RESULTS",
    "MAX_SECTION_CHARS",
    "RunbookSearchInput",
    "search_runbook",
]
