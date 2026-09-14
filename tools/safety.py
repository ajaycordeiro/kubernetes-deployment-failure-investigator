"""Pure safety helpers for treating diagnostic text as untrusted data."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final


UNTRUSTED_INSTRUCTION_MARKER: Final[str] = (
    "__UNTRUSTED_INSTRUCTION_REMOVED__"
)
_INSTRUCTION_PATTERN = re.compile(
    r"(?i)(?:"
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|system)\s+instructions|"
    r"reveal\s+(?:the\s+)?(?:system\s+prompt|secrets?|credentials?)|"
    r"(?:run|execute|invoke)\s+(?:this\s+|the\s+)?(?:command|kubectl|shell)|"
    r"you\s+are\s+(?:chatgpt|an?\s+assistant)|"
    r"system\s+prompt"
    r")"
)


def neutralize_untrusted_text(value: str) -> tuple[str, int]:
    """Replace instruction-shaped lines without interpreting their contents."""

    lines = value.splitlines() or [value]
    cleaned: list[str] = []
    removed = 0
    for line in lines:
        if _INSTRUCTION_PATTERN.search(line):
            cleaned.append(UNTRUSTED_INSTRUCTION_MARKER)
            removed += 1
        else:
            cleaned.append(line)
    return "\n".join(cleaned), removed


def neutralize_untrusted_value(value: object) -> tuple[object, int]:
    """Neutralize instruction-shaped strings in a parsed data structure."""

    if isinstance(value, str):
        return neutralize_untrusted_text(value)
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        removed = 0
        for key, item in value.items():
            safe_item, item_count = neutralize_untrusted_value(item)
            cleaned[str(key)[:253]] = safe_item
            removed += item_count
        return cleaned, removed
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        cleaned_items: list[object] = []
        removed = 0
        for item in value:
            safe_item, item_count = neutralize_untrusted_value(item)
            cleaned_items.append(safe_item)
            removed += item_count
        return cleaned_items, removed
    if value is None or isinstance(value, (bool, int, float)):
        return value, 0
    return str(value)[:500], 0


def bound_text(
    value: str,
    *,
    maximum_chars: int,
    maximum_lines: int,
    keep_recent: bool = False,
) -> tuple[str, bool]:
    """Return a deterministic excerpt bounded by both lines and characters."""

    lines = value.splitlines()
    selected = (
        lines[-maximum_lines:] if keep_recent else lines[:maximum_lines]
    )
    excerpt = "\n".join(selected)
    truncated = len(selected) < len(lines)
    if len(excerpt) > maximum_chars:
        excerpt = (
            excerpt[-maximum_chars:]
            if keep_recent
            else excerpt[:maximum_chars]
        )
        truncated = True
    return excerpt, truncated


__all__ = [
    "UNTRUSTED_INSTRUCTION_MARKER",
    "bound_text",
    "neutralize_untrusted_text",
    "neutralize_untrusted_value",
]
