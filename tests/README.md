# Test-Only Corpus Files

The files in this directory are for automated validation and must not be available to the investigation agent at runtime.

**ground_truth.json** contains expected root causes, acceptable recommendation language, escalation decisions, required evidence sources, and the retry expectation for case 009.

Runtime fixture loaders and LangChain tools must be restricted to:

~~~text
data/cases/
data/runbooks/
~~~

Tests should fail if application code attempts to load ground truth while an investigation is running.

## Natural-language scenario evaluation

`test_scenarios.py` evaluates the submitted-input graph with a mocked model; it
does not require a Nebius API key. Every corpus case is exercised with a direct
description, a differently worded description, and a partial-evidence
submission. Additional scenarios cover missing and irrelevant questions,
contradictory evidence, invalid and oversized input, prompt injection,
credential redaction, unavailable logs, retry recovery, and malformed model
output.

Each run records its final outcome, diagnosis or handoff/clarification detail,
tool sequence, visible evidence sources and citations, attempts and retries,
redaction markers, validation and model errors, unhandled errors, and execution
time. Ground truth is loaded only by the test class after the investigation has
finished and is never included in model prompts or graph state.
