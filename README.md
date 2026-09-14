# Kubernetes Deployment Failure Investigator

A Week 3 certification project demonstrating a bounded, stateful AI agent with
LangChain, LangGraph, Nebius Token Factory, and Streamlit. `app.py` is the
application entry point.

## Problem statement

Kubernetes deployment troubleshooting often requires an engineer to correlate a
problem description with workload status, warning events, container logs, and
manifest configuration. This application accepts those details in plain English,
selects only relevant read-only inspection tools, preserves investigation state,
and returns one of three safe outcomes:

- an evidence-supported diagnosis;
- a focused request for missing evidence; or
- a structured handoff for engineer review.

The project uses synthetic data and never connects to a Kubernetes cluster.

## Target users

- SRE and platform engineers learning or reviewing Kubernetes diagnostics
- Cloud and DevOps engineers triaging a failed application rollout
- Evaluators assessing tool use, state, recovery, safety, and human-in-the-loop
  behavior in an agentic application

## Architecture

~~~text
Streamlit form / demonstration loader
                |
                v
Pydantic validation, normalization, YAML checks, and redaction
                |
                v
Checkpointed LangGraph investigation
  |             |                    |
  v             v                    v
Nebius       Five fixed          Deterministic
structured   read-only tools     safety/routing guards
output           |                    |
                 +---------+----------+
                           v
          Diagnosis / clarification / engineer handoff
~~~

`langchain-openai` supplies the OpenAI-compatible client, but requests use only
the fixed Nebius Token Factory endpoint. LangGraph uses an in-memory checkpointer
and a session-scoped thread ID. Runtime fixtures are resolved from the repository
location, so execution does not depend on the current working directory.

## LangGraph workflow

~~~text
START
  -> validate_input
  -> assess_question
  -> select_next_action
  -> execute_tool
  -> analyze_evidence
  -> route_investigation
       -> select_next_action (bounded loop)
       -> request_clarification
       -> produce_diagnosis
       -> prepare_handoff -> human_handoff
  -> format_result
  -> END
~~~

The model classifies the reported symptom, proposes an action, and drafts a
diagnosis through Pydantic-validated structured output. Code-level guards
enforce the tool allowlist, source availability, evidence threshold, retry
limit, repetition prevention, and final confidence requirement.
Model-generated identity or evidence arguments are not trusted; tools receive
the canonical sanitized submission from graph state.

## Who decides what

The split between model judgment and deterministic control is deliberate. The
model is given the open judgment calls; code holds the veto on anything that
could turn into an unsupported conclusion. A diagnostic tool that confidently
invents a root cause is worse than one that escalates.

| Decision | Owner | Guard on it |
|---|---|---|
| Which failure category the question describes | Model | Schema restricts it to seven categories; local keyword rules take over if the call fails |
| Which evidence source to inspect next | Model | Allowlist, availability, and no-repeat checks can replace the choice |
| Whether to stop gathering and diagnose | Model proposes | Replaced unless the independent-evidence rules are already satisfied |
| Whether the evidence is sufficient | Code | Not delegated |
| Whether evidence is contradictory or unsafe | Code | Not delegated |
| Root cause and recommended action wording | Model | Rejected unless every cited evidence ID exists and the support rules hold |
| Ending the investigation as escalated | Human | `interrupt()` pause; only an explicit acknowledgment resumes |

The public trace records whether each step was the model's own choice, a
deterministic fallback, or a guard replacing the proposal, so this split is
visible at run time instead of only described here.

## Diagnostic tools

| Tool | Purpose |
|---|---|
| `inspect_workload_status` | Inspect bounded submitted Deployment and Pod status. |
| `inspect_kubernetes_events` | Inspect bounded recent event data and warning signals. |
| `inspect_container_logs` | Inspect a bounded recent log excerpt or record that logs are unavailable. |
| `inspect_manifest_config` | Parse sanitized YAML and expose only diagnosis-relevant configuration. |
| `search_runbook` | Search the bundled sanitized runbook by heading and keyword; results are reference context, never incident proof. |

All five capabilities are read-only. The three UI demonstrations populate the
same validated form used for user input; they do not bypass the graph or read
test ground truth.

## State design

`InvestigationState` carries only validated and sanitized information:

- question, optional workload identity, and evidence availability;
- structured symptom assessment;
- collected `EvidenceItem` records and separate runbook reference context;
- selected tools, successful tools, per-tool attempts, and normalized errors;
- public decisions, provisional hypotheses, and contradiction/unsafe flags;
- clarification, diagnosis, confidence, or `HandoffPayload`;
- human acknowledgment and final outcome.

API keys are never added to graph state. Raw submitted secrets are replaced with
visible redaction markers before checkpointing. The UI shows a concise public
trace, not chain-of-thought, prompts, raw provider errors, or stack traces.

## Evidence sufficiency rules

A question describes the symptom but cannot prove a cause. Runbook passages also
cannot prove what occurred. A diagnosis requires compatible evidence from at
least two independent submitted incident sources, including either:

- one decisive runtime signal plus matching context; or
- two compatible independent incident signals for a cause-specific failure.

The diagnosis must cite visible evidence IDs and have confidence of at least
`0.80`. Contradictory, unsafe, unsupported, or exhausted evidence is handed to an
engineer instead of being guessed. A readiness-probe diagnosis additionally
requires manifest or container-log corroboration; workload status plus a generic
probe failure is not enough to infer a configuration mismatch.

## Retry, recovery, and stopping behavior

- At most six tool executions are allowed, including retries.
- A transient tool failure can be retried once.
- A successful tool cannot be repeated.
- Missing or unavailable evidence is a valid non-retryable result and can lead to
  clarification.
- The recovery demonstration (`case_009`) makes its first events inspection time
  out and its second inspection succeed; both attempts remain visible.

## Human in the loop

When evidence is contradictory, unsafe, insufficient after available sources are
exhausted, or the model cannot produce valid high-confidence structured output,
the graph creates a JSON-serializable `HandoffPayload`. LangGraph pauses with
`interrupt`. The UI displays the reason, preserved evidence, failures,
hypotheses, unresolved question, and recommended manual check. Only an explicit
**Acknowledge and End as Escalated** action resumes and closes the investigation.
`DiagnosisResult.human_review_required` is typed `Literal[True]` on purpose: it
is a structured-output constraint rather than a variable, so the model cannot
emit a diagnosis that claims its own recommendation is safe to apply unreviewed.
If the model response misses the citation contract, the UI identifies it as an
analysis-format failure and offers a retry without losing the sanitized form.
Clarification outcomes likewise return to the populated form so the user can add
the requested evidence instead of starting over.

## Security boundaries

- Synthetic, sanitized corpus or user-pasted evidence only
- No live Kubernetes or OpenShift client and no kubeconfig loading
- No `kubectl`, shell, subprocess, or operating-system command execution in
  runtime code
- No restart, delete, scale, patch, apply, rollback, redeploy, or other
  remediation capability
- Fixed five-tool runtime allowlist; every tool is read-only
- Strict field sizes, unexpected-field rejection, YAML validation, bounded tool
  results, prompt-injection neutralization, and credential/Secret redaction
- Fixed model base URL: `https://api.tokenfactory.nebius.com/v1/`
- Runtime code cannot import or open `tests/ground_truth.json`
- Recommendations are advisory and require engineer review before action

## Repository structure

~~~text
.
├── app.py                       # Streamlit entry point
├── agent/
│   ├── graph.py                 # The single natural-language LangGraph workflow
│   ├── input.py                 # Normalization, limits, YAML checks, redaction
│   ├── model.py                 # Nebius settings and structured-output adapter
│   ├── nodes.py                 # Workflow nodes and fixed tool registry
│   ├── prompts.py               # Bounded action and diagnosis prompts
│   ├── routing.py               # Evidence, retry, and stopping guards
│   ├── schemas.py               # Pydantic contracts
│   ├── state.py                 # LangGraph TypedDict state
├── tools/
│   ├── diagnostics.py           # Exactly four incident-evidence inspection tools
│   ├── runbook.py               # The fifth, reference-only search tool
│   └── base.py                  # Bounded synthetic demonstration loader
├── data/
│   ├── cases/                   # Ten synthetic cases
│   └── runbooks/                # Sanitized local reference content
├── tests/
│   ├── ground_truth.json        # Test-only evaluation expectations
│   └── test_*.py
├── .env.example
├── .gitignore
├── AGENTS.md
└── requirements.txt
~~~

## Local setup

Use Python 3.11 and run these commands from the repository root.

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
~~~

Add your Nebius settings to `.env`; never commit the populated file.

~~~dotenv
NEBIUS_API_KEY=your-token-factory-key
NEBIUS_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507
~~~

Start the application:

~~~powershell
streamlit run app.py
~~~

Environment variables take precedence. Tests use mocked models and require no
live API key or network access. If configuration is absent or invalid, the page
stays available and shows a safe actionable message only after analysis is
requested.

## Streamlit hosting

1. Push this repository without `.env` or `.streamlit/secrets.toml`.
2. Create a Streamlit Community Cloud app and select `app.py` as the entry point.
3. Select Python 3.11.
4. Add these root-level values in the hosted **Secrets** settings:

~~~toml
NEBIUS_API_KEY = "your-token-factory-key"
NEBIUS_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
~~~

5. Deploy and run the demonstration checklist below. Do not configure a base URL;
   the approved Nebius endpoint is fixed in code.

Required corpus paths are derived from `tools/base.py` using `Path(__file__)` and
remain inside `data/cases/` or `data/runbooks/`.

## Evaluation results

The automated evaluation uses a mocked model and never sends ground truth to the
graph. It covers direct, rephrased, and partial-evidence versions of all ten
cases, plus missing input, irrelevant questions, contradictions, invalid YAML,
oversized content, prompt injection, credential redaction, unavailable logs,
transient recovery, and malformed model output.

Current certification baseline:

- `103` tests passed;
- `249` scenario subtests passed;
- all 30 complete-evidence description variants diagnosed correctly or escalated
  in the provider-independent 50-scenario evaluation;
- every incomplete case clarified or escalated rather than guessing;
- `case_009` retried exactly once and `case_010` escalated;
- every diagnosis cited visible submitted incident evidence;
- no investigation exceeded six tool calls; and
- no secret-like test value appeared in graph state or rendered output.

Run the same checks locally:

~~~powershell
python -m compileall app.py agent tools tests
python -m pytest -q
python -m pip check
~~~

## Known limitations

- There is no live-cluster ingestion, Kubernetes API, `kubectl`, or remediation.
- User evidence is pasted manually; accuracy depends on its completeness and
  authenticity.
- Runbook search uses simple local heading/keyword matching, not RAG, embeddings,
  a vector database, or external memory.
- Checkpoints are in memory and do not survive a process restart.
- There is no authentication, durable audit store, multi-user isolation service,
  or external notification/ticket integration.
- Live results depend on the configured Nebius model producing valid structured
  output; guarded retries and handoff handle failure but cannot guarantee a
  diagnosis.


## Related documentation

- [Corpus guide](data/CORPUS_README.md)
- [Test-only data boundary](tests/README.md)
