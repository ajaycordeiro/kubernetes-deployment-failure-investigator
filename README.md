# Kubernetes Deployment Failure Investigator

A Stateful AI agent with
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

Evidence can come from a real cluster, from captured `kubectl` output, from
text an engineer pastes in, or from the bundled synthetic corpus. Every path is
read-only.

## Where evidence comes from

The agent resolves each evidence source in this order, and the interface names
the one in use on every run:

| Order | Source | What it does | How it is enabled |
|---|---|---|---|
| 1 | Live cluster | Read-only Kubernetes API calls for pods, events, one pod log, and one workload definition | `K8S_LIVE_CLUSTER=true` |
| 2 | `kubectl` snapshot | Parses `kubectl ... -o json` files from a directory and filters them to the target workload | `K8S_SNAPSHOT_DIR=<dir>` |
| 3 | Bundled synthetic case | Reads one of ten fixture cases from `data/cases/` at inspection time | Selecting a demonstration |
| 4 | Pasted evidence | Uses text supplied in the form | Filling in the evidence fields |

Sources 1 and 2 retrieve information the application did not already hold, so a
tool call genuinely acquires evidence rather than reformatting its own input.
Sources 3 and 4 exist so the project runs with no cluster at all.

The source is chosen from operator environment variables, never from user
input: a hosted deployment must not let a visitor aim the filesystem reader at
an arbitrary path.

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
                 v                    |
   Live cluster (read-only API)       |
   kubectl snapshot on disk           |
   bundled synthetic case             |
   pasted evidence                    |
                 |                    |
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
| `inspect_workload_status` | Retrieve Deployment and Pod status for the target workload and project it onto diagnosis-relevant fields. |
| `inspect_kubernetes_events` | Retrieve recent events, filter them to the target workload and namespace, and order them by time. |
| `inspect_container_logs` | Retrieve a bounded recent log excerpt, or record that logs are unavailable. |
| `inspect_manifest_config` | Read the workload definition and expose only diagnosis-relevant configuration. |
| `search_runbook` | Search the bundled runbook by heading and keyword; results are reference context, never incident proof. |

All five capabilities are read-only. Against a live cluster the first four use
only read verbs — listing pods and events, reading one pod log, and reading one
Deployment. `tools/cluster.py` imports no write operation, and a test fails the
build if one ever appears there.

`search_runbook` is honest keyword matching over one bundled document, not
retrieval over an index; it is deliberately the weakest of the five and is
never allowed to count as incident evidence.

The three UI demonstrations select a case and let the agent read that case's
evidence from disk as it inspects each source. They do not pre-fill the form,
bypass the graph, or read test ground truth.

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
- A snapshot file that cannot be read, or a cluster call that fails with a 5xx
  or 429, is a genuine retryable failure and consumes the single retry.
- The recovery demonstration uses a synthetic case whose fixture declares its
  own fault: a `tool_behavior` block asks the events source to fail once. This
  is data-driven, so no case identifier appears in runtime tool code, and both
  attempts remain visible in the trace.

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

- Read-only cluster access: only listing pods and events, reading one pod log,
  and reading one Deployment. `tools/cluster.py` imports no write operation and
  a test fails the build if one appears
- No `kubectl`, shell, subprocess, or operating-system command execution in
  runtime code; cluster reads use the official client library
- No restart, delete, scale, patch, apply, rollback, redeploy, or other
  remediation capability
- The evidence source is operator configuration, never user input, so a hosted
  visitor cannot aim the filesystem reader at an arbitrary path
- Retrieved cluster content is sanitized on the way in: credential redaction,
  prompt-injection neutralization, and per-source size bounds apply to live and
  snapshot data exactly as they do to pasted text
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
├── run_all_cases.py             # Optional live-model corpus smoke runner
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
│   ├── cluster.py               # Read-only live and snapshot cluster sources
│   ├── diagnostics.py           # Exactly four incident-evidence inspection tools
│   ├── runbook.py               # The fifth, reference-only search tool
│   └── base.py                  # Bounded synthetic corpus loader
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
live API key, cluster, or network access. If configuration is absent or invalid,
the page stays available and shows a safe actionable message only after analysis
is requested.

## Capturing a cluster snapshot

A snapshot lets the agent investigate real cluster output on a machine with no
cluster access — useful for a laptop, a demo, or attaching evidence to a ticket.
Capture it with read-only `kubectl` commands:

~~~bash
NS=orders
mkdir -p snapshots/$NS/logs
kubectl get pods -n $NS -o json        > snapshots/$NS/pods.json
kubectl get events -n $NS -o json      > snapshots/$NS/events.json
kubectl get deployments -n $NS -o json > snapshots/$NS/deployments.json
kubectl logs -n $NS <pod-name> --tail=200 > snapshots/$NS/logs/<pod-name>.log
~~~

Then point the application at it:

~~~dotenv
K8S_SNAPSHOT_DIR=./snapshots/orders
~~~

The agent filters the snapshot to the workload and namespace it was asked
about; log files are matched to pods by filename. Snapshots contain real
cluster data, so treat them as sensitive and keep them out of version control.

To read from a cluster directly instead, set `K8S_LIVE_CLUSTER=true` and
optionally `K8S_CONTEXT`. Use a local or throwaway cluster — for example a
`kind` cluster with a deliberately broken Deployment such as
`image: nginx:doesnotexist`, which produces a genuine `ImagePullBackOff` with
real events. Never point this project at a production cluster.

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


## Related documentation

- [Corpus guide](data/CORPUS_README.md)
- [Test-only data boundary](tests/README.md)
