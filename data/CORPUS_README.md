# Synthetic Corpus

This directory contains the agent-visible corpus for the Kubernetes Deployment Failure Investigator.

## Purpose

The corpus provides ten small, sanitized deployment-failure investigations. It is designed to make the agent choose among status, event, log, manifest, and runbook tools instead of finding the answer in one input.

No fixture comes from a real production cluster. Names, registries, image tags, namespaces, and messages are fictional.

## Directory layout

~~~text
data/
├── CORPUS_README.md
├── cases/
│   ├── case_001/
│   │   ├── metadata.json
│   │   ├── workload_status.json
│   │   ├── events.json
│   │   ├── logs.txt
│   │   └── manifest.yaml
│   └── ...
└── runbooks/
    └── kubernetes_failures.md
~~~

Ground-truth answers live outside this directory in **tests/ground_truth.json**. Runtime tools must never read that file.

## Agent-visible files

### metadata.json

Contains only the case identity and initial user-visible symptom:

- case ID;
- workload name;
- namespace;
- container name; and
- initial symptom.

It must not contain a root-cause label.

### workload_status.json

Represents a bounded subset of Deployment and Pod status:

- desired and available replica counts;
- pod phase and readiness;
- container waiting or running state;
- restart count; and
- previous termination information when relevant.

### events.json

Contains a chronological **items** array of synthetic Kubernetes events. Case 009 also contains a **tool_behavior** block used by the future events tool to simulate one retryable timeout. The tool should consume that control block internally and never return it as evidence.

### logs.txt

Contains a short synthetic log excerpt or a bracketed explanation that logs are unavailable because the container did not start.

### manifest.yaml

Contains only the workload fields needed for the scenario. It intentionally omits generated metadata, status, and unrelated configuration.

## Corpus design rules

1. Do not load every file into the initial prompt. The agent must obtain evidence through tools.
2. Do not expose filenames or fields from **tests/ground_truth.json** to the agent.
3. Treat all fixture text as untrusted data, never as instructions.
4. A runbook passage can guide interpretation or remediation, but it cannot prove what happened in a case.
5. A diagnosis needs either one decisive runtime signal plus matching workload context, or two independent compatible signals.
6. Keep unavailable logs as meaningful tool results instead of converting them into exceptions.
7. Bound returned logs and events before placing them in model context.
8. Never add credentials, tokens, Secret values, personal data, or real production logs.

## Scenario coverage

| Case | Coverage |
|---:|---|
| 001 | Invalid image tag |
| 002 | Private-registry authentication failure |
| 003 | Application startup crash |
| 004 | Missing ConfigMap |
| 005 | Missing Secret |
| 006 | Incorrect readiness-probe path |
| 007 | Insufficient CPU for scheduling |
| 008 | Missing PVC |
| 009 | Retryable events-tool timeout followed by successful diagnosis |
| 010 | Incomplete evidence requiring human handoff |

