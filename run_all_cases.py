"""Run all ten corpus cases through the real graph exactly as a demo button
would: select the case, submit its metadata, no pasted evidence.
"""

from __future__ import annotations

import os
from uuid import uuid4

from agent.graph import (
    build_investigation_graph,
    graph_input,
    investigation_config,
)
from agent.schemas import SubmittedInvestigation
from tools import load_case_metadata
from tools.cluster import reset_cluster_source_cache
from tools.diagnostics import _reset_demo_event_call_counts

CASE_IDS = tuple(f"case_{number:03d}" for number in range(1, 11))


def main() -> None:
    """Run every synthetic case and print a concise result table."""

    # Force the bundled-case source even when the operator's shell is configured
    # for a snapshot or throwaway live cluster.
    # Empty values also prevent python-dotenv from restoring settings from .env.
    os.environ["K8S_SNAPSHOT_DIR"] = ""
    os.environ["K8S_LIVE_CLUSTER"] = ""
    os.environ["K8S_CONTEXT"] = ""

    print(
        f"{'case':<10} {'status':<18} {'category':<14} "
        f"{'conf':<6} {'tools':<40} root_cause"
    )
    print("-" * 130)

    for case_id in CASE_IDS:
        reset_cluster_source_cache()
        _reset_demo_event_call_counts()
        metadata = load_case_metadata(case_id)
        submission = SubmittedInvestigation(
            question=metadata.initial_symptom,
            workload_name=metadata.workload_name,
            namespace=metadata.namespace,
            demo_id=case_id,
        )
        graph = build_investigation_graph()
        result = graph.invoke(
            graph_input(submission),
            config=investigation_config(f"case-runner-{uuid4()}"),
        )

        status = result["status"]
        category = result["symptom_assessment"].category
        tools = ",".join(
            tool.replace("inspect_", "") for tool in result["tools_called"]
        )
        diagnosis = result.get("diagnosis")
        if diagnosis is not None:
            confidence = f"{diagnosis.confidence:.2f}"
            root_cause = diagnosis.root_cause[:70]
        elif result.get("handoff") is not None:
            confidence = "-"
            root_cause = f"HANDOFF: {result['handoff'].handoff_reason[:60]}"
        elif result.get("clarification_request") is not None:
            confidence = "-"
            requested = result["clarification_request"].requested_sources
            root_cause = f"ASKS FOR: {requested}"
        else:
            confidence = "-"
            root_cause = "(no diagnosis, no handoff, no clarification)"

        print(
            f"{case_id:<10} {status:<18} {category:<14} "
            f"{confidence:<6} {tools:<40} {root_cause}"
        )


if __name__ == "__main__":
    main()
