"""Tests for read-only retrieval of real Kubernetes evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase
from unittest.mock import patch
from uuid import uuid4

from agent.graph import (
    build_investigation_graph,
    graph_input,
    investigation_config,
)
from agent.schemas import (
    ActionSelection,
    DiagnosisResult,
    SubmittedInvestigation,
)
from tools.diagnostics import inspect_kubernetes_events
from tools.cluster import (
    ClusterAccessError,
    ClusterSnapshotSource,
    LiveClusterSource,
    WorkloadTarget,
    reset_cluster_source_cache,
    resolve_cluster_source,
)


NAMESPACE = "orders"
WORKLOAD = "order-api"
POD = "order-api-7c77dc9457-q6b9s"


def _pods_document() -> dict[str, Any]:
    """Shaped exactly like `kubectl get pods -n orders -o json` output."""

    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "metadata": {"name": POD, "namespace": NAMESPACE},
                "status": {
                    "phase": "Running",
                    "containerStatuses": [
                        {
                            "name": WORKLOAD,
                            "ready": False,
                            "restartCount": 7,
                            "state": {
                                "waiting": {
                                    "reason": "CrashLoopBackOff",
                                    "message": "back-off restarting failed container",
                                }
                            },
                            "lastState": {
                                "terminated": {"reason": "Error", "exitCode": 1}
                            },
                        }
                    ],
                    "conditions": [
                        {
                            "type": "Ready",
                            "status": "False",
                            "reason": "ContainersNotReady",
                        }
                    ],
                },
            },
            {
                "metadata": {"name": "unrelated-api-123", "namespace": NAMESPACE},
                "status": {"phase": "Running"},
            },
            {
                "metadata": {"name": f"{WORKLOAD}-999", "namespace": "other"},
                "status": {"phase": "Running"},
            },
        ],
    }


def _events_document() -> dict[str, Any]:
    return {
        "items": [
            {
                "type": "Warning",
                "reason": "BackOff",
                "message": "Back-off restarting failed container",
                "count": 9,
                "lastTimestamp": "2026-09-13T10:05:00Z",
                "involvedObject": {
                    "kind": "Pod",
                    "name": POD,
                    "namespace": NAMESPACE,
                },
            },
            {
                "type": "Normal",
                "reason": "Pulled",
                "message": "Container image already present",
                "count": 1,
                "lastTimestamp": "2026-09-13T10:01:00Z",
                "involvedObject": {
                    "kind": "Pod",
                    "name": POD,
                    "namespace": NAMESPACE,
                },
            },
            {
                "type": "Warning",
                "reason": "Unhealthy",
                "message": "Should be filtered out; different workload",
                "involvedObject": {
                    "kind": "Pod",
                    "name": "unrelated-api-123",
                    "namespace": NAMESPACE,
                },
            },
        ]
    }


def _deployments_document() -> dict[str, Any]:
    return {
        "items": [
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": WORKLOAD, "namespace": NAMESPACE},
                "spec": {
                    "replicas": 3,
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": WORKLOAD,
                                    "image": "example/order-api:1.4.2",
                                    "readinessProbe": {
                                        "httpGet": {"path": "/healthz", "port": 8080}
                                    },
                                }
                            ]
                        }
                    },
                },
                "status": {
                    "replicas": 3,
                    "readyReplicas": 0,
                    "availableReplicas": 0,
                    "unavailableReplicas": 3,
                },
            }
        ]
    }


def _write_snapshot(directory: Path, *, logs: bool = True) -> None:
    (directory / "pods.json").write_text(
        json.dumps(_pods_document()), encoding="utf-8"
    )
    (directory / "events.json").write_text(
        json.dumps(_events_document()), encoding="utf-8"
    )
    (directory / "deployments.json").write_text(
        json.dumps(_deployments_document()), encoding="utf-8"
    )
    if logs:
        log_directory = directory / "logs"
        log_directory.mkdir(exist_ok=True)
        (log_directory / f"{POD}.log").write_text(
            "Traceback (most recent call last):\n"
            "ModuleNotFoundError: No module named 'gunicorn'\n",
            encoding="utf-8",
        )


class ClusterSnapshotSourceTests(TestCase):
    """A snapshot source retrieves data the application did not already hold."""

    def setUp(self) -> None:
        reset_cluster_source_cache()
        self._temporary = TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        _write_snapshot(self.directory)
        self.source = ClusterSnapshotSource(self.directory)
        self.target = WorkloadTarget(workload_name=WORKLOAD, namespace=NAMESPACE)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_workload_status_is_filtered_to_the_target_workload(self) -> None:
        document = self.source.workload_status(self.target)

        self.assertIsNotNone(document)
        assert document is not None
        payload = json.loads(document.content)
        self.assertEqual(len(payload["pods"]), 1)
        self.assertEqual(payload["pods"][0]["name"], POD)
        self.assertFalse(payload["pods"][0]["ready"])
        self.assertEqual(payload["pods"][0]["reason"], "CrashLoopBackOff")
        self.assertEqual(payload["deployment"]["readyReplicas"], 0)
        self.assertEqual(payload["deployment"]["unavailableReplicas"], 3)
        self.assertIn("snapshot:", document.origin)

    def test_events_are_filtered_sorted_and_projected(self) -> None:
        document = self.source.events(self.target)

        self.assertIsNotNone(document)
        assert document is not None
        payload = json.loads(document.content)
        messages = [item["message"] for item in payload["items"]]
        self.assertNotIn("Should be filtered out; different workload", messages)
        self.assertEqual(len(payload["items"]), 2)
        self.assertEqual(
            payload["items"][-1]["message"],
            "Back-off restarting failed container",
        )

    def test_container_logs_are_read_for_the_matching_pod(self) -> None:
        document = self.source.container_logs(self.target)

        self.assertIsNotNone(document)
        assert document is not None
        self.assertIn("ModuleNotFoundError", document.content)
        self.assertEqual(document.content_format, "text")

    def test_manifest_is_projected_onto_diagnostic_fields(self) -> None:
        document = self.source.manifest(self.target)

        self.assertIsNotNone(document)
        assert document is not None
        payload = json.loads(document.content)
        container = payload["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(container["image"], "example/order-api:1.4.2")
        self.assertEqual(
            container["readinessProbe"]["httpGet"]["path"], "/healthz"
        )

    def test_a_workload_with_no_matching_data_returns_nothing(self) -> None:
        absent = WorkloadTarget(workload_name="not-deployed", namespace=NAMESPACE)

        self.assertIsNone(self.source.workload_status(absent))
        self.assertIsNone(self.source.events(absent))
        self.assertIsNone(self.source.container_logs(absent))
        self.assertIsNone(self.source.manifest(absent))

    def test_missing_logs_directory_is_not_an_error(self) -> None:
        with TemporaryDirectory() as name:
            directory = Path(name)
            _write_snapshot(directory, logs=False)

            self.assertIsNone(
                ClusterSnapshotSource(directory).container_logs(self.target)
            )

    def test_a_missing_directory_is_reported_as_a_cluster_error(self) -> None:
        source = ClusterSnapshotSource(self.directory / "absent")

        with self.assertRaises(ClusterAccessError):
            source.events(self.target)

    def test_unreadable_json_is_reported_as_a_cluster_error(self) -> None:
        (self.directory / "events.json").write_text("{not json", encoding="utf-8")

        with self.assertRaises(ClusterAccessError) as caught:
            self.source.events(self.target)
        self.assertFalse(caught.exception.retryable)

    def test_credentials_in_snapshot_content_are_redacted(self) -> None:
        log_file = self.directory / "logs" / f"{POD}.log"
        log_file.write_text(
            "startup failed\napi_key=sk-livedemosecret0123456789\n",
            encoding="utf-8",
        )

        document = self.source.container_logs(self.target)

        assert document is not None
        self.assertNotIn("sk-livedemosecret0123456789", document.content)
        self.assertIn("__REDACTED", document.content)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


class _FakeCoreApi:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def list_namespaced_pod(self, namespace: str) -> _FakeResponse:
        self.calls.append(f"list_namespaced_pod:{namespace}")
        return _FakeResponse(_pods_document())

    def list_namespaced_event(self, namespace: str) -> _FakeResponse:
        self.calls.append(f"list_namespaced_event:{namespace}")
        return _FakeResponse(_events_document())

    def read_namespaced_pod_log(
        self, name: str, namespace: str, tail_lines: int
    ) -> str:
        self.calls.append(f"read_namespaced_pod_log:{name}")
        return "ModuleNotFoundError: No module named 'gunicorn'\n"


class _FakeAppsApi:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def read_namespaced_deployment(
        self, name: str, namespace: str
    ) -> _FakeResponse:
        self.calls.append(f"read_namespaced_deployment:{name}")
        return _FakeResponse(_deployments_document()["items"][0])


def _live_source_with_fakes() -> tuple[LiveClusterSource, _FakeCoreApi]:
    """Build a live source without touching kubeconfig or the network."""

    source = object.__new__(LiveClusterSource)
    core = _FakeCoreApi()
    source._core = core  # noqa: SLF001 - constructing a test double
    source._apps = _FakeAppsApi()  # noqa: SLF001
    source._serialize = lambda value: value.payload  # noqa: SLF001
    return source, core


class LiveClusterSourceTests(TestCase):
    """The live path issues only read calls and shares the snapshot shape."""

    def setUp(self) -> None:
        self.source, self.core = _live_source_with_fakes()
        self.target = WorkloadTarget(workload_name=WORKLOAD, namespace=NAMESPACE)

    def test_status_matches_the_snapshot_projection(self) -> None:
        document = self.source.workload_status(self.target)

        assert document is not None
        payload = json.loads(document.content)
        self.assertEqual(payload["pods"][0]["reason"], "CrashLoopBackOff")
        self.assertEqual(payload["deployment"]["readyReplicas"], 0)
        self.assertIn("live:", document.origin)

    def test_events_and_logs_are_retrieved_read_only(self) -> None:
        events = self.source.events(self.target)
        logs = self.source.container_logs(self.target)

        assert events is not None and logs is not None
        self.assertIn("Back-off restarting", events.content)
        self.assertIn("ModuleNotFoundError", logs.content)
        for call in self.core.calls:
            self.assertTrue(
                call.startswith(("list_", "read_")),
                msg=f"non read-only client call: {call}",
            )

    def test_an_api_failure_becomes_a_retryable_cluster_error(self) -> None:
        class _Failing:
            status = 503

        def _raise(**_: Any) -> None:
            raise RuntimeError("boom") from _Failing()

        self.source._core.list_namespaced_event = _raise  # noqa: SLF001

        with self.assertRaises(ClusterAccessError) as caught:
            self.source.events(self.target)
        self.assertTrue(caught.exception.retryable)


class ClusterSourceResolutionTests(TestCase):
    """Sources come from operator configuration, never from user input."""

    def setUp(self) -> None:
        reset_cluster_source_cache()

    def tearDown(self) -> None:
        reset_cluster_source_cache()

    def test_no_configuration_means_no_cluster_source(self) -> None:
        self.assertIsNone(resolve_cluster_source({}))

    def test_a_snapshot_directory_selects_the_snapshot_source(self) -> None:
        source = resolve_cluster_source({"K8S_SNAPSHOT_DIR": "/tmp/snapshot"})

        self.assertIsInstance(source, ClusterSnapshotSource)

    def test_resolution_is_cached_until_reset(self) -> None:
        settings = {"K8S_SNAPSHOT_DIR": "/tmp/snapshot"}
        first = resolve_cluster_source(settings)
        self.assertIs(first, resolve_cluster_source(settings))

        reset_cluster_source_cache()
        self.assertIsNot(first, resolve_cluster_source(settings))


class _SnapshotDrivenModel:
    """Deterministic model double for the snapshot integration test."""

    def __init__(self) -> None:
        self.order = ["inspect_container_logs", "inspect_workload_status"]
        self.diagnosis_calls = 0

    def select_action(self, messages: Any) -> ActionSelection:
        payload = json.loads(messages[-1].content)
        called = set(payload.get("tools_called", []))
        for tool_name in self.order:
            if tool_name not in called:
                return ActionSelection(
                    action=tool_name, reason="Inspect the next source."
                )
        return ActionSelection(action="diagnose", reason="Evidence supports it.")

    def produce_diagnosis(self, messages: Any) -> DiagnosisResult:
        self.diagnosis_calls += 1
        payload = json.loads(messages[-1].content)
        return DiagnosisResult(
            root_cause="The image is missing the gunicorn runtime dependency.",
            confidence=0.91,
            evidence_ids=[
                item["evidence_id"]
                for item in payload["visible_incident_evidence"]
            ],
            recommended_action="An engineer should rebuild the image.",
            human_review_required=True,
        )


class SnapshotBackedInvestigationTests(TestCase):
    """The agent investigates using data that was never in its input."""

    def setUp(self) -> None:
        reset_cluster_source_cache()
        self._temporary = TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        _write_snapshot(self.directory)

    def tearDown(self) -> None:
        self._temporary.cleanup()
        reset_cluster_source_cache()

    def test_a_tool_retrieves_evidence_absent_from_the_submission(self) -> None:
        submission = SubmittedInvestigation(
            question="Why does the order-api pod keep restarting?",
            workload_name=WORKLOAD,
            namespace=NAMESPACE,
        )
        self.assertIsNone(submission.kubernetes_events)

        with patch.dict(
            os.environ, {"K8S_SNAPSHOT_DIR": str(self.directory)}, clear=False
        ):
            reset_cluster_source_cache()
            result = inspect_kubernetes_events.invoke(
                {"submission": submission}
            )

        self.assertTrue(result.ok)
        self.assertIn("Back-off restarting", result.data["content"])
        self.assertEqual(result.data["mode"], "cluster")
        self.assertIn("snapshot:", result.source)

    def test_a_full_investigation_runs_with_no_pasted_evidence(self) -> None:
        submission = SubmittedInvestigation(
            question="Why does the order-api pod keep restarting?",
            workload_name=WORKLOAD,
            namespace=NAMESPACE,
        )
        model = _SnapshotDrivenModel()

        with patch.dict(
            os.environ, {"K8S_SNAPSHOT_DIR": str(self.directory)}, clear=False
        ):
            reset_cluster_source_cache()
            graph = build_investigation_graph(model=model)
            result = graph.invoke(
                graph_input(submission),
                config=investigation_config(f"snapshot-{uuid4()}"),
            )

        self.assertTrue(result["cluster_source_available"])
        self.assertEqual(result["status"], "diagnosed")
        self.assertEqual(
            result["tools_called"],
            ["inspect_container_logs", "inspect_workload_status"],
        )
        origins = {item.raw_reference for item in result["evidence"]}
        self.assertTrue(
            all("snapshot:" in origin for origin in origins),
            msg=f"evidence did not come from the snapshot: {origins}",
        )


class SampleSnapshotTests(TestCase):
    """The snapshot shipped for demonstrations must stay usable."""

    def setUp(self) -> None:
        directory = (
            Path(__file__).resolve().parents[1] / "data" / "sample_snapshot"
        )
        self.source = ClusterSnapshotSource(directory)
        self.target = WorkloadTarget(
            workload_name="checkout-api", namespace="checkout"
        )

    def test_the_sample_snapshot_supports_an_image_pull_investigation(
        self,
    ) -> None:
        status = self.source.workload_status(self.target)
        events = self.source.events(self.target)
        manifest = self.source.manifest(self.target)

        assert status is not None and events is not None
        assert manifest is not None
        self.assertIn("ImagePullBackOff", status.content)
        self.assertEqual(json.loads(status.content)["deployment"]["readyReplicas"], 0)
        self.assertIn("manifest unknown", events.content)
        self.assertIn("v3.2.0-rc4", manifest.content)

    def test_the_sample_snapshot_excludes_unrelated_workloads(self) -> None:
        status = self.source.workload_status(self.target)
        events = self.source.events(self.target)

        assert status is not None and events is not None
        self.assertNotIn("checkout-worker", status.content)
        self.assertNotIn("checkout-worker", events.content)

    def test_the_sample_snapshot_reports_logs_as_unavailable(self) -> None:
        logs = self.source.container_logs(self.target)

        assert logs is not None
        self.assertIn("logs unavailable", logs.content)


class ClusterModuleSafetyTests(TestCase):
    """The module must remain incapable of changing cluster state."""

    FORBIDDEN = (
        "create_namespaced",
        "patch_namespaced",
        "replace_namespaced",
        "delete_namespaced",
        "delete_collection",
        "create_namespaced_pod_eviction",
    )

    def test_no_write_verb_appears_in_the_cluster_module(self) -> None:
        source_text = (
            Path(__file__).resolve().parents[1] / "tools" / "cluster.py"
        ).read_text(encoding="utf-8")

        for verb in self.FORBIDDEN:
            self.assertNotIn(verb, source_text)


if __name__ == "__main__":
    import unittest

    unittest.main()
