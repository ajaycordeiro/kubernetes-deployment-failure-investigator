"""Read-only retrieval of real Kubernetes evidence.

Two sources are supported, and both return data the application did not
already hold:

* ``ClusterSnapshotSource`` reads genuine ``kubectl ... -o json`` output that
  an engineer captured into a directory.
* ``LiveClusterSource`` issues read-only Kubernetes API calls.

Both are strictly read-only. No create, patch, replace, scale, evict, or
delete verb is imported or called anywhere in this module, so neither source
can change cluster state. ``LiveClusterSource`` converts client objects into
the same shape ``kubectl -o json`` produces, so a single set of extractors
serves both paths and the two behave identically.

The active source is chosen from operator environment variables rather than
from user input: a hosted deployment must never let a visitor point the
filesystem reader at an arbitrary path.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from agent.input import redact_sensitive_text
from tools.safety import neutralize_untrusted_text, neutralize_untrusted_value


SNAPSHOT_DIR_VARIABLE: Final[str] = "K8S_SNAPSHOT_DIR"
LIVE_ENABLED_VARIABLE: Final[str] = "K8S_LIVE_CLUSTER"
LIVE_CONTEXT_VARIABLE: Final[str] = "K8S_CONTEXT"

MAX_PODS: Final[int] = 20
MAX_EVENTS: Final[int] = 40
MAX_LOG_LINES: Final[int] = 200
DEFAULT_NAMESPACE: Final[str] = "default"

SNAPSHOT_FILES: Final[dict[str, tuple[str, ...]]] = {
    "pods": ("pods.json",),
    "events": ("events.json",),
    "deployments": ("deployments.json", "deployment.json"),
}


class ClusterAccessError(Exception):
    """A real failure while retrieving evidence from a cluster source."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class WorkloadTarget:
    """Which workload the investigation is about."""

    workload_name: str | None
    namespace: str | None

    @property
    def resolved_namespace(self) -> str:
        return (self.namespace or DEFAULT_NAMESPACE).strip() or DEFAULT_NAMESPACE

    def matches_name(self, candidate: str | None) -> bool:
        """Match a pod or object name against the workload it belongs to."""

        if not self.workload_name:
            return True
        if not candidate:
            return False
        return candidate == self.workload_name or candidate.startswith(
            f"{self.workload_name}-"
        )


@dataclass(frozen=True)
class SourceDocument:
    """One bounded, sanitized document retrieved from a cluster source."""

    content: str
    content_format: Literal["json", "text"]
    origin: str
    untrusted_instructions_removed: int = 0


class ClusterSource(Protocol):
    """Read-only evidence provider. None means 'nothing for this kind'."""

    origin_label: str

    def workload_status(self, target: WorkloadTarget) -> SourceDocument | None: ...

    def events(self, target: WorkloadTarget) -> SourceDocument | None: ...

    def container_logs(self, target: WorkloadTarget) -> SourceDocument | None: ...

    def manifest(self, target: WorkloadTarget) -> SourceDocument | None: ...


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(document: object) -> list[Mapping[str, Any]]:
    listing = _as_mapping(document).get("items")
    if not isinstance(listing, Sequence) or isinstance(listing, (str, bytes)):
        return []
    return [item for item in listing if isinstance(item, Mapping)]


def _json_document(payload: object, origin: str) -> SourceDocument:
    """Sanitize a projection and render it as bounded JSON."""

    neutralized, removed = neutralize_untrusted_value(payload)
    rendered = json.dumps(neutralized, ensure_ascii=True, sort_keys=True)
    return SourceDocument(
        content=redact_sensitive_text(rendered),
        content_format="json",
        origin=origin,
        untrusted_instructions_removed=removed,
    )


def _text_document(text: str, origin: str) -> SourceDocument:
    sanitized = redact_sensitive_text(text)
    neutralized, removed = neutralize_untrusted_text(sanitized)
    return SourceDocument(
        content=neutralized,
        content_format="text",
        origin=origin,
        untrusted_instructions_removed=removed,
    )


def _pod_projection(pod: Mapping[str, Any]) -> dict[str, Any]:
    """Project one kubectl pod object onto diagnosis-relevant fields."""

    metadata = _as_mapping(pod.get("metadata"))
    status = _as_mapping(pod.get("status"))
    projection: dict[str, Any] = {
        "name": metadata.get("name"),
        "phase": status.get("phase"),
    }

    container_statuses = status.get("containerStatuses")
    if isinstance(container_statuses, Sequence) and container_statuses:
        container = _as_mapping(container_statuses[0])
        projection["ready"] = bool(container.get("ready"))
        projection["restartCount"] = container.get("restartCount")
        state = _as_mapping(container.get("state"))
        for state_name in ("waiting", "terminated", "running"):
            detail = _as_mapping(state.get(state_name))
            if detail:
                projection["state"] = state_name
                if detail.get("reason"):
                    projection["reason"] = detail.get("reason")
                if detail.get("message"):
                    projection["message"] = str(detail.get("message"))[:500]
                break
        last_terminated = _as_mapping(
            _as_mapping(container.get("lastState")).get("terminated")
        )
        if last_terminated:
            projection["lastTerminated"] = {
                "reason": last_terminated.get("reason"),
                "exitCode": last_terminated.get("exitCode"),
            }

    conditions = status.get("conditions")
    if isinstance(conditions, Sequence):
        for condition in conditions:
            entry = _as_mapping(condition)
            if entry.get("type") == "Ready":
                projection.setdefault("ready", entry.get("status") == "True")
                if entry.get("reason"):
                    projection["readyReason"] = entry.get("reason")
                break
    return projection


def _deployment_projection(deployment: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _as_mapping(deployment.get("metadata"))
    spec = _as_mapping(deployment.get("spec"))
    status = _as_mapping(deployment.get("status"))
    return {
        "name": metadata.get("name"),
        "namespace": metadata.get("namespace"),
        "desiredReplicas": spec.get("replicas"),
        "readyReplicas": status.get("readyReplicas", 0),
        "availableReplicas": status.get("availableReplicas", 0),
        "unavailableReplicas": status.get("unavailableReplicas", 0),
    }


def _event_projection(event: Mapping[str, Any]) -> dict[str, Any]:
    involved = _as_mapping(event.get("involvedObject"))
    return {
        "type": event.get("type"),
        "reason": event.get("reason"),
        "message": str(event.get("message", ""))[:1_000],
        "count": event.get("count"),
        "firstTimestamp": event.get("firstTimestamp"),
        "lastTimestamp": event.get("lastTimestamp"),
        "involvedObject": {
            "kind": involved.get("kind"),
            "name": involved.get("name"),
            "namespace": involved.get("namespace"),
        },
    }


def _in_namespace(item: Mapping[str, Any], namespace: str) -> bool:
    metadata = _as_mapping(item.get("metadata"))
    found = metadata.get("namespace")
    return found is None or found == namespace


def _build_status_document(
    pods: list[Mapping[str, Any]],
    deployments: list[Mapping[str, Any]],
    target: WorkloadTarget,
    origin: str,
) -> SourceDocument | None:
    matching_pods = [
        _pod_projection(pod)
        for pod in pods
        if _in_namespace(pod, target.resolved_namespace)
        and target.matches_name(_as_mapping(pod.get("metadata")).get("name"))
    ][:MAX_PODS]
    matching_deployments = [
        _deployment_projection(deployment)
        for deployment in deployments
        if _in_namespace(deployment, target.resolved_namespace)
        and target.matches_name(
            _as_mapping(deployment.get("metadata")).get("name")
        )
    ]
    if not matching_pods and not matching_deployments:
        return None
    payload: dict[str, Any] = {"pods": matching_pods}
    if matching_deployments:
        payload["deployment"] = matching_deployments[0]
    return _json_document(payload, origin)


def _build_events_document(
    events: list[Mapping[str, Any]],
    target: WorkloadTarget,
    origin: str,
) -> SourceDocument | None:
    matching = [
        event
        for event in events
        if _as_mapping(event.get("involvedObject")).get("namespace")
        in (None, target.resolved_namespace)
        and target.matches_name(
            _as_mapping(event.get("involvedObject")).get("name")
        )
    ]
    if not matching:
        return None
    matching.sort(key=lambda event: str(event.get("lastTimestamp") or ""))
    projected = [_event_projection(event) for event in matching[-MAX_EVENTS:]]
    return _json_document({"items": projected}, origin)


def _bounded_log_text(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-MAX_LOG_LINES:])


class ClusterSnapshotSource:
    """Read kubectl JSON output from a directory on disk.

    The format is what `kubectl ... -o json` produces, so a directory may hold
    genuine captured output or equivalent fixtures; this class does not and
    cannot distinguish between them.
    """

    origin_label = "snapshot"

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory).expanduser().resolve()

    @property
    def directory(self) -> Path:
        return self._directory

    def _read_json(self, kind: str) -> object | None:
        if not self._directory.is_dir():
            raise ClusterAccessError(
                "The configured cluster snapshot directory does not exist."
            )
        for filename in SNAPSHOT_FILES[kind]:
            candidate = self._directory / filename
            if not candidate.is_file():
                continue
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError) as error:
                raise ClusterAccessError(
                    f"Snapshot file {filename} could not be read.",
                    retryable=True,
                ) from error
            except json.JSONDecodeError as error:
                raise ClusterAccessError(
                    f"Snapshot file {filename} is not valid JSON."
                ) from error
        return None

    def _origin(self, filename: str) -> str:
        return f"snapshot:{self._directory.name}/{filename}"

    def workload_status(self, target: WorkloadTarget) -> SourceDocument | None:
        pods = _items(self._read_json("pods"))
        deployments = _items(self._read_json("deployments"))
        if not pods and not deployments:
            return None
        return _build_status_document(
            pods, deployments, target, self._origin("pods.json")
        )

    def events(self, target: WorkloadTarget) -> SourceDocument | None:
        events = _items(self._read_json("events"))
        if not events:
            return None
        return _build_events_document(events, target, self._origin("events.json"))

    def container_logs(self, target: WorkloadTarget) -> SourceDocument | None:
        logs_directory = self._directory / "logs"
        if not logs_directory.is_dir():
            return None
        for candidate in sorted(logs_directory.glob("*.log")):
            if not target.matches_name(candidate.stem):
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError as error:
                raise ClusterAccessError(
                    "A snapshot log file could not be read.", retryable=True
                ) from error
            return _text_document(
                _bounded_log_text(text), self._origin(f"logs/{candidate.name}")
            )
        return None

    def manifest(self, target: WorkloadTarget) -> SourceDocument | None:
        for deployment in _items(self._read_json("deployments")):
            metadata = _as_mapping(deployment.get("metadata"))
            if not _in_namespace(
                deployment, target.resolved_namespace
            ) or not target.matches_name(metadata.get("name")):
                continue
            return _json_document(
                _manifest_projection(deployment),
                self._origin("deployments.json"),
            )
        return None


def _manifest_projection(deployment: Mapping[str, Any]) -> dict[str, Any]:
    """Project a real Deployment onto the same fields a pasted manifest uses."""

    from tools.manifest import select_diagnostic_config

    return select_diagnostic_config(deployment)


class LiveClusterSource:
    """Issue read-only Kubernetes API calls against a reachable cluster.

    Only read verbs are used: listing pods and events, reading one pod log,
    and reading one Deployment. The module imports no write operation.
    """

    origin_label = "live"

    def __init__(self, context: str | None = None) -> None:
        try:
            from kubernetes import client, config
        except ModuleNotFoundError as error:  # pragma: no cover - optional dep
            raise ClusterAccessError(
                "The kubernetes client library is not installed."
            ) from error

        try:
            config.load_kube_config(context=context)
        except Exception:  # noqa: BLE001 - fall back to in-cluster config
            try:
                config.load_incluster_config()
            except Exception as error:  # noqa: BLE001
                raise ClusterAccessError(
                    "No reachable Kubernetes configuration was found."
                ) from error

        self._core = client.CoreV1Api()
        self._apps = client.AppsV1Api()
        self._serialize = client.ApiClient().sanitize_for_serialization

    def _call(self, operation: str, function: Any, /, **kwargs: Any) -> Any:
        try:
            return function(**kwargs)
        except Exception as error:  # noqa: BLE001 - client raises many types
            status = getattr(error, "status", None)
            if status == 404:
                return None
            raise ClusterAccessError(
                f"The cluster could not complete {operation}.",
                retryable=status in (None, 429, 500, 502, 503, 504),
            ) from error

    def workload_status(self, target: WorkloadTarget) -> SourceDocument | None:
        namespace = target.resolved_namespace
        pod_list = self._call(
            "a pod listing",
            self._core.list_namespaced_pod,
            namespace=namespace,
        )
        pods = _items(self._serialize(pod_list)) if pod_list is not None else []
        deployments: list[Mapping[str, Any]] = []
        if target.workload_name:
            deployment = self._call(
                "a deployment read",
                self._apps.read_namespaced_deployment,
                name=target.workload_name,
                namespace=namespace,
            )
            if deployment is not None:
                deployments = [_as_mapping(self._serialize(deployment))]
        if not pods and not deployments:
            return None
        return _build_status_document(
            pods, deployments, target, f"live:{namespace}/pods"
        )

    def events(self, target: WorkloadTarget) -> SourceDocument | None:
        namespace = target.resolved_namespace
        event_list = self._call(
            "an event listing",
            self._core.list_namespaced_event,
            namespace=namespace,
        )
        if event_list is None:
            return None
        events = _items(self._serialize(event_list))
        if not events:
            return None
        return _build_events_document(events, target, f"live:{namespace}/events")

    def container_logs(self, target: WorkloadTarget) -> SourceDocument | None:
        namespace = target.resolved_namespace
        pod_list = self._call(
            "a pod listing",
            self._core.list_namespaced_pod,
            namespace=namespace,
        )
        if pod_list is None:
            return None
        for pod in _items(self._serialize(pod_list)):
            name = _as_mapping(pod.get("metadata")).get("name")
            if not target.matches_name(name):
                continue
            text = self._call(
                "a pod log read",
                self._core.read_namespaced_pod_log,
                name=name,
                namespace=namespace,
                tail_lines=MAX_LOG_LINES,
            )
            if not isinstance(text, str) or not text.strip():
                continue
            return _text_document(
                _bounded_log_text(text), f"live:{namespace}/{name}/log"
            )
        return None

    def manifest(self, target: WorkloadTarget) -> SourceDocument | None:
        if not target.workload_name:
            return None
        namespace = target.resolved_namespace
        deployment = self._call(
            "a deployment read",
            self._apps.read_namespaced_deployment,
            name=target.workload_name,
            namespace=namespace,
        )
        if deployment is None:
            return None
        return _json_document(
            _manifest_projection(_as_mapping(self._serialize(deployment))),
            f"live:{namespace}/{target.workload_name}",
        )


_SOURCE_CACHE: dict[tuple[str, str, str], ClusterSource | None] = {}


def _build_cluster_source(
    settings: Mapping[str, str],
) -> ClusterSource | None:
    if settings.get(LIVE_ENABLED_VARIABLE, "").strip().casefold() in {
        "1",
        "true",
        "yes",
    }:
        return LiveClusterSource(
            context=settings.get(LIVE_CONTEXT_VARIABLE, "").strip() or None
        )
    snapshot_directory = settings.get(SNAPSHOT_DIR_VARIABLE, "").strip()
    if snapshot_directory:
        return ClusterSnapshotSource(snapshot_directory)
    return None


def resolve_cluster_source(
    environment: Mapping[str, str] | None = None,
) -> ClusterSource | None:
    """Select the operator-configured source, or None when none is set.

    A successfully built source is cached because creating a live client
    loads configuration and opens connections. Failures are never cached, so
    a transient configuration problem can be retried.
    """

    settings = environment if environment is not None else os.environ
    key = (
        settings.get(LIVE_ENABLED_VARIABLE, ""),
        settings.get(LIVE_CONTEXT_VARIABLE, ""),
        settings.get(SNAPSHOT_DIR_VARIABLE, ""),
    )
    if key in _SOURCE_CACHE:
        return _SOURCE_CACHE[key]
    source = _build_cluster_source(settings)
    _SOURCE_CACHE[key] = source
    return source


def reset_cluster_source_cache() -> None:
    """Forget cached sources so configuration changes take effect."""

    _SOURCE_CACHE.clear()


__all__ = [
    "ClusterAccessError",
    "ClusterSnapshotSource",
    "ClusterSource",
    "LiveClusterSource",
    "LIVE_CONTEXT_VARIABLE",
    "LIVE_ENABLED_VARIABLE",
    "MAX_EVENTS",
    "MAX_LOG_LINES",
    "MAX_PODS",
    "SNAPSHOT_DIR_VARIABLE",
    "SourceDocument",
    "WorkloadTarget",
    "reset_cluster_source_cache",
    "resolve_cluster_source",
]
