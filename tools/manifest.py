"""Safe projection helpers for submitted Kubernetes manifests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

def _copy_scalars(
    source: Mapping[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    return {
        field: source[field]
        for field in fields
        if isinstance(source.get(field), (str, int, float, bool))
    }


def _sanitize_reference(
    value: object, fields: tuple[str, ...]
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result = _copy_scalars(value, fields)
    return result or None


def _sanitize_env(entries: object) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        return []
    safe_entries: list[dict[str, Any]] = []
    for entry in entries[:50]:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            continue
        safe_entry: dict[str, Any] = {"name": entry["name"]}
        value_from = entry.get("valueFrom")
        if isinstance(value_from, Mapping):
            safe_value_from: dict[str, Any] = {}
            for reference_name in ("configMapKeyRef", "secretKeyRef"):
                reference = _sanitize_reference(
                    value_from.get(reference_name), ("name", "key", "optional")
                )
                if reference:
                    safe_value_from[reference_name] = reference
            if safe_value_from:
                safe_entry["valueFrom"] = safe_value_from
        safe_entries.append(safe_entry)
    return safe_entries


def _sanitize_env_from(entries: object) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        return []
    safe_entries: list[dict[str, Any]] = []
    for entry in entries[:50]:
        if not isinstance(entry, Mapping):
            continue
        safe_entry: dict[str, Any] = {}
        for reference_name in ("configMapRef", "secretRef"):
            reference = _sanitize_reference(
                entry.get(reference_name), ("name", "optional")
            )
            if reference:
                safe_entry[reference_name] = reference
        if safe_entry:
            safe_entries.append(safe_entry)
    return safe_entries


def _sanitize_resources(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for category in ("requests", "limits"):
        resources = value.get(category)
        if isinstance(resources, Mapping):
            selected = _copy_scalars(
                resources, ("cpu", "memory", "ephemeral-storage")
            )
            if selected:
                result[category] = selected
    return result


def _sanitize_probe(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = _copy_scalars(
        value,
        (
            "initialDelaySeconds",
            "periodSeconds",
            "timeoutSeconds",
            "successThreshold",
            "failureThreshold",
        ),
    )
    for handler_name, fields in (
        ("httpGet", ("path", "port", "scheme")),
        ("tcpSocket", ("host", "port")),
        ("grpc", ("port", "service")),
    ):
        handler = _sanitize_reference(value.get(handler_name), fields)
        if handler:
            result[handler_name] = handler
    exec_handler = value.get("exec")
    if isinstance(exec_handler, Mapping) and isinstance(
        exec_handler.get("command"), list
    ):
        result["exec"] = {
            "command": [
                str(part)[:500] for part in exec_handler["command"][:20]
            ]
        }
    return result


def _sanitize_container(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result = _copy_scalars(value, ("name", "image", "imagePullPolicy"))
    for field in ("command", "args"):
        parts = value.get(field)
        if isinstance(parts, list):
            result[field] = [str(part)[:500] for part in parts[:20]]

    ports = value.get("ports")
    if isinstance(ports, list):
        result["ports"] = [
            _copy_scalars(port, ("name", "containerPort", "protocol"))
            for port in ports[:20]
            if isinstance(port, Mapping)
        ]

    resources = _sanitize_resources(value.get("resources"))
    if resources:
        result["resources"] = resources
    env = _sanitize_env(value.get("env"))
    if env:
        result["env"] = env
    env_from = _sanitize_env_from(value.get("envFrom"))
    if env_from:
        result["envFrom"] = env_from

    volume_mounts = value.get("volumeMounts")
    if isinstance(volume_mounts, list):
        result["volumeMounts"] = [
            _copy_scalars(
                mount, ("name", "mountPath", "subPath", "readOnly")
            )
            for mount in volume_mounts[:20]
            if isinstance(mount, Mapping)
        ]

    for probe_name in ("readinessProbe", "livenessProbe", "startupProbe"):
        probe = _sanitize_probe(value.get(probe_name))
        if probe:
            result[probe_name] = probe
    return result or None


def _sanitize_volumes(entries: object) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        return []
    safe_volumes: list[dict[str, Any]] = []
    for entry in entries[:20]:
        if not isinstance(entry, Mapping):
            continue
        result = _copy_scalars(entry, ("name",))
        for reference_name, fields in (
            ("persistentVolumeClaim", ("claimName", "readOnly")),
            ("configMap", ("name", "optional")),
            ("secret", ("secretName", "optional")),
        ):
            reference = _sanitize_reference(entry.get(reference_name), fields)
            if reference:
                result[reference_name] = reference
        if result:
            safe_volumes.append(result)
    return safe_volumes


def select_diagnostic_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project a manifest onto diagnosis-relevant, non-secret fields."""

    result = _copy_scalars(manifest, ("apiVersion", "kind"))
    metadata = manifest.get("metadata")
    if isinstance(metadata, Mapping):
        result["metadata"] = _copy_scalars(metadata, ("name", "namespace"))

    raw_spec = manifest.get("spec")
    if not isinstance(raw_spec, Mapping):
        return result
    safe_spec = _copy_scalars(raw_spec, ("replicas",))
    template = raw_spec.get("template")
    if isinstance(template, Mapping):
        raw_pod_spec = template.get("spec")
        if isinstance(raw_pod_spec, Mapping):
            safe_pod_spec: dict[str, Any] = {}
            containers = raw_pod_spec.get("containers")
            if isinstance(containers, list):
                safe_pod_spec["containers"] = [
                    safe_container
                    for container in containers[:20]
                    if (safe_container := _sanitize_container(container))
                ]
            image_pull_secrets = raw_pod_spec.get("imagePullSecrets")
            if isinstance(image_pull_secrets, list):
                safe_pod_spec["imagePullSecrets"] = [
                    reference
                    for item in image_pull_secrets[:20]
                    if (reference := _sanitize_reference(item, ("name",)))
                ]
            volumes = _sanitize_volumes(raw_pod_spec.get("volumes"))
            if volumes:
                safe_pod_spec["volumes"] = volumes
            safe_spec["template"] = {"spec": safe_pod_spec}
    result["spec"] = safe_spec
    return result


__all__ = ["select_diagnostic_config"]
