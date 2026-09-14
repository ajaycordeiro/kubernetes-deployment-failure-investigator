# Kubernetes Deployment Failure Runbook

This runbook supports read-only investigation. It provides interpretation guidance and safe manual checks. It does not authorize the agent to execute commands or change Kubernetes resources.

## ImagePullBackOff and ErrImagePull

### Recognizable symptoms

The Deployment has unavailable replicas and one or more containers are waiting with **ErrImagePull** or **ImagePullBackOff**. Logs are commonly unavailable because the container image never started.

### Evidence to collect

Inspect workload status for the exact waiting reason, Kubernetes events for the image-pull error, and the manifest for the complete image registry, repository, and tag. Events usually distinguish an image-name problem from authentication, networking, or registry availability.

### Interpretation

Messages such as **manifest unknown**, **not found**, or **failed to resolve reference** usually indicate that the requested repository or tag does not exist. Confirm that the event names the same image declared in the manifest. Do not diagnose a bad tag from ImagePullBackOff alone.

Messages such as **unauthorized**, **authentication required**, **pull access denied**, or **no basic auth credentials** indicate a registry-access problem rather than a nonexistent tag. Review the registry host and image-pull-secret configuration.

### Safe manual action

An engineer should verify that the declared image and tag exist in the intended registry and that the normal reviewed deployment configuration references the correct artifact. Any manifest change must follow the team's standard review and deployment process.

### Escalate when

Escalate if the event message is unavailable, if image existence cannot be verified, or if the evidence does not distinguish authentication, network, and image-name causes.

## Private Registry Authentication

### Recognizable symptoms

The pod is in **ErrImagePull** or **ImagePullBackOff**, and events mention authorization or authentication. The image points to a private registry.

### Evidence to collect

Inspect the exact event message, the registry hostname in the manifest, and the names of configured **imagePullSecrets**. Secret names may be inspected, but Secret values must never be collected or displayed.

### Interpretation

An empty or absent imagePullSecrets list combined with an authorization event supports a missing credential reference. If a secret name is present, the evidence may instead indicate an expired credential, incorrect scope, wrong namespace, or registry-side permissions. Those alternatives normally require engineer verification.

### Safe manual action

An engineer should verify that an approved registry credential exists in the workload namespace, is referenced by the pod specification or service account, and has permission to pull the requested repository.

### Escalate when

Escalate if the diagnosis would require inspecting credential contents, if a referenced credential exists but authorization still fails, or if registry availability and network evidence are missing.

## CrashLoopBackOff

### Recognizable symptoms

The container starts, exits, and is restarted repeatedly. Workload status shows **CrashLoopBackOff**, a growing restart count, and usually a previous termination exit code.

### Evidence to collect

Container logs are the primary source. Correlate the startup error with restart information and, when relevant, manifest command, arguments, environment references, or ports. Kubernetes BackOff events confirm repeated restarts but rarely identify the application cause by themselves.

### Interpretation

A specific exception, missing runtime dependency, invalid argument, or configuration parsing error can be decisive when it occurs immediately before process exit. Generic lines such as **startup failed** are insufficient without supporting detail. Do not infer a Kubernetes platform failure merely from the CrashLoopBackOff state.

### Safe manual action

An engineer should correct the application image or reviewed configuration that caused the startup process to exit, validate the fix outside production, and deploy it through the normal release process.

### Escalate when

Escalate if logs are truncated or unavailable, several independent startup causes remain plausible, or the error belongs to an external dependency that the corpus cannot inspect.

## CreateContainerConfigError: ConfigMap

### Recognizable symptoms

The pod was scheduled but the container cannot be created. Status may show **CreateContainerConfigError**, while logs are unavailable because the application process never started.

### Evidence to collect

Inspect events for an explicit missing ConfigMap name and compare it with **envFrom**, **env**, or volume references in the manifest. The resource name and namespace context must match.

### Interpretation

An event such as **configmap "name" not found** plus a matching manifest reference is decisive evidence that the workload configuration depends on an unavailable ConfigMap. A ConfigMap reference alone does not prove the object is missing.

### Safe manual action

An engineer should verify that the approved ConfigMap exists in the workload namespace and that the manifest references the intended name. Creation or modification must follow the normal reviewed change process.

### Escalate when

Escalate if the event does not identify the missing object, the expected configuration owner is unknown, or several ConfigMap references could be responsible.

## CreateContainerConfigError: Secret

### Recognizable symptoms

The pod is unable to create a container and events identify a missing Secret. Logs are normally unavailable.

### Evidence to collect

Inspect only the Secret name referenced by the manifest and the corresponding event. Never request or reveal Secret contents.

### Interpretation

An event such as **secret "name" not found** plus the same manifest reference is sufficient to identify the missing dependency. If the Secret exists but a key is missing, the exact event may identify the key; do not attempt to recover its value.

### Safe manual action

An engineer should verify that the correct approved Secret exists in the workload namespace and that the manifest references the correct Secret and key names.

### Escalate when

Escalate when the investigation would require Secret values, credential rotation, or access beyond resource-name metadata.

## Readiness Probe Failures

### Recognizable symptoms

The pod phase is Running, but the container remains not ready and the Deployment has no available replicas. Events repeatedly report **Readiness probe failed**.

### Evidence to collect

Collect the probe failure message, readiness-probe path and port from the manifest, and bounded application startup logs. Compare the configured endpoint with the endpoint the application actually exposes.

### Interpretation

An HTTP 404 indicates that the probe reached the container but requested a path the application did not serve. Connection refused may instead mean the process is not listening on the configured port or was not ready when probed. Timeouts can result from slow startup, overloaded code, or network behavior and require more caution.

### Safe manual action

An engineer should align the readiness probe with the application's documented health endpoint and port, test the change, and submit it through the normal review process.

### Escalate when

Escalate if application logs do not identify the served endpoint, if failures are intermittent, or if the evidence points to application latency rather than a simple configuration mismatch.

## Pending Pods and Failed Scheduling

### Recognizable symptoms

The pod remains Pending, has not been assigned to a node, and has no container logs. Events commonly show **FailedScheduling**.

### Evidence to collect

Inspect the scheduler event and the manifest's CPU and memory requests. The scheduler message is the primary runtime evidence; resource requests provide matching workload context.

### Interpretation

Messages such as **Insufficient cpu** or **Insufficient memory** indicate that no eligible node currently has enough allocatable capacity for the request. Other messages may point to node selectors, taints, topology constraints, or quota and should not be mislabeled as simple resource shortage.

### Safe manual action

An engineer should review whether the resource request is justified and whether approved cluster capacity is available. Changes to requests, quotas, or node capacity require normal operational review.

### Escalate when

Escalate if the scheduler reports several distinct constraints, if capacity data is missing, or if resolving the problem requires changing cluster policy.

## PVC Binding and Mount Failures

### Recognizable symptoms

The pod remains Pending or ContainerCreating and events contain **FailedMount**, **FailedAttachVolume**, or a persistent-volume-claim error. Application logs are often unavailable.

### Evidence to collect

Inspect the exact event, the claim name in the pod manifest, and namespace context. Distinguish a missing claim from an unbound claim, attach failure, mount permission issue, and storage-provider outage.

### Interpretation

An event stating that a named PersistentVolumeClaim was not found, combined with the same claim name in the manifest, is decisive. A Pending claim or provider timeout represents a different failure and may require storage-team involvement.

### Safe manual action

An engineer should verify that the intended claim exists in the workload namespace, is bound when required, and matches the reviewed workload configuration. The agent must not create, delete, resize, or rebind storage.

### Escalate when

Escalate for storage-provider errors, attachment conflicts, access-mode questions, data-safety concerns, or any remediation that could affect persistent data.

## Missing, Conflicting, or Unavailable Evidence

### Principle

The absence of evidence is not evidence for a preferred hypothesis. A tool failure should be recorded with its retryability, and already collected evidence must remain in state.

### Recovery

Retry a transient timeout once. If the retry fails, choose an unused evidence source only when it can answer the unresolved question. Treat expected conditions, such as unavailable logs for a container that never started, as diagnostic context rather than retryable system failures.

### Human handoff

Hand off when a decisive source remains unavailable, evidence materially conflicts, the failure type is unsupported, or the investigation reaches its tool budget. The handoff must list the symptom, evidence, tools attempted, failures, leading hypotheses, unresolved question, and safest manual check.

### Safety

Never invent missing events, logs, resource state, or Secret values. Never execute remediation. All proposed changes remain advisory and require an engineer's normal review process.

