# Sample cluster snapshot

**This is synthetic data, not a real capture.** It exists so the snapshot
evidence path can be demonstrated on a machine with no cluster and no
`kubectl`. Every name, registry, and image tag is fictional.

The files are shaped exactly like real command output, so the same code path
runs against this directory and against a genuine capture:

| File | Real equivalent |
|---|---|
| `pods.json` | `kubectl get pods -n checkout -o json` |
| `events.json` | `kubectl get events -n checkout -o json` |
| `deployments.json` | `kubectl get deployments -n checkout -o json` |
| `logs/<pod>.log` | `kubectl logs -n checkout <pod> --tail=200` |

## Using it

~~~dotenv
K8S_SNAPSHOT_DIR=./data/sample_snapshot
~~~

Then ask about the failing workload, leaving every evidence field blank:

> The checkout-api deployment is stuck in ImagePullBackOff after last night's
> release.

Set the workload name to `checkout-api` and the namespace to `checkout`. The
agent retrieves status, events, and the workload definition from these files
itself; nothing is pasted in. The scenario is an image tag that was never
pushed, so the events carry `manifest unknown` and no container ever starts,
which is why the log file records that logs are unavailable.

The deliberately unrelated `checkout-worker` pod and its event are included so
the workload and namespace filtering has something to exclude.

To capture a real snapshot instead, see "Capturing a cluster snapshot" in the
project README. Real snapshots contain real cluster data: `snapshots/` is
gitignored for that reason, and this sample lives under `data/` only because it
is fictional and safe to commit.
