# Kubernetes Operator

The Hindsight operator manages memory banks as Kubernetes resources. You declare a bank's configuration, mental models, and directives in a `HindsightBank` resource, and the operator keeps the bank in Hindsight matching it. This fits GitOps tools such as Argo CD and Flux: the bank definition lives in Git next to the rest of your deployment.

A `HindsightBank` holds a [bank template](./api/bank-templates) manifest, the same format that the export endpoint returns and the import endpoint accepts. The operator does not define its own copy of the bank settings, so settings added in later Hindsight versions work without an operator update.

The operator manages configuration only. It never reads, writes, or migrates memories.

## Install

The operator runs anywhere that can reach the Hindsight API. Install the CRD, RBAC, and Deployment from the repository:

```bash
kubectl apply -k https://github.com/vectorize-io/hindsight//hindsight-operator/config/default
```

The image is `ghcr.io/vectorize-io/hindsight-operator`. It is published with each Hindsight release, starting with the first release that includes the operator. Install from the matching release tag, for example `?ref=v0.11.0`, so the manifests and the image version agree.

The operator can read Secrets in any namespace so that each `HindsightBank` can reference an API key next to it. It reads them directly and does not cache them. To limit it to one namespace, add `--watch-namespace=<namespace>` to the container arguments and replace the `hindsight-operator` ClusterRoleBinding with a RoleBinding in that namespace. Leader election uses a separate Role in the `hindsight-operator` namespace, so it keeps working.

## Declare a bank

[Export](./api/bank-templates#export) an existing bank to start from its current setup.

Put the manifest under `spec.template`:

```yaml
apiVersion: hindsight.vectorize.io/v1alpha1
kind: HindsightBank
metadata:
  name: support-agent
spec:
  bankId: support-agent        # defaults to metadata.name; cannot be changed
  connection:
    url: http://hindsight-api.hindsight.svc.cluster.local:8888
    apiKeySecretRef:           # omit when the API does not require a key
      name: hindsight-api
      key: token
  deletionPolicy: Retain       # Retain (default) or Delete
  template:
    version: "1"
    bank:
      retain_mission: Extract customer issues, resolutions, and sentiment.
      reflect_mission: You are helping a support agent remember customer interactions.
    mental_models:
      - id: sentiment-overview
        name: Customer Sentiment Overview
        source_query: What is the overall sentiment trend?
        trigger:
          refresh_after_consolidation: true
    directives:
      - name: Always be empathetic
        content: Always respond with empathy and understanding.
        priority: 10
```

```bash
kubectl get hindsightbanks
NAME            BANK            READY   REASON   AGE
support-agent   support-agent   True    Synced   2m
```

## How reconciliation works

On each reconcile, the operator exports the bank, compares it with `spec.template`, and imports only the items that differ.

- **Only declared items are managed.** Bank settings, mental models, and directives that the template omits are left as they are, which matches import behavior. Removing an item from the template does not delete it from the bank.
- **Unchanged mental models are not regenerated.** Import regenerates the content of every mental model it receives. Because the operator sends only changed models, a resync or an unrelated change does not spend LLM calls on regeneration.
- **Changes made outside Kubernetes are corrected.** The operator compares the bank again every 10 minutes (`--resync-period`) and after every change to the resource. A setting edited in the Control Plane UI is set back to the declared value.
- **A missing bank is created** from the full template.

After an import, the operator exports the bank again to confirm that it matches. If it still differs, the `Ready` condition becomes `False` with reason `NotConverged`, and the operator stops importing until the resource changes. This prevents a template that the server stores differently, for example a value outside an allowed range, from regenerating mental models on every resync.

## Deleting a resource

`deletionPolicy` decides what happens to the bank when you delete the `HindsightBank`:

| Policy | Result |
| --- | --- |
| `Retain` (default) | The bank and all of its memories stay in Hindsight. |
| `Delete` | The bank and all of its memories are deleted, but only if this resource created the bank on the server that `spec.connection.url` points at (`status.bankCreated` and `status.createdOnURL`). |

A resource that adopted an existing bank never deletes it, even with `Delete`. This keeps a mistaken or copied resource from destroying memories it did not create. Changing `spec.connection.url` to another server also makes the bank there count as adopted.

Deleting a bank needs the API and the API key Secret. If either is already gone, for example when a whole namespace is deleted at once, the resource stays in `Terminating` and the operator records a `DeleteFailed` event. Restore access, or set the `hindsight.vectorize.io/skip-bank-deletion: "true"` annotation to keep the bank and finish deleting the resource.

## Status

| Reason | Meaning |
| --- | --- |
| `Synced` | The bank matches the template. |
| `InvalidTemplate` | The template is not a JSON object, or Hindsight rejected it. The condition message has the API's validation error. |
| `SecretError` | The API key Secret or key is missing. |
| `APIError` | Hindsight could not be reached or returned an error. The operator retries with backoff. |
| `NotConverged` | The bank still differed from the template after an import. |

`status.lastAppliedChanges` lists what the last import changed, for example `bank.retain_mission` or `mental_models/sentiment-overview`. The operator also records a Kubernetes event for each import.
