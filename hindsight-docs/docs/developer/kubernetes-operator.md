# Kubernetes Operator

The Hindsight operator lets GitOps tools such as Argo CD and Flux manage bank configuration. A `HindsightBank` resource holds a [bank template](./api/bank-templates) manifest. The operator imports the manifest when the resource changes and again every 10 minutes, which restores settings that were changed outside Kubernetes.

Repeating an import is cheap: import skips mental models and directives that have not changed, so their content is not regenerated.

The operator manages configuration only. Deleting a `HindsightBank` keeps the bank and its memories.

## Install

The operator is in [`hindsight-operator/`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-operator). Build the image and push it to your registry, then install the CRD, RBAC, and Deployment:

```bash
docker build -t <registry>/hindsight-operator:<tag> hindsight-operator
docker push <registry>/hindsight-operator:<tag>
```

Reference `hindsight-operator/config/default` from a kustomization that sets your image, then apply it with `kubectl apply -k`:

```yaml
resources:
  - https://github.com/vectorize-io/hindsight//hindsight-operator/config/default
images:
  - name: hindsight-operator
    newName: <registry>/hindsight-operator
    newTag: <tag>
```

## Declare a bank

[Export](./api/bank-templates#export) an existing bank to start from its current setup, and put the manifest under `spec.template`:

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
  template:
    version: "1"
    bank:
      retain_mission: Extract customer issues, resolutions, and sentiment.
    mental_models:
      - id: sentiment-overview
        name: Customer Sentiment Overview
        source_query: What is the overall sentiment trend?
```

Import semantics apply: settings, mental models, and directives that the template omits are left as they are, and a missing bank is created.

The `Ready` condition is `True` when the last import succeeded. When it failed, the condition message has the error, for example the API's validation message for a rejected template.
