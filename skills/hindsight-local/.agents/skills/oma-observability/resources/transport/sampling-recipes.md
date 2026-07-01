---
otel_spec: "1.x (stable API/SDK)"
---

# Sampling Recipes: Tail-Based, Cost-Aware, and Tenant-Aware


---

## 1. Sampling Taxonomy

| Type | Where Decision Is Made | Trace Completeness | Typical Use |
|------|----------------------|-------------------|-------------|
| **Head-based** | At trace start (first span) | May be incomplete; downstream services inherit decision but cannot guarantee it | Simple, low-overhead; works well for single-service or homogeneous traffic |
| **Tail-based** | After all spans arrive (end of trace) | Complete; sampler buffers spans until decision | Production multi-service; required for error/latency policies |
| **Adaptive** | Dynamic rate based on traffic volume | Depends on implementation | Auto-scaling sampling rate under variable load |

**Head-based limitation**: If a service at hop 2 is sampled out but hop 1 was sampled in (or vice versa), the reconstructed trace is missing spans. This is the core motivation for tail-based sampling in multi-service systems.

**Tail-based trade-off**: The sampler must buffer all spans for a trace until a decision is made (default `decision_wait: 30s`). Memory usage scales with trace volume and decision wait duration.

---

## 2. Tail-Based Recipe (Recommended for Production)

The production-standard recipe retains high-signal traces at 100% while keeping a low-rate baseline for ambient visibility.

**Policy hierarchy** (evaluated in order; first match wins unless using `and` policy):

1. **100% error retention**: any trace with an error span is always kept
2. **100% cost/latency threshold retention**: traces exceeding a cost or latency threshold are always kept
3. **5-10% baseline**: probabilistic retention of remaining traffic

This requires the `tail_sampling` processor running in the **gateway tier only**, combined with a `loadbalancing` exporter upstream to ensure trace completeness via consistent hash by `trace_id`. See `collector-topology.md` Section 6.

```yaml
processors:
  tail_sampling:
    decision_wait: 30s          # wait up to 30s for all spans to arrive
    num_traces: 100000          # in-memory trace buffer capacity
    expected_new_traces_per_sec: 1000
    policies:
      # Policy 1: Always keep traces with errors
      - name: keep-errors
        type: status_code
        status_code:
          status_codes: [ERROR]

      # Policy 2: Always keep high-latency traces (p99 threshold)
      - name: keep-slow-traces
        type: latency
        latency:
          threshold_ms: 2000    # keep traces with root span > 2s

      # Policy 3: Always keep high-cost LLM traces (see Section 3)
      - name: keep-high-cost
        type: string_attribute
        string_attribute:
          key: sampling.keep_reason
          values: ["high_cost"]

      # Policy 4: Baseline probabilistic sampling for remaining traces
      - name: baseline-sample
        type: probabilistic
        probabilistic:
          sampling_percentage: 8  # 5-10% baseline
```

> The `tail_sampling` processor evaluates policies top-to-bottom. A trace matching any policy is kept. Policies do not chain; `probabilistic` at the bottom catches everything not already retained.

---

## 3. Cost-Aware Sampling (LLM / FinOps Context)

LLM workloads (OpenAI, Anthropic, Bedrock, Vertex AI) attach cost attributes to spans. These should be treated as first-class sampling dimensions.

**Relevant span attributes** (following GenAI semantic conventions):
- `gen_ai.usage.input_tokens`: prompt token count
- `gen_ai.usage.output_tokens`: completion token count
- `llm.request.cost_usd`: estimated cost if pre-computed by SDK
- `gen_ai.cost.total_usd`: total cost attribute (custom, team-defined)

**Strategy**: Set a cost threshold (e.g., $0.50 per trace). Any trace with cumulative LLM cost above the threshold is always retained for cost attribution and FinOps analysis.

Because `tail_sampling` does not natively support numeric comparisons on span attributes, use a transform processor to annotate high-cost traces before sampling:

```yaml
processors:
  transform/cost_annotation:
    trace_statements:
      - context: span
        statements:
          # If cost attribute exceeds threshold, mark the trace for retention
          - set(attributes["sampling.keep_reason"], "high_cost")
              where attributes["llm.request.cost_usd"] > 0.50
          - set(attributes["sampling.keep_reason"], "high_cost")
              where attributes["gen_ai.cost.total_usd"] > 0.50

  tail_sampling:
    decision_wait: 30s
    policies:
      - name: keep-high-cost
        type: string_attribute
        string_attribute:
          key: sampling.keep_reason
          values: ["high_cost"]
      - name: keep-errors
        type: status_code
        status_code:
          status_codes: [ERROR]
      - name: baseline
        type: probabilistic
        probabilistic:
          sampling_percentage: 8

service:
  pipelines:
    traces:
      processors: [transform/cost_annotation, tail_sampling]
```

---

## 4. Tenant-Aware Sampling (Multi-Tenant B2B SaaS)

Different customer tiers have different observability value. Enterprise customers justify full retention; free tier customers do not.

**Per-tier retention targets**:

| Tenant Tier | Retention Rate | Rationale |
|------------|---------------|-----------|
| `enterprise` | 100% | SLA obligations, full debugging capability |
| `pro` | 20% | Representative sample, cost-controlled |
| `free` | 2% | Ambient visibility only |

**Option A: `routing_connector` for per-tenant pipeline branching**

The `routing_connector` routes telemetry to different pipelines based on attribute values, allowing different `tail_sampling` configurations per tenant tier.

> Alpha stability warning: `routing_connector` is in **alpha** as of 2025. The API and behavior may change in minor releases. Do not use in production if pipeline stability is required. Prefer Option B for production workloads.

```yaml
connectors:
  routing:
    default_pipelines: [traces/pro]  # fallback if no rule matches
    error_mode: ignore
    table:
      - statement: route() where attributes["tenant.tier"] == "enterprise"
        pipelines: [traces/enterprise]
      - statement: route() where attributes["tenant.tier"] == "free"
        pipelines: [traces/free]

service:
  pipelines:
    traces/input:
      receivers: [otlp]
      exporters: [routing]

    traces/enterprise:
      receivers: [routing]
      processors: [tail_sampling/enterprise]
      exporters: [otlp/backend]

    traces/pro:
      receivers: [routing]
      processors: [tail_sampling/pro]
      exporters: [otlp/backend]

    traces/free:
      receivers: [routing]
      processors: [tail_sampling/free]
      exporters: [otlp/backend]
```

**Option B: `tail_sampling` with per-tier policies (stable, recommended for production)**

Use composite `and` policies combining tenant tier with probabilistic sampling:

```yaml
processors:
  tail_sampling:
    decision_wait: 30s
    num_traces: 100000
    policies:
      # Enterprise: always keep
      - name: enterprise-full
        type: and
        and:
          and_sub_policy:
            - name: is-enterprise
              type: string_attribute
              string_attribute:
                key: tenant.tier
                values: ["enterprise"]
            - name: keep-all
              type: probabilistic
              probabilistic:
                sampling_percentage: 100

      # Pro: 20%
      - name: pro-sample
        type: and
        and:
          and_sub_policy:
            - name: is-pro
              type: string_attribute
              string_attribute:
                key: tenant.tier
                values: ["pro"]
            - name: pro-rate
              type: probabilistic
              probabilistic:
                sampling_percentage: 20

      # Free: 2%
      - name: free-sample
        type: probabilistic
        probabilistic:
          sampling_percentage: 2  # baseline for unmatched / free tier
```

---

## 5. Complete Example: Four-Policy `tail_sampling`

This YAML combines error, latency, cost, and baseline policies into a single production-ready configuration:

```yaml
processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 1500
    spike_limit_mib: 400

  transform/cost_annotation:
    trace_statements:
      - context: span
        statements:
          - set(attributes["sampling.keep_reason"], "high_cost")
              where attributes["llm.request.cost_usd"] > 0.50

  tail_sampling:
    decision_wait: 30s
    num_traces: 200000
    expected_new_traces_per_sec: 2000
    policies:
      # Policy 1: 100% error traces
      - name: policy-errors
        type: status_code
        status_code:
          status_codes: [ERROR]

      # Policy 2: 100% high-latency traces (>2s root span)
      - name: policy-latency
        type: latency
        latency:
          threshold_ms: 2000

      # Policy 3: 100% high-cost LLM traces (>$0.50)
      - name: policy-high-cost
        type: string_attribute
        string_attribute:
          key: sampling.keep_reason
          values: ["high_cost"]

      # Policy 4: 8% baseline probabilistic sampling
      - name: policy-baseline
        type: probabilistic
        probabilistic:
          sampling_percentage: 8

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, transform/cost_annotation, tail_sampling]
      exporters: [otlp/backend]
```

---

## 6. Pitfalls

| Pitfall | Description | Mitigation |
|---------|-------------|-----------|
| Head-based sampling on multi-service paths | A service sampled out at hop 2 drops spans; trace is reconstructed with gaps | Use tail-based sampling in gateway; propagate `traceparent` regardless of local sampling decision |
| Tail sampling memory exhaustion | Buffer holds all in-flight spans per trace for `decision_wait` duration; high RPS + long wait = large heap | Set `num_traces` based on `expected_new_traces_per_sec * decision_wait`; always include `memory_limiter` |
| Decision wait too short | Spans from slow downstream services arrive after decision is made; those spans are dropped | Set `decision_wait` to exceed your p99 inter-service latency; typically 30-60s |
| Sample-rate mismatch between gateway tiers | Tier-1 and Tier-2 each apply independent sampling; effective rate is multiplied (e.g., 50% × 20% = 10%) | Assign sampling responsibility to one tier only; the other tier passes all traffic through |
| `routing_connector` in production (alpha) | Alpha components may break on minor version upgrades of collector-contrib | Use `tail_sampling` with `and` sub-policies (stable) for production; evaluate `routing_connector` only in staging |
| Missing `loadbalancing` exporter before tail sampler | Spans for the same trace land on different gateway replicas; sampler on each replica sees an incomplete trace and makes wrong decisions | Always deploy `loadbalancing` exporter (routing_key: traceID) in the tier upstream of `tail_sampling` |

## References

- https://opentelemetry.io/docs/specs/otlp/
- https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/processor/tailsamplingprocessor
- https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/connector/routingconnector
