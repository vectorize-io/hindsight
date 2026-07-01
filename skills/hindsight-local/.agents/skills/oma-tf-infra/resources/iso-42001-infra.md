# TF Infra Agent - ISO/IEC 42001 / ISO 22301 / ISO/IEC/IEEE 42010 Guide

Use this file when the task involves AI systems and the user wants infrastructure controls or audit-friendly recommendations aligned with ISO/IEC 42001, continuity and recovery guidance aligned with ISO 22301, or infrastructure architecture documentation guidance aligned with ISO/IEC/IEEE 42010.

## Scope

This guide is for:

- **infrastructure controls** that support AI governance
- **continuity and recovery controls** for infrastructure resilience
- **architecture description guidance** for infrastructure decisions

It does **not** replace:

- organizational policy
- model governance
- human oversight design
- AI risk management process ownership

Those belong primarily to PM, QA, backend, or security governance work.

## What TF Infra Agent Should Cover

For AI systems, focus on infrastructure controls such as:

- IAM and least privilege
- service-to-service identity
- network segmentation and private connectivity
- encryption at rest and in transit
- secrets management
- centralized logging and audit trails
- monitoring and alerting
- backup and retention
- environment isolation
- change traceability in CI/CD and Terraform state
- continuity and failover design
- architecture views, dependencies, and decision rationale

## Recommended Control Areas

### 1. Access Control

Implement:

- least-privilege IAM roles
- workload identity / OIDC for CI/CD
- separation of admin, deployer, runtime, and read-only roles
- scoped access to model endpoints, vector stores, object stores, and databases

Avoid:

- shared admin roles
- long-lived keys
- wildcard permissions to AI-related data or services

### 2. Network Boundaries

Implement:

- private subnets for sensitive workloads
- VPC/VNet/VCN peering or private service endpoints
- restricted egress where feasible
- ingress limited to approved paths such as ALB, API Gateway, or internal load balancers

Use stronger isolation when:

- prompts or training data contain sensitive data
- model outputs affect business-critical actions
- cross-region or cross-tenant exposure is a concern

### 3. Encryption and Secrets

Implement:

- provider-managed KMS or customer-managed keys where required
- encrypted storage for object stores, DBs, queues, and state backends
- TLS for internal and external traffic where supported
- secrets in cloud secret managers, not in Terraform variables committed to git

### 4. Logging and Auditability

Implement:

- centralized audit logs for IAM, network, storage, and runtime services
- immutable or retention-locked log targets where needed
- request tracing across gateways, application services, and data stores
- Terraform plan/apply evidence in CI/CD logs

For AI-related systems, especially capture evidence around:

- who changed infrastructure
- who can access model-serving paths
- where prompts, embeddings, or outputs are stored
- whether logging retention matches policy

### 5. Monitoring and Alerting

Implement:

- uptime and health monitoring
- resource utilization metrics
- security event alerting
- data egress anomaly alerting where available
- alerting for failed backups, replication, or logging pipelines

If the AI system is production-critical, recommend:

- SLO-aligned alerting
- model endpoint availability monitoring
- queue/backlog monitoring for ingestion pipelines
- vector or search cluster health checks if present

### 6. Retention, Backup, and Recovery

Implement:

- backup policies for stateful systems
- snapshot retention and lifecycle rules
- archival tiers for logs and historical data
- restore procedures and periodic recovery validation

For AI systems, think about:

- prompt / output retention requirements
- embedding index rebuild strategy
- backup of canonical data stores rather than only derived indexes

### 7. Environment and Tenant Isolation

Implement:

- separate environments by account/project/subscription/compartment when practical
- isolated state backends per environment
- tagging/labeling for ownership and cost attribution
- tenant-aware segmentation when AI data is multi-tenant

## 8. ISO 22301 Continuity View

For infrastructure continuity, recommend improvements around:

- critical service identification
- RTO / RPO assumptions
- backup scope and retention
- restore verification
- redundancy and failover
- cross-region or cross-zone strategy
- dependency mapping
- manual fallback and degraded-mode assumptions

Examples of infra-focused suggestions:

- multi-AZ or multi-zone deployment for critical paths
- warm standby or replicated storage for essential services
- tested snapshot restore workflow
- documented dependency chain for identity, DNS, storage, and network services
- alerting for failed replication or backup jobs

## 9. ISO/IEC/IEEE 42010 Architecture View

Use 42010 as an architecture description guide, not as a control checklist.

Capture:

- stakeholders
- concerns
- architecture views
- interfaces and dependencies
- constraints
- rationale and tradeoffs

For infrastructure work, useful views include:

- context view
- deployment / runtime view
- network boundary view
- identity and trust view
- resilience / recovery view
- observability view

Useful documentation prompts:

- Who needs this architecture description?
- What concerns are they trying to answer?
- Which infrastructure elements and interfaces matter?
- What tradeoffs were chosen and why?
- What dependencies create operational risk?

## Review Questions

When applying ISO/IEC 42001 / ISO 22301 / ISO/IEC/IEEE 42010 thinking, ask:

- Can access to AI-related data paths be audited?
- Are model-serving and data-serving paths protected with least privilege?
- Are logs centralized, retained, and protected against tampering?
- Are backups, retention, and restore procedures defined?
- Are continuity objectives such as RTO / RPO explicit?
- Are failover paths and critical dependencies visible?
- Is network exposure minimized?
- Is CI/CD access ephemeral and traceable?
- Is there clear ownership for each critical resource?
- Are the stakeholders, concerns, and architecture rationale documented clearly?

## Output Pattern

When relevant, add a short section like:

```md
## ISO/IEC 42001 / ISO 22301 / 42010 Notes
- Access control:
- Network boundary:
- Encryption:
- Logging / auditability:
- Monitoring / alerting:
- Backup / retention:
- Continuity / RTO / RPO:
- Failover / dependency view:
- Stakeholders / concerns / views:
- Gaps:
- Recommended Terraform changes:
```

## Guardrails

- Do not claim ISO/IEC 42001 compliance from Terraform alone
- Do not claim ISO 22301 compliance from backup settings alone
- Do not expand this skill into full AI governance process design
- Do not turn 42010 into paperwork without architectural value
- Keep suggestions concrete and infra-implementable
- If the gap is organizational rather than infrastructural, say so explicitly
