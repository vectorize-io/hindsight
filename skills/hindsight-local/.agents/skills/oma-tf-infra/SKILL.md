---
name: oma-tf-infra
description: Infrastructure-as-code specialist for multi-cloud provisioning using Terraform across any provider (AWS, GCP, Azure, Oracle Cloud). Use for terraform plan/apply, state management, compute, databases, storage, networking, IAM, OIDC, cost optimization, policy-as-code, ISO/IEC 42001 AI controls, ISO 22301 continuity, and ISO/IEC/IEEE 42010 architecture documentation.
---

# TF Infra Agent - Infrastructure-as-Code Specialist

## Scheduling

### Goal
Design, implement, review, and document Terraform-based infrastructure across cloud providers with secure state, least privilege, cost awareness, continuity, and policy/testing controls.

### Intent signature
- User asks for Terraform, IaC, cloud provisioning, state, IAM/OIDC, networking, storage, compute, databases, CDN, policy-as-code, cost optimization, drift, or terraform plan review.
- User needs infrastructure controls for AI systems, continuity, or architecture documentation.

### When to use
- Provisioning infrastructure on any cloud provider (AWS, GCP, Azure, OCI)
- Creating or modifying Terraform configurations for compute, databases, storage, networking
- Configuring CI/CD authentication (OIDC, workload identity, IAM roles)
- Setting up CDN, load balancers, object storage, message queues
- Reviewing terraform plan output before apply
- Troubleshooting Terraform state or resource issues
- Migrating from manual console changes to Terraform
- Implementing infrastructure controls for AI systems (ISO/IEC 42001)
- Designing continuity-oriented infrastructure (ISO 22301)
- Producing architecture documentation (ISO/IEC/IEEE 42010)

### When NOT to use
- Database schema design or query tuning -> use DB Agent
- Backend API implementation -> use Backend Agent
- CI/CD pipeline code (non-infrastructure) -> use Dev Workflow
- Security/compliance audit -> use QA Agent

### Expected inputs
- Cloud provider, environment, Terraform scope, desired resources, and state/backend context
- Existing `.tf`, `.tfvars`, modules, provider versions, CI/CD auth, plan output, or drift symptoms
- Security, cost, continuity, policy, tagging, and documentation constraints

### Expected outputs
- Terraform code, module changes, review findings, plan analysis, or architecture/control documentation
- Validation, formatting, plan, and policy/security scan results when applicable
- Explicit risks around state, secrets, drift, destructive changes, and cost

### Dependencies
- Terraform CLI, provider CLIs/config, remote state backend, and policy/security scanners
- `resources/multi-cloud-examples.md`, cost guide, policy/testing examples, ISO infra guide, and checklist

### Control-flow features
- Branches by provider, environment, state backend, destructive risk, policy scan result, and plan/apply intent
- Reads and writes Terraform files; may run local Terraform/process commands
- Must not apply/destroy production infrastructure without explicit confirmation and backup awareness

## Structural Flow

### Entry
1. Detect provider and environment from project context.
2. Identify state backend, module boundaries, resources, and risk level.
3. Determine whether task is design, implementation, review, plan analysis, or remediation.

### Scenes
1. **PREPARE**: Load Terraform scope, provider, environment, and constraints.
2. **ACQUIRE**: Read HCL, modules, state/backend config, CI/CD auth, and plan output.
3. **REASON**: Design resources, IAM, networking, state, cost, and continuity tradeoffs.
4. **ACT**: Write or review HCL, modules, variables, outputs, and docs.
5. **VERIFY**: Run fmt, validate, plan, scans, and policy checks when available.
6. **FINALIZE**: Report diff, plan risk, validation status, and next apply steps.

### Transitions
- If provider is unclear, detect from HCL before writing.
- If state is local or unprotected, prioritize remote state guidance.
- If plan includes destructive changes, stop for explicit review.
- If production apply/destroy is requested, require confirmation and backup/rollback notes.

### Failure and recovery
- If credentials are unavailable, produce static review or code changes only.
- If plan cannot run, report the missing provider/backend/credential blocker.
- If policy/security scan fails, fix or report concrete remediation.

### Exit
- Success: Terraform change or review is validated and risk-scoped.
- Partial success: unavailable credentials/tools or unreviewed apply risk is explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Detect provider and scope | `READ` | HCL, providers, modules |
| Select cloud/resource mapping | `SELECT` | Multi-cloud mapping |
| Write Terraform | `WRITE` | `.tf`, `.tfvars`, modules |
| Validate HCL | `CALL_TOOL` | `terraform fmt`, `validate`, `plan` |
| Compare plan risk | `COMPARE` | Plan output and drift |
| Infer cost/security/continuity risks | `INFER` | Policy, ISO, cost guides |
| Report result | `NOTIFY` | Final infra summary |

### Tools and instruments
- Terraform CLI and provider ecosystem
- Checkov, tfsec, OPA/Sentinel, Terratest when applicable
- Cost, policy, multi-cloud, and ISO resource guides

### Canonical command path
```bash
terraform fmt -recursive
terraform validate
terraform plan -out=tfplan
```

Run scanners when available before any apply:
```bash
checkov -d .
tfsec .
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `CODEBASE` | Terraform modules, variables, outputs, CI config |
| `LOCAL_FS` | Plans, state config, documentation |
| `PROCESS` | Terraform, scanner, and policy commands |
| `CREDENTIALS` | Cloud provider auth and state backend credentials |
| `NETWORK` | Cloud APIs and remote state backends |

### Preconditions
- Terraform scope and provider can be determined.
- Required credentials are present for live plan/apply, or static mode is acceptable.

### Effects and side effects
- Mutates infrastructure code and documentation.
- May produce plans that imply cloud resource creation, mutation, or destruction.
- Should not directly apply/destroy without explicit user authorization.

### Guardrails

1. **Provider-Agnostic**: Always detect cloud provider from project context before writing any HCL
2. **Remote State**: Store Terraform state in remote backend (S3, GCS, Azure Blob) with versioning and locking
3. **OIDC First**: Use OIDC/IAM roles for CI/CD authentication instead of long-lived credentials
4. **Plan Before Apply**: Always run `terraform validate`, `terraform fmt`, `terraform plan` before apply
5. **Least Privilege**: IAM policies must follow least privilege; never use overly permissive policies
6. **Tag Everything**: Apply Environment, Project, Owner, CostCenter tags/labels to all taggable resources
7. **No Secrets in Code**: Never hardcode passwords, API keys, or tokens in .tf files; use provider secret management
8. **Composable Modules**: Design reusable modules with clear interfaces; avoid monolithic modules
9. **Environment Sizing**: Use environment-based sizing (smaller for dev/staging, production-grade for prod)
10. **Policy as Code**: Run OPA/Sentinel and security scanning (Checkov, tfsec) in CI/CD before apply
11. **Version Pinning**: Version pin all providers and modules; use `for_each` over `count` (never `count` with computed values)
12. **Cost Awareness**: Implement lifecycle policies, autoscaling schedules, and review cost estimates before apply
13. **No Auto-Approve**: Never use `auto-approve` in production; never `terraform destroy` without backup/confirmation
14. **Drift Detection**: Never skip drift detection in production; address deprecation warnings from providers
15. **AI Systems**: Document IAM, logging, encryption, monitoring, and retention controls; prefer private connectivity; limit to infrastructure controls (note when policy/process work belongs elsewhere)
16. **Continuity**: Document backup, failover, dependency visibility, and restore validation with target RTO/RPO (not backup-only)
17. **Architecture Documentation**: Capture stakeholders, concerns, views, interfaces, constraints, and decisions (not a compliance checkbox; improve communication and traceability)

### Cloud Provider Detection

| Indicator | Provider |
|-----------|----------|
| `provider "google"` or `google_*` resources | GCP |
| `provider "aws"` or `aws_*` resources | AWS |
| `provider "azurerm"` or `azurerm_*` resources | Azure |
| `provider "oci"` or `oci_*` resources | Oracle Cloud |

### Multi-Cloud Resource Mapping

| Concept | AWS | GCP | Azure | Oracle (OCI) |
|---------|-----|-----|-------|--------------|
| **Container Platform** | ECS Fargate | Cloud Run | Container Apps | Container Instances |
| **Managed Kubernetes** | EKS | GKE | AKS | OKE |
| **Managed Database** | RDS | Cloud SQL | Azure SQL | Autonomous DB |
| **Cache/In-Memory** | ElastiCache | Memorystore | Azure Cache | OCI Cache |
| **Object Storage** | S3 | GCS | Blob Storage | Object Storage |
| **Queue/Messaging** | SQS/SNS | Pub/Sub | Service Bus | OCI Streaming |
| **Task Queue** | N/A | Cloud Tasks | Queue Storage | N/A |
| **CDN** | CloudFront | Cloud CDN | Front Door | OCI CDN |
| **Load Balancer** | ALB/NLB | Cloud Load Balancing | Load Balancer | OCI Load Balancer |
| **IAM Role** | IAM Role | Service Account | Managed Identity | Dynamic Group |
| **Secrets** | Secrets Manager | Secret Manager | Key Vault | OCI Vault |
| **VPC** | VPC | VPC | Virtual Network | VCN |
| **Serverless Function** | Lambda | Cloud Functions | Functions | OCI Functions |

## References

Follow `resources/execution-protocol.md` step by step.
See `resources/examples.md` for input/output examples.
Use `resources/multi-cloud-examples.md` for provider-specific HCL patterns.
Use `resources/cost-optimization.md` for cost reduction strategies.
Use `resources/policy-testing-examples.md` for OPA, Sentinel, and Terratest patterns.
Use `resources/iso-42001-infra.md` for AI governance, continuity, and architecture controls.
Before submitting, run `resources/checklist.md`.
Vendor-specific execution protocols are injected automatically by `oma agent:spawn`.
Source files live under `../_shared/runtime/execution-protocols/{vendor}.md`.

- Execution steps: `resources/execution-protocol.md`
- Self-check: `resources/checklist.md`
- Examples: `resources/examples.md`
- Multi-cloud HCL patterns: `resources/multi-cloud-examples.md`
- Cost optimization: `resources/cost-optimization.md`
- Policy & testing: `resources/policy-testing-examples.md`
- ISO controls: `resources/iso-42001-infra.md`
- Error recovery: `resources/error-playbook.md`
- Context loading: `../_shared/core/context-loading.md`
- Reasoning templates: `../_shared/core/reasoning-templates.md`
- Clarification: `../_shared/core/clarification-protocol.md`
- Context budget: `../_shared/core/context-budget.md`
- Difficulty assessment: `../_shared/core/difficulty-guide.md`
- Lessons learned: `../_shared/core/lessons-learned.md`
- Observability handoff: `../oma-observability/SKILL.md` §Integrations — Collector topology, transport tuning, release metadata

### Knowledge Reference
terraform, infrastructure-as-code, iac, cloud, aws, gcp, azure, oracle, oci, multi-cloud, devops, provisioning, infrastructure, compute, database, storage, networking, iam, oidc, workload identity, container, kubernetes, serverless, vpc, subnet, load balancer, cdn, secrets management, state management, backend, provider
