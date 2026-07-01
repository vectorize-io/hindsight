# TF Infra Agent: Execution Protocol

## Step 0: Prepare

1. **Assess difficulty**: see `../../_shared/core/difficulty-guide.md`
   - **Simple**: Skip to Step 3 | **Medium**: All 4 steps | **Complex**: All steps + checkpoints
2. **Check lessons**: read the infrastructure section in `../../_shared/core/lessons-learned.md` for past Terraform pitfalls
3. **Clarify requirements**: follow `../../_shared/core/clarification-protocol.md`
   - Check **Uncertainty Triggers**: IAM/security, compliance (PII, residency, audit), cost/sizing, existing-resource or state conflicts, provider/region ambiguity?
   - Determine level: LOW → proceed | MEDIUM → present options | HIGH → ask immediately
4. **Budget context**: follow `../../_shared/core/context-budget.md` (read symbols and file overviews, not whole files)

**Intelligent Escalation**: When uncertain, escalate early. Provisioning on wrong assumptions costs more than asking. Don't blindly proceed.

Follow these steps in order (adjust depth by difficulty).

## Step 1: Analyze

1. **Identify Cloud Provider**: Detect from `provider.tf`, backend config, or existing resources
2. Scan existing Terraform files for naming conventions, module patterns, and state configuration
3. Identify required services, resource dependencies, and security constraints
4. Load domain-specific references:
   - `multi-cloud-examples.md` when creating compute, OIDC, or secret resources
   - `cost-optimization.md` when sizing or cost review is needed
   - `policy-testing-examples.md` when setting up policy enforcement or tests
   - `iso-42001-infra.md` when AI governance, continuity, or architecture documentation is required
5. List assumptions and unknowns

## Step 2: Plan

1. Define resource naming using project conventions and `locals` block:

   | Resource Type | Pattern | Examples |
   |--------------|---------|----------|
   | Compute | `{prefix}-{service}` | `fs-dev-api`, `fs-prod-web` |
   | Database | `{prefix}-db` | `fs-dev-db`, `fs-prod-postgres` |
   | Storage | `{prefix}-{purpose}` | `fs-dev-assets`, `fs-dev-tfstate` |
   | IAM Role/SA | `{prefix}-{role}` | `fs-dev-api-role`, `fs-dev-deployer` |
   | Network | `{prefix}-{type}` | `fs-dev-vpc`, `fs-dev-subnet` |

2. Plan file structure following project patterns:
   ```
   apps/infra/
   ├── provider.tf          # Provider configuration
   ├── versions.tf          # Version constraints
   ├── variables.tf         # Input variables
   ├── locals.tf            # Local values and naming
   ├── backend.tf           # State backend
   ├── compute.tf           # Compute resources
   ├── database.tf          # Databases
   ├── storage.tf           # Object storage
   ├── networking.tf        # VPC, subnets, LB, CDN
   ├── messaging.tf         # Queues and messaging
   ├── iam.tf               # IAM roles and policies
   ├── cicd-auth.tf         # OIDC/workload identity
   ├── security.tf          # Security groups, WAF, secrets
   ├── outputs.tf           # Output values
   └── terraform.tfvars     # Variable values (gitignored)
   ```
3. Plan module interfaces following composability principles:
   ```
   modules/
   ├── vpc/                    # Reusable VPC module
   │   ├── main.tf
   │   ├── variables.tf
   │   ├── outputs.tf
   │   └── README.md
   ├── database/               # Reusable database module
   └── compute/                # Reusable compute module
   ```
   - Expose required variables only; provide sensible defaults for optional
   - Export essential outputs only
   - Document all inputs/outputs in README.md
   - Version modules using Git tags or Terraform Registry
4. Identify security requirements (IAM, encryption, network boundaries)
5. Estimate cost impact for new resources

## Step 3: Implement

1. Create/modify .tf files in dependency order (provider → backend → networking → compute → outputs)
2. Use `locals` for environment-specific naming and tags
3. Apply consistent tags: Environment, Project, Owner, CostCenter
4. Use `for_each` over `count` for resource collections
5. Set `depends_on` for explicit resource ordering where needed
6. Design composable modules with:
   - Required variables only, sensible defaults for optional
   - Essential outputs only
   - Documented inputs/outputs
7. Run `terraform validate` and `terraform fmt`
8. Run `terraform plan` and review output

## Infrastructure Testing Levels

| Level | Tool | Purpose |
|-------|------|---------|
| Unit | `terraform validate` | Syntax, variable types |
| Static Analysis | TFLint, Checkov | Best practices, security |
| Integration | Terratest | Resource creation verification |
| Compliance | OPA/Sentinel | Organizational policy enforcement |
| E2E | Custom scripts | Full workflow validation |

See `policy-testing-examples.md` for Terratest, Kitchen-Terraform, and CI/CD integration examples.

## Step 4: Verify

1. Run `checklist.md` self-verification
2. Run `../../_shared/core/common-checklist.md` common checks
3. Confirm:
   - `terraform validate` passes
   - `terraform plan` shows expected changes only
   - No secrets in .tf files
   - All resources tagged
   - IAM follows least privilege
   - Cost impact acceptable
4. For AI systems: verify IAM, logging, encryption, monitoring, and retention controls documented
5. For continuity: verify RTO/RPO, failover, backup, and restore documented

## Output Format

When creating new infrastructure, provide:
1. Cloud provider identified from context
2. Complete HCL code blocks for each resource
3. Required variable definitions with types and descriptions
4. Outputs for resource IDs and endpoints
5. Migration notes if importing existing resources
6. Cost estimation considerations
7. For AI systems: infrastructure control notes covering access, logging, encryption, monitoring, and retention
8. For continuity-sensitive systems: RTO/RPO, failover, backup, and restore notes
9. For architecture documentation requests: stakeholder/concern/view/rationale notes

When reviewing terraform plan, provide:
1. Summary of changes (add/change/destroy counts)
2. Risk assessment for destructive changes
3. Provider-specific considerations
4. Confirmation checklist before apply
