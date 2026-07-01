# TF Infra Agent: Examples

## Example 1: GCP Cloud Run Service with Cloud SQL

**Input:** "Deploy an API service on Cloud Run and connect it to Cloud SQL PostgreSQL"

**Output:**
- Files created: `compute.tf`, `database.tf`, `networking.tf`, `iam.tf`, `outputs.tf`
- Cloud provider: GCP (detected from existing `provider "google"`)
- Resources: Cloud Run service, Cloud SQL PostgreSQL, VPC connector, service account
- Key decisions:
  - VPC connector for private Cloud SQL access
  - Dedicated service account with minimal permissions
  - Cloud SQL Auth Proxy sidecar for secure connection
  - Environment-based sizing: `db-f1-micro` (dev) vs `db-custom-4-16384` (prod)

## Example 2: AWS ECS Fargate with GitHub OIDC

**Input:** "Set up OIDC so GitHub Actions can deploy to ECS Fargate"

**Output:**
- Files created: `cicd-auth.tf`, `iam.tf`
- Cloud provider: AWS (detected from existing `aws_*` resources)
- Resources: OIDC provider, IAM role with trust policy, ECR/ECS permissions
- Key decisions:
  - Trust policy scoped to specific repo and branch
  - Permissions limited to ECR push + ECS task update only
  - No long-lived access keys

## Example 3: Terraform Plan Review

**Input:** "Review this terraform plan output"

**Output:**
- Summary: 3 to add, 1 to change, 1 to destroy
- Risk: HIGH; `aws_db_instance` will be destroyed and recreated (engine version change)
- Recommendation: Use `lifecycle { prevent_destroy = true }` or manual migration
- Checklist: backup verified, maintenance window set, rollback plan documented
