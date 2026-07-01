---
description: Terraform infrastructure-as-code standards for multi-cloud (AWS,
  GCP, Azure, OCI) provisioning
paths: "**/*.{tf,tfvars,hcl}"
---


# Infrastructure Standards

## Core Rules

1. **Provider-Agnostic**: Detect cloud provider from project context before writing any HCL
2. **Remote State**: Store state in remote backend (S3, GCS, Azure Blob) with versioning and locking
3. **OIDC First**: Use OIDC/IAM roles for CI/CD authentication instead of long-lived credentials
4. **Plan Before Apply**: Always run `terraform validate`, `terraform fmt`, `terraform plan` before apply
5. **Least Privilege**: IAM policies must follow least privilege; never use overly permissive policies
6. **Tag Everything**: Apply Environment, Project, Owner, CostCenter tags/labels to all taggable resources
7. **No Secrets in Code**: Never hardcode passwords, API keys, or tokens in .tf files
8. **Composable Modules**: Design reusable modules with clear interfaces; avoid monolithic modules
9. **Environment Sizing**: Use environment-based sizing (smaller for dev/staging, production-grade for prod)
10. **Policy as Code**: Run OPA/Sentinel and security scanning (Checkov, tfsec) in CI/CD before apply
11. **Version Pinning**: Pin all providers and modules; use `for_each` over `count`
12. **Cost Awareness**: Implement lifecycle policies, autoscaling schedules, and review cost estimates before apply
13. **No Auto-Approve**: Never use `auto-approve` in production; never `terraform destroy` without backup
14. **Drift Detection**: Never skip drift detection in production
