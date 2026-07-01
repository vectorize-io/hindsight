# TF Infra Agent: Self-Verification Checklist

## Configuration
- [ ] `terraform validate` passes without errors
- [ ] `terraform fmt` applied (consistent formatting)
- [ ] Provider and module versions pinned in `versions.tf`
- [ ] Remote backend configured with state locking and encryption

## Security
- [ ] No secrets, passwords, or API keys in .tf files
- [ ] `terraform.tfvars` excluded from git
- [ ] IAM policies follow least privilege
- [ ] OIDC/workload identity used for CI/CD (no long-lived credentials)
- [ ] Sensitive outputs marked with `sensitive = true`
- [ ] Encryption at rest enabled for storage and databases
- [ ] Network boundaries defined (security groups, firewall rules)

## Naming & Tags
- [ ] Consistent naming convention using `locals` block
- [ ] All taggable resources include: Environment, Project, Owner, CostCenter
- [ ] Resource names match project patterns

## Resource Design
- [ ] `for_each` used over `count` for resource collections
- [ ] `count` not used with computed values that could cause recreation
- [ ] `depends_on` set for explicit ordering where needed
- [ ] Modules are composable with clear interfaces
- [ ] No monolithic modules doing too many things
- [ ] Environment-based sizing (smaller for dev/staging)

## Operations Safety
- [ ] No `auto-approve` in production environments
- [ ] No `terraform destroy` without explicit backup/confirmation
- [ ] Drift detection not skipped in production
- [ ] Provider deprecation warnings addressed
- [ ] State files not stored locally in team environments

## Cost
- [ ] Cost impact reviewed via `terraform plan`
- [ ] Storage lifecycle rules configured
- [ ] Autoscaling schedules set for off-hours scaling
- [ ] Reserved Instances / Savings Plans considered for production

## Policy & Testing
- [ ] Policy checks (OPA/Sentinel) pass
- [ ] Security scanning (Checkov/tfsec) clean
- [ ] Critical modules have integration tests

## Compliance (when applicable)
- [ ] AI systems: IAM, logging, encryption, monitoring, retention controls documented
- [ ] Continuity: RTO/RPO, failover, backup, restore assumptions documented
- [ ] Architecture: stakeholders, concerns, views, rationale captured
