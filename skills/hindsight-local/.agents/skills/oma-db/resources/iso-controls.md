# DB Agent - ISO 27001 / 27002 / 22301 Guide

Use this file when the task involves database controls, audit readiness, resilience, or continuity requirements.

## Positioning

- **ISO 27001**: management-system and control objectives view
- **ISO 27002**: practical control guidance view
- **ISO 22301**: business continuity and recovery view

Do not claim compliance from schema work alone. Use these standards to improve database design decisions, control coverage, and operational evidence.

## 1. ISO 27001 / 27002 for Databases

Recommend improvements when the database design weakens:

- access control and least privilege
- separation of duties for admin, operator, developer, and read-only access
- encryption at rest and in transit
- key and secret management
- audit logging and tamper-resistant log retention
- change management and schema migration traceability
- backup security and retention handling
- secure configuration and baseline hardening
- data classification, masking, and PII handling

Typical DB-level suggestions:

- role-based access with least privilege
- dedicated break-glass admin path
- column or application-layer protection for sensitive fields where needed
- immutable or centralized audit log export
- secret rotation and non-human service identities
- explicit approval path for destructive schema changes

## 2. ISO 22301 for Databases

Recommend improvements when the design weakens business continuity:

- RTO / RPO definition
- failover readiness
- backup coverage and retention
- restore validation frequency
- dependency mapping for upstream/downstream systems
- data replication strategy
- hot/warm/cold standby decisions
- archival and cold-data recovery assumptions

Typical DB-level suggestions:

- define tiered continuity expectations by database criticality
- document backup cadence aligned to RPO
- document restore and failover process aligned to RTO
- test restore drills regularly, not only backup completion
- make derived indexes rebuildable from canonical data
- separate source-of-truth data from caches and secondary indexes

## 3. Review Questions

Ask these when standards matter:

- Who can read, write, alter schema, and administer the database?
- Are privileged actions logged and retained?
- Are secrets and keys managed outside code and migration files?
- What evidence exists that backups can actually be restored?
- What is the target RPO and RTO?
- Which systems depend on this database to continue service?
- Can cold/archive data be recovered within required time?
- Are vector indexes or search indexes rebuildable from canonical data?

## 4. Output Pattern

When relevant, add a short section like:

```md
## ISO Control Notes

### ISO 27001 / 27002
- Access control gap:
- Logging / traceability gap:
- Encryption / secret gap:
- Recommended control:

### ISO 22301
- Criticality:
- RPO / RTO gap:
- Backup / restore gap:
- Failover / dependency gap:
- Recommended continuity improvement:
```

## 5. Guardrails

- Keep standards comments practical and database-specific
- Prefer implementable controls over abstract policy language
- Distinguish between source-of-truth data and rebuildable derived data
- If the issue is infrastructure-only, involve `tf-infra-agent`
- If the issue is org-wide audit/process ownership, involve QA or PM
