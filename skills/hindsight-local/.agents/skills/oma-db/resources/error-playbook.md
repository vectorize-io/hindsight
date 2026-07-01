# DB Agent - Error Playbook

## Symptom: Relational model is breaking 3NF everywhere
- Re-check whether reporting columns, derived values, and snapshots are being mixed into OLTP tables
- Split transactional source-of-truth from read models or marts
- Keep denormalization only where query cost justifies it

## Symptom: NoSQL model keeps requiring cross-document joins
- Re-evaluate aggregate boundaries and access patterns
- If strong cross-entity consistency dominates, switch recommendation toward relational storage

## Symptom: Isolation level is unclear
- Start from anomaly prevention required by the business flow
- Document what must be prevented: dirty read, non-repeatable read, phantom, lost update
- Choose the lowest level that still blocks unacceptable anomalies

## Symptom: Capacity estimate is unreliable
- Separate online traffic, batch traffic, retention, and backup retention
- Estimate by object first, then aggregate to tablespace and disk totals
- Add growth and reindex/maintenance headroom

## Symptom: Backup plan exists but restore confidence is low
- Add restore drill frequency
- Define sample recovery scenarios and max tolerable data loss
- Distinguish snapshot/full backup from incremental/log-based recovery
