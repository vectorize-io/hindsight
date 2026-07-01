-- Seed development data for CollabMind dashboard testing
-- Run with: docker exec -i collabmind-postgres psql -U dev -d collabmind < seed_dev_data.sql

-- Add sample connectors
INSERT INTO source_connectors (id, workspace_id, provider, status, connected_by, account_email, config, created_at, updated_at)
VALUES 
  ('conn-gdrive-001', 'default', 'google-drive', 'active', '00000000-0000-0000-0000-000000000002', 'user@example.com', '{"folder_id": "sample-folder-123"}', now() - interval '2 days', now() - interval '1 day'),
  ('conn-gdrive-002', 'default', 'google-drive', 'active', '00000000-0000-0000-0000-000000000002', 'team@example.com', '{"folder_id": "sample-folder-456"}', now() - interval '1 day', now() - interval '12 hours'),
  ('conn-notion-001', 'default', 'notion', 'pending', '00000000-0000-0000-0000-000000000002', 'admin@example.com', '{"workspace_id": "sample-workspace"}', now() - interval '3 hours', now() - interval '3 hours')
ON CONFLICT (id) DO NOTHING;

-- Add sample documents
INSERT INTO source_documents (id, workspace_id, connector_id, external_id, title, mime_type, url, size_bytes, status, indexed_at, created_at, updated_at)
VALUES 
  ('doc-001', 'default', 'conn-gdrive-001', 'gdrive-file-123', 'Project Requirements.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'https://drive.google.com/file/d/123', 45000, 'indexed', now() - interval '1 day', now() - interval '2 days', now() - interval '1 day'),
  ('doc-002', 'default', 'conn-gdrive-001', 'gdrive-file-456', 'Architecture Design.pdf', 'application/pdf', 'https://drive.google.com/file/d/456', 128000, 'indexed', now() - interval '1 day', now() - interval '2 days', now() - interval '1 day'),
  ('doc-003', 'default', 'conn-gdrive-002', 'gdrive-file-789', 'Meeting Notes Q1.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'https://drive.google.com/file/d/789', 32000, 'indexed', now() - interval '12 hours', now() - interval '1 day', now() - interval '12 hours'),
  ('doc-004', 'default', 'conn-gdrive-001', 'gdrive-file-abc', 'Code Review Guidelines.md', 'text/markdown', 'https://drive.google.com/file/d/abc', 8500, 'indexed', now() - interval '6 hours', now() - interval '1 day', now() - interval '6 hours'),
  ('doc-005', 'default', 'conn-gdrive-002', 'gdrive-file-def', 'Q2 Planning.pdf', 'application/pdf', 'https://drive.google.com/file/d/def', 95000, 'indexed', now() - interval '3 hours', now() - interval '12 hours', now() - interval '3 hours')
ON CONFLICT (id) DO NOTHING;

-- Add sample ingestion jobs
INSERT INTO source_ingestion_jobs (id, workspace_id, connector_id, document_id, source, external_id, mime_type, operation, status, error, metadata, created_at, updated_at)
VALUES 
  ('job-001', 'default', 'conn-gdrive-001', 'doc-001', 'google-drive', 'gdrive-file-123', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'index', 'completed', NULL, '{"chunks": 45, "tokens": 12000}', now() - interval '2 hours', now() - interval '1 hour'),
  ('job-002', 'default', 'conn-gdrive-002', 'doc-003', 'google-drive', 'gdrive-file-789', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'index', 'completed', NULL, '{"chunks": 28, "tokens": 7500}', now() - interval '1 hour', now() - interval '30 minutes'),
  ('job-003', 'default', 'conn-gdrive-001', 'doc-002', 'google-drive', 'gdrive-file-456', 'application/pdf', 'index', 'running', NULL, '{"chunks": 15, "tokens": 4000}', now() - interval '15 minutes', now() - interval '5 minutes'),
  ('job-004', 'default', 'conn-notion-001', NULL, 'notion', 'notion-page-123', 'text/html', 'index', 'failed', 'Invalid API key', '{}', now() - interval '3 hours', now() - interval '3 hours'),
  ('job-005', 'default', 'conn-gdrive-002', 'doc-005', 'google-drive', 'gdrive-file-def', 'application/pdf', 'index', 'completed', NULL, '{"chunks": 38, "tokens": 9800}', now() - interval '3 hours', now() - interval '2 hours')
ON CONFLICT (id) DO NOTHING;

-- Add more diverse audit events
INSERT INTO audit_events (tenant_id, actor_id, source_app_id, operation, resource_type, resource_id, outcome, metadata, created_at)
VALUES 
  ('dev-tenant', 'user-001', 'collabmind-console', 'connector_create', 'connector', 'conn-gdrive-001', 'success', '{"provider": "google-drive"}', EXTRACT(epoch FROM (now() - interval '2 days')) * 1000),
  ('dev-tenant', 'user-001', 'collabmind-console', 'sync_trigger', 'connector', 'conn-gdrive-001', 'success', '{"job_id": "job-001"}', EXTRACT(epoch FROM (now() - interval '2 hours')) * 1000),
  ('dev-tenant', 'user-002', 'collabmind-console', 'document_view', 'document', 'doc-001', 'success', NULL, EXTRACT(epoch FROM (now() - interval '1 hour')) * 1000),
  ('dev-tenant', 'user-001', 'collabmind-console', 'sync_trigger', 'connector', 'conn-gdrive-002', 'success', '{"job_id": "job-002"}', EXTRACT(epoch FROM (now() - interval '1 hour')) * 1000),
  ('dev-tenant', 'user-003', 'collabmind-api', 'memory_search', 'memory', NULL, 'success', '{"query": "project requirements", "hits": 5}', EXTRACT(epoch FROM (now() - interval '30 minutes')) * 1000),
  ('dev-tenant', 'user-001', 'collabmind-console', 'sync_trigger', 'connector', 'conn-gdrive-001', 'success', '{"job_id": "job-003"}', EXTRACT(epoch FROM (now() - interval '15 minutes')) * 1000),
  ('dev-tenant', 'user-002', 'collabmind-console', 'document_search', 'document', NULL, 'success', '{"query": "architecture", "hits": 3}', EXTRACT(epoch FROM (now() - interval '10 minutes')) * 1000),
  ('dev-tenant', 'user-001', 'collabmind-console', 'connector_create', 'connector', 'conn-notion-001', 'success', '{"provider": "notion"}', EXTRACT(epoch FROM (now() - interval '3 hours')) * 1000),
  ('dev-tenant', 'user-001', 'collabmind-console', 'sync_trigger', 'connector', 'conn-notion-001', 'error', '{"job_id": "job-004", "error": "Invalid API key"}', EXTRACT(epoch FROM (now() - interval '3 hours')) * 1000),
  ('dev-tenant', 'user-002', 'collabmind-api', 'memory_store', 'memory', 'mem-001', 'success', '{"type": "conversation", "sensitivity": "internal"}', EXTRACT(epoch FROM (now() - interval '25 minutes')) * 1000),
  ('dev-tenant', 'user-003', 'collabmind-console', 'model_call', 'ai', 'localai-llama', 'success', '{"model": "llama-3-8b", "tokens": 500}', EXTRACT(epoch FROM (now() - interval '20 minutes')) * 1000),
  ('dev-tenant', 'user-001', 'collabmind-api', 'policy_check', 'governance', NULL, 'reject', '{"sensitivity": "secret", "reasons": ["contains API key"]}', EXTRACT(epoch FROM (now() - interval '18 minutes')) * 1000)
ON CONFLICT DO NOTHING;

SELECT 'Seed data inserted successfully!' as message;
SELECT count(*) as connectors FROM source_connectors;
SELECT count(*) as documents FROM source_documents;
SELECT count(*) as jobs FROM source_ingestion_jobs;
SELECT count(*) as audit_events FROM audit_events WHERE tenant_id = 'dev-tenant';
