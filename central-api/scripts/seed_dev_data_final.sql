-- Final seed data matching actual CollabMind schema
-- Run: docker exec -i collabmind-postgres psql -U dev -d collabmind < /path/to/this/file

-- Clear existing seed data (optional)
DELETE FROM source_ingestion_jobs WHERE id LIKE 'job-%';
DELETE FROM source_documents WHERE id LIKE 'doc-%';
DELETE FROM source_connectors WHERE id LIKE 'conn-%';

-- Add sample connectors
INSERT INTO source_connectors (id, workspace_id, provider, status, connected_by, account_email, config, created_at, updated_at)
VALUES 
  ('conn-gdrive-001', 'default', 'google-drive', 'active', '00000000-0000-0000-0000-000000000002', 'user@example.com', '{"folder_id": "sample-folder-123"}', now() - interval '2 days', now() - interval '1 day'),
  ('conn-gdrive-002', 'default', 'google-drive', 'active', '00000000-0000-0000-0000-000000000002', 'team@example.com', '{"folder_id": "sample-folder-456"}', now() - interval '1 day', now() - interval '12 hours'),
  ('conn-notion-001', 'default', 'notion', 'pending', '00000000-0000-0000-0000-000000000002', 'admin@example.com', '{"workspace_id": "sample-workspace"}', now() - interval '3 hours', now() - interval '3 hours')
ON CONFLICT (id) DO UPDATE SET updated_at = EXCLUDED.updated_at;

-- Add sample documents (matching actual schema)
INSERT INTO source_documents (id, workspace_id, connector_id, provider, external_id, name, mime_type, size, web_view_link, checksum, trashed, enabled, sync_status, metadata, created_at, updated_at)
VALUES 
  ('doc-001', 'default', 'conn-gdrive-001', 'google-drive', 'gdrive-file-123', 'Project Requirements.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 45000, 'https://drive.google.com/file/d/123', 'abc123', false, true, 'synced', '{}', now() - interval '2 days', now() - interval '1 day'),
  ('doc-002', 'default', 'conn-gdrive-001', 'google-drive', 'gdrive-file-456', 'Architecture Design.pdf', 'application/pdf', 128000, 'https://drive.google.com/file/d/456', 'def456', false, true, 'synced', '{}', now() - interval '2 days', now() - interval '1 day'),
  ('doc-003', 'default', 'conn-gdrive-002', 'google-drive', 'gdrive-file-789', 'Meeting Notes Q1.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 32000, 'https://drive.google.com/file/d/789', 'ghi789', false, true, 'synced', '{}', now() - interval '1 day', now() - interval '12 hours'),
  ('doc-004', 'default', 'conn-gdrive-001', 'google-drive', 'gdrive-file-abc', 'Code Review Guidelines.md', 'text/markdown', 8500, 'https://drive.google.com/file/d/abc', 'jkl012', false, true, 'synced', '{}', now() - interval '1 day', now() - interval '6 hours'),
  ('doc-005', 'default', 'conn-gdrive-002', 'google-drive', 'gdrive-file-def', 'Q2 Planning.pdf', 'application/pdf', 95000, 'https://drive.google.com/file/d/def', 'mno345', false, true, 'synced', '{}', now() - interval '12 hours', now() - interval '3 hours')
ON CONFLICT (id) DO UPDATE SET updated_at = EXCLUDED.updated_at;

-- Add sample ingestion jobs
INSERT INTO source_ingestion_jobs (id, workspace_id, connector_id, document_id, source, external_id, mime_type, operation, status, error, metadata, created_at, updated_at)
VALUES 
  ('job-001', 'default', 'conn-gdrive-001', 'doc-001', 'google-drive', 'gdrive-file-123', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'index', 'completed', NULL, '{"chunks": 45, "tokens": 12000}', now() - interval '2 hours', now() - interval '1 hour'),
  ('job-002', 'default', 'conn-gdrive-002', 'doc-003', 'google-drive', 'gdrive-file-789', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'index', 'completed', NULL, '{"chunks": 28, "tokens": 7500}', now() - interval '1 hour', now() - interval '30 minutes'),
  ('job-003', 'default', 'conn-gdrive-001', 'doc-002', 'google-drive', 'gdrive-file-456', 'application/pdf', 'index', 'running', NULL, '{"chunks": 15, "tokens": 4000}', now() - interval '15 minutes', now() - interval '5 minutes'),
  ('job-004', 'default', 'conn-notion-001', NULL, 'notion', 'notion-page-123', 'text/html', 'index', 'failed', 'Invalid API key', '{}', now() - interval '3 hours', now() - interval '3 hours'),
  ('job-005', 'default', 'conn-gdrive-002', 'doc-005', 'google-drive', 'gdrive-file-def', 'application/pdf', 'index', 'completed', NULL, '{"chunks": 38, "tokens": 9800}', now() - interval '3 hours', now() - interval '2 hours')
ON CONFLICT (id) DO UPDATE SET updated_at = EXCLUDED.updated_at;

-- Show results
SELECT 'Seed data loaded successfully!' as message;
SELECT count(*) as connectors FROM source_connectors WHERE id LIKE 'conn-%';
SELECT count(*) as documents FROM source_documents WHERE id LIKE 'doc-%';
SELECT count(*) as jobs FROM source_ingestion_jobs WHERE id LIKE 'job-%';
