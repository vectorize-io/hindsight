"""
End-to-end tests for file retain (upload, convert, retain) functionality.
"""

import io
import json

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def sample_pdf_content():
    """Create a simple PDF-like content for testing."""
    # This is a minimal PDF that markitdown can parse
    return b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test Document) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000317 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
410
%%EOF
"""


@pytest.fixture
def sample_txt_content():
    """Create simple text content."""
    return b"This is a test document.\nIt contains some important information.\nAlice works at Google."


@pytest.mark.asyncio
async def test_file_retain_basic(memory_no_llm_verify, sample_txt_content):
    """Test basic file upload and conversion."""
    from hindsight_api.api.http import create_app

    app = create_app(memory_no_llm_verify, initialize_memory=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create a bank first
        bank_response = await client.put("/v1/default/banks/test-file-bank", json={"name": "Test File Bank"})
        assert bank_response.status_code in (200, 201)

        # Upload file
        request_data = {
            "document_tags": ["test"],
            "async": True,
        }

        files = {"files": ("test.txt", sample_txt_content, "text/plain")}
        data = {"request": json.dumps(request_data)}

        response = await client.post(
            "/v1/default/banks/test-file-bank/files/retain",
            files=files,
            data=data,
        )

        assert response.status_code == 200
        result = response.json()
        assert "operation_ids" in result
        assert len(result["operation_ids"]) == 1


@pytest.mark.asyncio
async def test_file_retain_with_metadata(memory_no_llm_verify, sample_txt_content):
    """Test file upload with per-file metadata."""
    from hindsight_api.api.http import create_app

    app = create_app(memory_no_llm_verify, initialize_memory=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create bank
        bank_response = await client.put("/v1/default/banks/test-file-meta-bank", json={"name": "Test Meta Bank"})
        assert bank_response.status_code in (200, 201)

        # Upload file with metadata
        request_data = {
            "document_tags": ["work", "reports"],
            "async": True,
            "files_metadata": [
                {
                    "document_id": "test_doc_123",
                    "context": "quarterly report",
                    "metadata": {"author": "Alice", "year": "2024"},
                    "tags": ["Q1"],
                }
            ],
        }

        files = {"files": ("report.txt", sample_txt_content, "text/plain")}
        data = {"request": json.dumps(request_data)}

        response = await client.post(
            "/v1/default/banks/test-file-meta-bank/files/retain",
            files=files,
            data=data,
        )

        assert response.status_code == 200
        result = response.json()
        assert "operation_ids" in result
        assert len(result["operation_ids"]) == 1


@pytest.mark.asyncio
async def test_file_retain_multiple_files(memory_no_llm_verify, sample_txt_content):
    """Test uploading multiple files at once."""
    from hindsight_api.api.http import create_app

    app = create_app(memory_no_llm_verify, initialize_memory=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create bank
        bank_response = await client.put("/v1/default/banks/test-multi-file-bank", json={"name": "Test Multi Bank"})
        assert bank_response.status_code in (200, 201)

        # Upload multiple files
        request_data = {
            "async": True,
            "files_metadata": [
                {"document_id": "doc1", "tags": ["file1"]},
                {"document_id": "doc2", "tags": ["file2"]},
            ],
        }

        content1 = b"First document content"
        content2 = b"Second document content"

        files = [
            ("files", ("file1.txt", content1, "text/plain")),
            ("files", ("file2.txt", content2, "text/plain")),
        ]
        data = {"request": json.dumps(request_data)}

        response = await client.post(
            "/v1/default/banks/test-multi-file-bank/files/retain",
            files=files,
            data=data,
        )

        assert response.status_code == 200
        result = response.json()
        assert "operation_ids" in result
        assert len(result["operation_ids"]) == 2


@pytest.mark.asyncio
async def test_file_retain_validation_errors(memory_no_llm_verify):
    """Test validation errors."""
    from hindsight_api.api.http import create_app

    app = create_app(memory_no_llm_verify, initialize_memory=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create bank
        bank_response = await client.put("/v1/default/banks/test-validation-bank", json={"name": "Test Validation Bank"})
        assert bank_response.status_code in (200, 201)

        # Test: metadata count mismatch
        request_data = {
            "async": True,
            "files_metadata": [
                {"document_id": "doc1"},
                {"document_id": "doc2"},  # 2 metadata entries
            ],
        }

        files = {"files": ("file1.txt", b"content", "text/plain")}  # But only 1 file
        data = {"request": json.dumps(request_data)}

        response = await client.post(
            "/v1/default/banks/test-validation-bank/files/retain",
            files=files,
            data=data,
        )

        assert response.status_code == 400
        assert "files_metadata count" in response.json()["detail"]


@pytest.mark.asyncio
async def test_file_retain_no_files(memory_no_llm_verify):
    """Test error when no files provided."""
    from hindsight_api.api.http import create_app

    app = create_app(memory_no_llm_verify, initialize_memory=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create bank
        bank_response = await client.put("/v1/default/banks/test-no-files-bank", json={"name": "Test No Files Bank"})
        assert bank_response.status_code in (200, 201)

        request_data = {
            "async": True,
        }

        # No files provided
        data = {"request": json.dumps(request_data)}

        response = await client.post(
            "/v1/default/banks/test-no-files-bank/files/retain",
            data=data,
        )

        # FastAPI will return 422 for missing required field
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_file_retain_sync_not_supported(memory_no_llm_verify, sample_txt_content):
    """Test that file retain is always async (sync is not supported)."""
    from hindsight_api.api.http import create_app

    app = create_app(memory_no_llm_verify, initialize_memory=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create bank
        bank_response = await client.put("/v1/default/banks/test-sync-bank", json={"name": "Test Sync Bank"})
        assert bank_response.status_code in (200, 201)

        # File retain is always async - just verify it succeeds and returns operation_ids
        files = {"files": ("test.txt", sample_txt_content, "text/plain")}
        data = {"request": json.dumps({})}

        response = await client.post(
            "/v1/default/banks/test-sync-bank/files/retain",
            files=files,
            data=data,
        )

        assert response.status_code == 200
        result = response.json()
        assert "operation_ids" in result


@pytest.mark.asyncio
async def test_file_storage_postgresql(memory_no_llm_verify, sample_txt_content):
    """Test file storage in PostgreSQL."""
    # Test that files are stored and retrieved correctly
    storage = memory_no_llm_verify._file_storage

    # Store a file
    key = "test/file1.txt"
    stored_key = await storage.store(
        file_data=sample_txt_content,
        key=key,
        metadata={"content_type": "text/plain"},
    )

    assert stored_key == key

    # Retrieve the file
    retrieved = await storage.retrieve(key)
    assert retrieved == sample_txt_content

    # Check if file exists
    exists = await storage.exists(key)
    assert exists is True

    # Delete the file
    await storage.delete(key)

    # Check file no longer exists
    exists_after = await storage.exists(key)
    assert exists_after is False


@pytest.mark.asyncio
async def test_markitdown_converter():
    """Test markitdown parser."""
    from hindsight_api.engine.parsers import MarkitdownParser

    parser = MarkitdownParser()

    # Test simple text file
    text_content = b"This is a test document.\nWith multiple lines."
    result = await parser.convert(text_content, "test.txt")

    assert isinstance(result, str)
    assert len(result) > 0
    assert "test document" in result.lower() or "multiple lines" in result.lower()


@pytest.mark.asyncio
async def test_converter_registry():
    """Test file parser registry."""
    from hindsight_api.engine.parsers import FileParserRegistry, MarkitdownParser

    registry = FileParserRegistry()
    parser = MarkitdownParser()
    registry.register(parser)

    # Test get by name
    retrieved = registry.get_parser("markitdown", "test.txt")
    assert retrieved is parser

    # Test auto-detection
    auto = registry.get_parser(None, "test.pdf")
    assert auto is parser

    # Test unsupported format
    with pytest.raises(ValueError, match="No parser found"):
        registry.get_parser(None, "test.xyz")


@pytest.mark.asyncio
async def test_file_conversion_creates_separate_retain_operation(memory_no_llm_verify, sample_txt_content):
    """Test that file conversion and retain are two separate async operations.

    The file_convert_retain task should:
    1. Convert the file to markdown
    2. In a single transaction: create a separate 'retain' operation AND mark itself as 'completed'
    3. Free the worker slot immediately after conversion

    The retain then runs as its own task. This prevents deadlocks where file conversion
    tasks hold worker slots while waiting for inline retain to finish.
    """
    from hindsight_api.models import RequestContext

    bank_id = "test_file_two_phase_bank"

    context = RequestContext(internal=True)
    await memory_no_llm_verify.get_bank_profile(bank_id, request_context=context)

    class MockFile:
        def __init__(self, content, filename, content_type):
            self.content = content
            self.filename = filename
            self.content_type = content_type

        async def read(self):
            return self.content

    mock_file = MockFile(sample_txt_content, "test.txt", "text/plain")

    file_items = [
        {
            "file": mock_file,
            "document_id": "test_doc_two_phase",
            "context": "test context",
            "metadata": {"source": "test"},
            "tags": ["test_tag"],
            "timestamp": None,
        }
    ]

    result = await memory_no_llm_verify.submit_async_file_retain(
        bank_id=bank_id,
        file_items=file_items,
        parser="markitdown",
        document_tags=["two_phase_test"],
        request_context=context,
    )

    assert "operation_ids" in result
    assert len(result["operation_ids"]) == 1
    convert_operation_id = result["operation_ids"][0]

    import asyncio

    await asyncio.sleep(0.1)

    pool = await memory_no_llm_verify._get_pool()
    from hindsight_api.engine.memory_engine import get_current_schema

    schema = get_current_schema()

    async with pool.acquire() as conn:
        # 1. The file_convert_retain operation must be completed
        convert_op = await conn.fetchrow(
            f"SELECT status, operation_type FROM {schema}.async_operations WHERE operation_id = $1",
            convert_operation_id,
        )
        assert convert_op is not None
        assert convert_op["operation_type"] == "file_convert_retain"
        assert convert_op["status"] == "completed", (
            f"file_convert_retain should be 'completed' after conversion, got '{convert_op['status']}'"
        )

        # 2. A separate retain operation must have been created
        retain_op = await conn.fetchrow(
            f"""
            SELECT status, operation_type
            FROM {schema}.async_operations
            WHERE bank_id = $1 AND operation_type = 'retain' AND operation_id != $2
            """,
            bank_id,
            convert_operation_id,
        )
        assert retain_op is not None, "A separate 'retain' operation should have been created by file conversion"
        # With SyncTaskBackend the retain runs immediately, so it should be completed
        assert retain_op["status"] == "completed"

        # 3. The document should exist with file metadata and retained content
        doc = await conn.fetchrow(
            f"""
            SELECT id, original_text, file_original_name, file_content_type
            FROM {schema}.documents
            WHERE id = $1 AND bank_id = $2
            """,
            "test_doc_two_phase",
            bank_id,
        )

    assert doc is not None
    assert doc["file_original_name"] == "test.txt"
    assert doc["file_content_type"] == "text/plain"
    assert doc["original_text"] is not None
    assert len(doc["original_text"]) > 0


@pytest.mark.asyncio
async def test_file_conversion_failure_sets_status_to_failed(memory_no_llm_verify, sample_txt_content):
    """Test that when file conversion fails, the operation status is set to 'failed' not 'completed'."""
    from hindsight_api.engine.parsers.base import FileParser
    from hindsight_api.models import RequestContext

    bank_id = "test_file_failure_bank"

    # Create a mock parser that always fails
    class FailingParser(FileParser):
        """Mock parser that raises an error."""

        async def convert(self, file_data: bytes, filename: str) -> str:
            # Simulate conversion failure
            raise RuntimeError(f"Failed to convert '{filename}': Mock conversion error")

        def supports(self, filename: str, content_type: str | None = None) -> bool:
            return filename.endswith(".fail")

        def name(self) -> str:
            return "failing_converter"

    # Register the failing parser
    failing_converter = FailingParser()
    memory_no_llm_verify._parser_registry.register(failing_converter)

    # Create bank
    context = RequestContext(internal=True)
    await memory_no_llm_verify.get_bank_profile(bank_id, request_context=context)

    # Create mock file
    class MockFile:
        def __init__(self, content, filename, content_type):
            self.content = content
            self.filename = filename
            self.content_type = content_type

        async def read(self):
            return self.content

    mock_file = MockFile(sample_txt_content, "test.fail", "application/octet-stream")

    file_items = [
        {
            "file": mock_file,
            "document_id": "test_doc_fail",
            "context": None,
            "metadata": {},
            "tags": [],
            "timestamp": None,
        }
    ]

    # Submit async file retain with failing parser
    result = await memory_no_llm_verify.submit_async_file_retain(
        bank_id=bank_id,
        file_items=file_items,
        parser="failing_converter",
        document_tags=None,
        request_context=context,
    )

    assert "operation_ids" in result
    assert len(result["operation_ids"]) == 1
    operation_id = result["operation_ids"][0]

    # Wait for async processing (with SyncTaskBackend, this is immediate)
    import asyncio

    await asyncio.sleep(0.2)

    # Check operation status - should be 'failed' not 'completed'
    pool = await memory_no_llm_verify._get_pool()
    from hindsight_api.engine.memory_engine import get_current_schema

    async with pool.acquire() as conn:
        operation = await conn.fetchrow(
            f"""
            SELECT status, error_message
            FROM {get_current_schema()}.async_operations
            WHERE operation_id = $1
            """,
            operation_id,
        )

    assert operation is not None, f"Operation {operation_id} not found"
    assert operation["status"] == "failed", f"Expected status 'failed' but got '{operation['status']}'"
    assert operation["error_message"] is not None
    assert "Mock conversion error" in operation["error_message"]
    assert "test.fail" in operation["error_message"]
