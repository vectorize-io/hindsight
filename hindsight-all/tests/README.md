# Hindsight Integration Tests

This directory contains integration tests for the Hindsight all-in-one package.

## Test Overview

### `test_server_integration.py`

Comprehensive integration tests that verify the complete workflow:

1. **`test_server_context_manager_basic_workflow`**: Main integration test that:
   - Starts the Hindsight server using a context manager
   - Creates a memory bank with background information
   - Stores multiple memories (both single and batch operations)
   - Recalls memories based on different queries (programming preferences, ML topics)
   - Reflects (generates contextual answers) multiple times with different contexts
   - Automatically stops the server on context exit

2. **`test_server_manual_start_stop`**: Tests explicit server lifecycle management without context managers

3. **`test_server_with_client_context_manager`**: Tests nested context managers for both server and client

## Running the Tests

### Prerequisites

1. Install the hindsight package with test dependencies:
   ```bash
   cd hindsight
   uv pip install -e ".[test]"
   ```

2. Set up your LLM credentials in the `.env` file at the project root:
   ```bash
   HINDSIGHT_API_LLM_PROVIDER=groq
   HINDSIGHT_API_LLM_API_KEY=your-api-key
   HINDSIGHT_API_LLM_MODEL=openai/gpt-oss-20b
   ```

### Run All Tests

```bash
cd hindsight
source ../.env
export HINDSIGHT_LLM_PROVIDER=$HINDSIGHT_API_LLM_PROVIDER
export HINDSIGHT_LLM_API_KEY=$HINDSIGHT_API_LLM_API_KEY
export HINDSIGHT_LLM_MODEL=$HINDSIGHT_API_LLM_MODEL
pytest tests/ -v
```

**Note on Parallel Execution**: These tests use embedded PostgreSQL (`pg0`), which is a singleton that cannot be shared across pytest-xdist worker processes. Therefore, these tests must run sequentially. Do not use `pytest -n` (parallel workers) with these tests.

**Why Random `bank_id` Values?**: Each test generates a unique `bank_id` using UUID. This provides several benefits:
- **Clean test isolation**: Tests don't interfere with each other's data
- **Repeatable runs**: Tests can be run multiple times without cleanup
- **Debugging**: Easy to identify which test created which data
- **Future-ready**: If you switch from `pg0` to a real PostgreSQL instance, these tests could run in parallel

### Run Specific Test

```bash
pytest tests/test_server_integration.py::test_server_context_manager_basic_workflow -v
```

### Run with Verbose Output

```bash
pytest tests/ -v -s
```

The `-s` flag shows print statements, which is useful to see the test progress.

### Run with Timeout

These tests involve LLM calls which can take time:

```bash
pytest tests/ --timeout=300
```

## Test Configuration

The tests use:
- **Embedded PostgreSQL** (`pg0`) for the database - no external database required
- **LLM provider** configured via environment variables
- **Automatic port allocation** for the server (no port conflicts)

## Expected Behavior

When tests run successfully, you should see:
1. Server starting on a random available port
2. Memory bank creation with background information
3. Multiple memories being stored (retain operations)
4. Memories being recalled based on different queries
5. Multiple reflection responses with contextual answers
6. Server automatically stopping (for context manager tests)

## Sample Output

The main test workflow demonstrates:
- **Step 1**: Create memory bank
- **Step 2**: Store 5 memories (3 individual + 2 batch)
- **Step 3**: Recall memories about programming preferences
- **Step 4**: Recall memories about machine learning
- **Step 5**: Reflect on tool recommendations
- **Step 6**: Reflect with additional context about framework choices
- **Step 7**: Server auto-stops via context manager

## Skipping Tests

Tests will automatically skip if:
- `HINDSIGHT_LLM_API_KEY` is not set

## Troubleshooting

### Test hangs or times out
- Increase timeout: `pytest tests/ --timeout=600`
- Check your LLM API key is valid
- Verify network connectivity to LLM provider

### Database errors
- The tests use embedded PostgreSQL (`pg0`) which should handle cleanup automatically
- If you see "database system is shutting down" errors, wait a few seconds and retry
- Between test runs, the embedded database needs time to properly shut down

### Port conflicts
- Tests automatically find free ports, so conflicts should be rare
- If you see port binding errors, check for other processes using high-numbered ports
