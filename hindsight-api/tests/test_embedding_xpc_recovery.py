"""
Tests for XPC error recovery in LocalSTEmbeddings.

This tests the automatic reinitialization of the embedding model when
XPC connection errors occur on macOS (common in long-running daemon processes).
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.engine.embeddings import LocalSTEmbeddings


class TestXPCErrorRecovery:
    """Tests for XPC error detection and recovery in LocalSTEmbeddings."""

    @pytest.fixture
    def embeddings(self):
        """Create a LocalSTEmbeddings instance."""
        return LocalSTEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def test_is_xpc_error_detection(self, embeddings):
        """Test that XPC errors are correctly detected."""
        # Test various XPC error message formats
        xpc_error = Exception("Compiler encountered XPC_ERROR_CONNECTION_INVALID (is the OS shutting down?)")
        assert embeddings._is_xpc_error(xpc_error)

        xpc_error2 = Exception("XPC error occurred")
        assert embeddings._is_xpc_error(xpc_error2)

        # Test that non-XPC errors are not detected
        normal_error = Exception("Some other error")
        assert not embeddings._is_xpc_error(normal_error)

    @pytest.mark.asyncio
    async def test_encode_with_xpc_recovery(self, embeddings):
        """Test that encode() recovers from XPC errors by reinitializing."""
        # Initialize the embeddings
        await embeddings.initialize()

        # Track calls to reinitialize
        reinit_called = False
        original_reinit = embeddings._reinitialize_model_sync

        def track_reinit():
            nonlocal reinit_called
            reinit_called = True
            original_reinit()

        # Track encode attempts
        encode_attempts = []
        original_encode = embeddings._model.encode

        def mock_encode(*args, **kwargs):
            encode_attempts.append(1)
            # Only fail on first attempt
            if len(encode_attempts) == 1:
                raise RuntimeError("Compiler encountered XPC_ERROR_CONNECTION_INVALID (is the OS shutting down?)")
            else:
                # After reinit: succeed
                return original_encode(*args, **kwargs)

        # Mock the initial encode to fail, reinit happens, then new model succeeds
        with patch.object(embeddings, "_reinitialize_model_sync", side_effect=track_reinit):
            with patch.object(embeddings._model, "encode", side_effect=mock_encode):
                # This should trigger XPC error on first attempt, then recover and succeed
                result = embeddings.encode(["test text"])

                # Verify we got a result
                assert result is not None
                assert len(result) == 1
                assert len(result[0]) > 0  # Should have embedding vector
                assert reinit_called  # Should have reinitialized
                assert len(encode_attempts) >= 1  # At least one attempt was made

    @pytest.mark.asyncio
    async def test_encode_fails_on_non_xpc_error(self, embeddings):
        """Test that encode() does not retry for non-XPC errors."""
        # Initialize the embeddings
        await embeddings.initialize()

        # Create a mock that raises a non-XPC error
        def mock_encode(*args, **kwargs):
            raise RuntimeError("Some other error")

        # Patch the model's encode method
        with patch.object(embeddings._model, "encode", side_effect=mock_encode):
            # This should fail without retry
            with pytest.raises(RuntimeError) as exc_info:
                embeddings.encode(["test text"])

            assert "Some other error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_reinitialize_clears_model(self, embeddings):
        """Test that _reinitialize_model_sync properly clears and reinits the model."""
        # Initialize the embeddings
        await embeddings.initialize()

        original_model = embeddings._model
        assert original_model is not None

        # Reinitialize
        embeddings._reinitialize_model_sync()

        # Model should be reinitialized (new instance)
        assert embeddings._model is not None
        assert embeddings._model is not original_model

        # Should still work
        result = embeddings.encode(["test"])
        assert len(result) == 1
        assert len(result[0]) > 0

    @pytest.mark.asyncio
    async def test_xpc_recovery_exhausts_retries(self, embeddings):
        """Test that XPC recovery gives up after max retries."""
        # Initialize the embeddings
        await embeddings.initialize()

        # Track reinit calls
        reinit_count = 0
        original_reinit = embeddings._reinitialize_model_sync

        def track_and_fail_reinit():
            nonlocal reinit_count
            reinit_count += 1
            # Call original reinit, but the new model will also be mocked to fail
            original_reinit()
            # After reinit, patch the new model too
            embeddings._model.encode = MagicMock(
                side_effect=RuntimeError("Compiler encountered XPC_ERROR_CONNECTION_INVALID")
            )

        # Mock that always raises XPC error
        embeddings._model.encode = MagicMock(
            side_effect=RuntimeError("Compiler encountered XPC_ERROR_CONNECTION_INVALID")
        )

        with patch.object(embeddings, "_reinitialize_model_sync", side_effect=track_and_fail_reinit):
            # Should try once, reinitialize, try again, and fail
            with pytest.raises(RuntimeError) as exc_info:
                embeddings.encode(["test"])

            assert "XPC_ERROR_CONNECTION_INVALID" in str(exc_info.value)
            assert reinit_count == 1  # Should have tried to reinitialize once
