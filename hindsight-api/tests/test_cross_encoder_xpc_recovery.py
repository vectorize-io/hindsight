"""
Tests for XPC error recovery in LocalSTCrossEncoder.

This tests the automatic reinitialization of the cross-encoder model when
XPC connection errors occur on macOS (common in long-running daemon processes).
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.engine.cross_encoder import LocalSTCrossEncoder


class TestCrossEncoderXPCErrorRecovery:
    """Tests for XPC error detection and recovery in LocalSTCrossEncoder."""

    @pytest.fixture
    def cross_encoder(self):
        """Create a LocalSTCrossEncoder instance."""
        return LocalSTCrossEncoder(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")

    def test_is_xpc_error_detection(self, cross_encoder):
        """Test that XPC errors are correctly detected."""
        # Test various XPC error message formats
        xpc_error = Exception("Compiler encountered XPC_ERROR_CONNECTION_INVALID (is the OS shutting down?)")
        assert cross_encoder._is_xpc_error(xpc_error)

        xpc_error2 = Exception("XPC error occurred")
        assert cross_encoder._is_xpc_error(xpc_error2)

        # Test that non-XPC errors are not detected
        normal_error = Exception("Some other error")
        assert not cross_encoder._is_xpc_error(normal_error)

    @pytest.mark.asyncio
    async def test_predict_with_xpc_recovery(self, cross_encoder):
        """Test that predict() recovers from XPC errors by reinitializing."""
        # Initialize the cross-encoder
        await cross_encoder.initialize()

        # Track calls to reinitialize
        reinit_called = False
        original_reinit = cross_encoder._reinitialize_model_sync

        def track_reinit():
            nonlocal reinit_called
            reinit_called = True
            original_reinit()

        # Track predict attempts
        predict_attempts = []
        original_predict = cross_encoder._model.predict

        def mock_predict(*args, **kwargs):
            predict_attempts.append(1)
            # Only fail on first attempt
            if len(predict_attempts) == 1:
                raise RuntimeError("Compiler encountered XPC_ERROR_CONNECTION_INVALID (is the OS shutting down?)")
            else:
                # After reinit: succeed
                return original_predict(*args, **kwargs)

        # Mock the initial predict to fail, reinit happens, then new model succeeds
        with patch.object(cross_encoder, "_reinitialize_model_sync", side_effect=track_reinit):
            with patch.object(cross_encoder._model, "predict", side_effect=mock_predict):
                # This should trigger XPC error on first attempt, then recover and succeed
                result = await cross_encoder.predict([("query", "document")])

                # Verify we got a result
                assert result is not None
                assert len(result) == 1
                assert isinstance(result[0], float)
                assert reinit_called  # Should have reinitialized
                assert len(predict_attempts) >= 1  # At least one attempt was made

    @pytest.mark.asyncio
    async def test_predict_fails_on_non_xpc_error(self, cross_encoder):
        """Test that predict() does not retry for non-XPC errors."""
        # Initialize the cross-encoder
        await cross_encoder.initialize()

        # Create a mock that raises a non-XPC error
        def mock_predict(*args, **kwargs):
            raise RuntimeError("Some other error")

        # Patch the model's predict method
        with patch.object(cross_encoder._model, "predict", side_effect=mock_predict):
            # This should fail without retry
            with pytest.raises(RuntimeError) as exc_info:
                await cross_encoder.predict([("query", "document")])

            assert "Some other error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_reinitialize_clears_model(self, cross_encoder):
        """Test that _reinitialize_model_sync properly clears and reinits the model."""
        # Initialize the cross-encoder
        await cross_encoder.initialize()

        original_model = cross_encoder._model
        assert original_model is not None

        # Reinitialize
        cross_encoder._reinitialize_model_sync()

        # Model should be reinitialized (new instance)
        assert cross_encoder._model is not None
        assert cross_encoder._model is not original_model

        # Should still work
        result = await cross_encoder.predict([("test query", "test document")])
        assert len(result) == 1
        assert isinstance(result[0], float)

    @pytest.mark.asyncio
    async def test_xpc_recovery_exhausts_retries(self, cross_encoder):
        """Test that XPC recovery gives up after max retries."""
        # Initialize the cross-encoder
        await cross_encoder.initialize()

        # Track reinit calls
        reinit_count = 0
        original_reinit = cross_encoder._reinitialize_model_sync

        def track_and_fail_reinit():
            nonlocal reinit_count
            reinit_count += 1
            # Call original reinit, but the new model will also be mocked to fail
            original_reinit()
            # After reinit, patch the new model too
            cross_encoder._model.predict = MagicMock(
                side_effect=RuntimeError("Compiler encountered XPC_ERROR_CONNECTION_INVALID")
            )

        # Mock that always raises XPC error
        cross_encoder._model.predict = MagicMock(
            side_effect=RuntimeError("Compiler encountered XPC_ERROR_CONNECTION_INVALID")
        )

        with patch.object(cross_encoder, "_reinitialize_model_sync", side_effect=track_and_fail_reinit):
            # Should try once, reinitialize, try again, and fail
            with pytest.raises(Exception) as exc_info:
                await cross_encoder.predict([("query", "document")])

            assert "XPC_ERROR_CONNECTION_INVALID" in str(exc_info.value) or "Failed to recover" in str(exc_info.value)
            assert reinit_count == 1  # Should have tried to reinitialize once
