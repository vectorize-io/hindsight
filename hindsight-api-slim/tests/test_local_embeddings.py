from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hindsight_api.engine.embeddings import LocalSTEmbeddings


def test_local_embeddings_use_sentence_transformer_input_types():
    model = MagicMock()
    model.encode.return_value = np.array([[1.0]])
    model.encode_query.return_value = np.array([[2.0]])
    model.encode_document.return_value = np.array([[3.0]])

    embeddings = LocalSTEmbeddings()
    embeddings._model = model
    embeddings._device_type = "mps"

    with patch("hindsight_api.engine.embeddings.release_local_inference_memory") as release:
        assert embeddings.encode(["text"]) == [[1.0]]
        assert embeddings.encode_query(["query"]) == [[2.0]]
        assert embeddings.encode_documents(["document"]) == [[3.0]]

    model.encode.assert_called_once_with(["text"], convert_to_numpy=True, show_progress_bar=False)
    model.encode_query.assert_called_once_with(["query"], convert_to_numpy=True, show_progress_bar=False)
    model.encode_document.assert_called_once_with(["document"], convert_to_numpy=True, show_progress_bar=False)
    assert release.call_count == 3


def test_sentence_transformer_exposes_input_type_entry_points():
    """Guard the >=5.0 floor in local-ml.

    The test above drives a MagicMock, so it passes on any sentence-transformers
    version while the real one raises AttributeError at the first encode. These
    entry points only exist from 5.0 onwards; assert against the real class so a
    loosened pin fails here instead of at recall time.
    """
    sentence_transformers = pytest.importorskip("sentence_transformers")

    assert hasattr(sentence_transformers.SentenceTransformer, "encode_query")
    assert hasattr(sentence_transformers.SentenceTransformer, "encode_document")
