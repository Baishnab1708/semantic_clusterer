"""Tests for embedding adapters."""


import numpy as np
import pytest

from semantic_clusterer.embedding.adapters import (
    CallableAdapter,
    EncodeAdapter,
    LangchainAdapter,
    NativeEmbedAdapter,
    normalize_embedding_model,
    validate_embeddings,
)


class TestEncodeAdapter:
    """Tests for EncodeAdapter."""

    def test_wraps_encode(self):
        """Wraps .encode() to .embed()."""
        class EncodeModel:
            def encode(self, texts, batch_size=64):
                return np.ones((len(texts), 384))

        adapter = EncodeAdapter(EncodeModel())
        result = adapter.embed(["hello", "world"])
        assert result.shape == (2, 384)

    def test_converts_to_numpy(self):
        """Converts output to numpy array."""
        class EncodeModel:
            def encode(self, texts, batch_size=64):
                return [[1, 2, 3] for _ in texts]

        adapter = EncodeAdapter(EncodeModel())
        result = adapter.embed(["a"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)

    def test_batch_size_passed_to_model(self):
        """Passes batch_size to model."""
        received = []

        class EncodeModel:
            def encode(self, texts, batch_size=64):
                received.append(batch_size)
                return np.ones((len(texts), 10))

        adapter = EncodeAdapter(EncodeModel())
        adapter.embed(["a", "b"], batch_size=128)
        assert received[0] == 128


class TestLangchainAdapter:
    """Tests for LangchainAdapter."""

    def test_wraps_embed_documents(self):
        """Wraps .embed_documents() to .embed()."""
        class LangchainModel:
            def embed_documents(self, texts):
                return [[0.5] * 768 for _ in texts]

        adapter = LangchainAdapter(LangchainModel())
        result = adapter.embed(["text1", "text2", "text3"])
        assert result.shape == (3, 768)

    def test_converts_to_list(self):
        """Converts texts to list."""
        class LangchainModel:
            def embed_documents(self, texts):
                assert isinstance(texts, list)
                return np.zeros((len(texts), 10))

        adapter = LangchainAdapter(LangchainModel())
        result = adapter.embed(("a", "b"))
        assert result.shape == (2, 10)

    def test_batch_size_ignored(self):
        """Ignores batch_size (LangChain handles batching)."""
        call_count = [0]

        class LangchainModel:
            def embed_documents(self, texts):
                call_count[0] += 1
                return [[0.5] * 10 for _ in texts]

        adapter = LangchainAdapter(LangchainModel())
        result = adapter.embed(["a", "b", "c", "d"], batch_size=2)
        assert call_count[0] == 1
        assert result.shape == (4, 10)


class TestCallableAdapter:
    """Tests for CallableAdapter."""

    def test_wraps_callable(self):
        """Wraps callable functions."""
        def my_embed(texts):
            return np.random.randn(len(texts), 256)

        adapter = CallableAdapter(my_embed)
        result = adapter.embed(["a", "b"])
        assert result.shape == (2, 256)

    def test_lambda(self):
        """Works with lambda functions."""
        adapter = CallableAdapter(lambda texts: np.eye(len(texts)))
        result = adapter.embed(["a", "b", "c"])
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result, np.eye(3))

    def test_manual_chunking_protects_api(self):
        """Chunks texts to protect custom APIs."""
        call_chunks = []

        def my_embed(texts):
            call_chunks.append(len(texts))
            return np.ones((len(texts), 10))

        adapter = CallableAdapter(my_embed)
        # 5 texts with batch_size=2 should result in 3 calls (2+2+1)
        result = adapter.embed(["a", "b", "c", "d", "e"], batch_size=2)

        assert result.shape == (5, 10)
        assert call_chunks == [2, 2, 1]

    def test_single_batch_no_chunking(self):
        """No chunking when within batch_size."""
        call_count = [0]

        def my_embed(texts):
            call_count[0] += 1
            return np.ones((len(texts), 10))

        adapter = CallableAdapter(my_embed)
        result = adapter.embed(["a", "b"], batch_size=64)
        assert call_count[0] == 1
        assert result.shape == (2, 10)

    def test_chunking_produces_correct_output(self):
        """Chunked embeddings concatenate correctly."""
        def my_embed(texts):
            return np.array([[i] * 5 for i, _ in enumerate(texts)])

        adapter = CallableAdapter(my_embed)
        result = adapter.embed(["a", "b", "c"], batch_size=2)

        assert result.shape == (3, 5)
        np.testing.assert_array_equal(result[0], [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(result[1], [1, 1, 1, 1, 1])
        np.testing.assert_array_equal(result[2], [0, 0, 0, 0, 0])


class TestNormalizeEmbeddingModel:
    """Tests for normalize_embedding_model."""

    def test_none_raises_typeerror(self):
        """None raises TypeError."""
        with pytest.raises(TypeError, match="cannot be None"):
            normalize_embedding_model(None)

    def test_embed_model_uses_native_adapter(self):
        """Models with .embed() use NativeEmbedAdapter."""
        class EmbedModel:
            def embed(self, texts):
                return np.zeros((len(texts), 10))

        model = EmbedModel()
        result = normalize_embedding_model(model)
        assert isinstance(result, NativeEmbedAdapter)
        assert result.model is model

    def test_encode_model_uses_encode_adapter(self):
        """Models with .encode() use EncodeAdapter."""
        class EncodeModel:
            def encode(self, texts):
                return np.zeros((len(texts), 10))

        result = normalize_embedding_model(EncodeModel())
        assert isinstance(result, EncodeAdapter)

    def test_langchain_model_uses_langchain_adapter(self):
        """Models with .embed_documents() use LangchainAdapter."""
        class LangchainModel:
            def embed_documents(self, texts):
                return [[0.0] * 10 for _ in texts]

        result = normalize_embedding_model(LangchainModel())
        assert isinstance(result, LangchainAdapter)

    def test_callable_uses_callable_adapter(self):
        """Callables use CallableAdapter."""
        def my_fn(texts):
            return np.zeros((len(texts), 10))

        result = normalize_embedding_model(my_fn)
        assert isinstance(result, CallableAdapter)

    def test_unsupported_raises_typeerror(self):
        """Unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported embedding_model"):
            normalize_embedding_model("not a model")

        with pytest.raises(TypeError, match="Unsupported embedding_model"):
            normalize_embedding_model({"not": "callable"})

        with pytest.raises(TypeError, match="Unsupported embedding_model"):
            normalize_embedding_model([1, 2, 3])

        class EmptyClass:
            pass

        with pytest.raises(TypeError, match="Unsupported embedding_model"):
            normalize_embedding_model(EmptyClass())

    def test_priority_embed_over_encode(self):
        """.embed() takes priority over .encode()."""
        class DualModel:
            def embed(self, texts):
                return np.ones((len(texts), 10))

            def encode(self, texts):
                return np.zeros((len(texts), 10))

        model = DualModel()
        result = normalize_embedding_model(model)
        assert isinstance(result, NativeEmbedAdapter)
        output = result.embed(["test"])
        np.testing.assert_array_equal(output, np.ones((1, 10)))


class TestValidateEmbeddings:
    """Tests for validate_embeddings."""

    def test_valid_embeddings(self):
        """Valid embeddings pass."""
        texts = ["a", "b", "c"]
        embeddings = np.random.randn(3, 384)
        result = validate_embeddings(embeddings, texts)
        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    def test_empty_texts_raises(self):
        """Empty texts raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            validate_embeddings(np.array([]), [])

    def test_empty_texts_with_allow_empty(self):
        """allow_empty=True returns empty array."""
        result = validate_embeddings(np.array([]), [], allow_empty=True)
        assert result.shape == (0, 0)

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        texts = ["a", "b", "c"]
        embeddings = np.random.randn(2, 10)
        with pytest.raises(ValueError, match="mismatch"):
            validate_embeddings(embeddings, texts)

    def test_1d_single_text_reshaped(self):
        """1D embedding for single text is reshaped."""
        texts = ["one"]
        embeddings = np.array([1, 2, 3, 4])  # 1D

        result = validate_embeddings(embeddings, texts)

        assert result.shape == (1, 4)

    def test_1d_multiple_texts_raises(self):
        """1D embedding for multiple texts raises."""
        texts = ["a", "b"]
        embeddings = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match="2D"):
            validate_embeddings(embeddings, texts)

    def test_3d_raises(self):
        """3D embeddings raise ValueError."""
        texts = ["a"]
        embeddings = np.random.randn(1, 10, 10)
        with pytest.raises(ValueError, match="2D"):
            validate_embeddings(embeddings, texts)

    def test_nan_raises(self):
        """NaN values raise ValueError."""
        texts = ["a", "b"]
        embeddings = np.array([[1, 2, np.nan], [4, 5, 6]])
        with pytest.raises(ValueError, match="NaN"):
            validate_embeddings(embeddings, texts)

    def test_inf_raises(self):
        """Inf values raise ValueError."""
        texts = ["a", "b"]
        embeddings = np.array([[1, 2, np.inf], [4, 5, 6]])
        with pytest.raises(ValueError, match="Inf"):
            validate_embeddings(embeddings, texts)

    def test_list_converted_to_numpy(self):
        """List input converts to numpy."""
        texts = ["a", "b"]
        embeddings = [[1, 2], [3, 4]]
        result = validate_embeddings(embeddings, texts)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_converts_to_float32(self):
        """Output is always float32."""
        texts = ["a"]
        embeddings = np.array([[1, 2, 3]], dtype=np.float64)
        result = validate_embeddings(embeddings, texts)
        assert result.dtype == np.float32

    def test_non_numeric_dtype_raises(self):
        """Non-numeric dtype raises TypeError."""
        texts = ["a", "b"]
        embeddings = np.array([["not", "numeric"], ["also", "strings"]], dtype=object)
        with pytest.raises(TypeError, match="numeric"):
            validate_embeddings(embeddings, texts)


class TestIntegrationWithSemanticClusterer:
    """Integration tests."""

    def test_sentence_transformer_style(self):
        """SentenceTransformer-style model works."""
        from semantic_clusterer import SemanticClusterer

        class MockSentenceTransformer:
            def encode(self, texts, batch_size=64):
                return np.random.randn(len(texts), 384).astype(np.float32)

        clusterer = SemanticClusterer(embedding_model=MockSentenceTransformer())
        texts = ["hello world", "goodbye world", "test text"]
        result = clusterer.cluster(texts)

        assert isinstance(result, list)
        all_clustered = sum(len(c) for c in result)
        assert all_clustered <= len(texts)

    def test_langchain_style(self):
        """LangChain-style model works."""
        from semantic_clusterer import SemanticClusterer

        class MockLangchainEmbeddings:
            def embed_documents(self, texts):
                return np.random.randn(len(texts), 1536).tolist()


        clusterer = SemanticClusterer(embedding_model=MockLangchainEmbeddings())
        texts = ["document one", "document two"]
        result = clusterer.cluster(texts)
        assert isinstance(result, list)

    def test_callable_style(self):
        """Callable function works."""
        from semantic_clusterer import SemanticClusterer

        def my_embedder(texts):
            return np.random.randn(len(texts), 256)

        clusterer = SemanticClusterer(embedding_model=my_embedder)
        texts = ["text a", "text b", "text c"]
        result = clusterer.cluster(texts)
        assert isinstance(result, list)

    def test_custom_class_with_embed(self):
        """Custom class with .embed() works."""
        from semantic_clusterer import SemanticClusterer

        class MyEmbedder:
            def embed(self, texts):
                return np.random.randn(len(texts), 512).astype(np.float32)

        clusterer = SemanticClusterer(embedding_model=MyEmbedder())
        texts = ["one", "two", "three"]
        result = clusterer.cluster(texts)
        assert isinstance(result, list)
