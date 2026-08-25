"""ONNX-based embedding model using MiniLM-L6-v2."""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

# Model configuration
MODEL_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model.onnx"
TOKENIZER_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json"
# SHA256 checksums pinned to commit c9745ed1d9f207416be6d2e6f8de32d1f16199bf
# Recompute with: Get-FileHash <file> -Algorithm SHA256  (PowerShell)
#             or: sha256sum <file>                        (Linux/macOS)
MODEL_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
TOKENIZER_SHA256 = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256


def get_cache_dir() -> Path:
    """Get the cache directory for model files."""
    cache_dir = Path.home() / ".cache" / "semantic_clusterer"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _sha256(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(file_path: Path, expected_hash: str) -> None:
    """Verify a file's SHA256 checksum, raising ValueError on mismatch."""
    actual = _sha256(file_path)
    if actual != expected_hash:
        raise ValueError(
            f"Checksum mismatch for {file_path.name}\n"
            f"  expected: {expected_hash}\n"
            f"  got:      {actual}"
        )


def _download_once(url: str, target_path: Path, show_progress: bool) -> None:
    """Unconditionally download *url* to *target_path* (atomic via .tmp)."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    filename = target_path.name

    if not show_progress:
        urllib.request.urlretrieve(url, temp_path)
        temp_path.replace(target_path)
        return

    # Professional progress bar (inspired by huggingface/sentence-transformers)
    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "━" * filled + "╺" + "─" * (bar_len - filled - 1) if filled < bar_len else "━" * bar_len
            sys.stderr.write(
                f"\r  Downloading {filename}: {pct:5.1f}%|{bar}| {downloaded_mb:.1f}/{total_mb:.1f} MB"
            )
            sys.stderr.flush()
        else:
            downloaded_mb = downloaded / (1024 * 1024)
            sys.stderr.write(f"\r  Downloading {filename}: {downloaded_mb:.1f} MB")
            sys.stderr.flush()

    sys.stderr.write(f"  Downloading {filename}...\n")
    urllib.request.urlretrieve(url, temp_path, reporthook=_reporthook)
    sys.stderr.write("\n")
    temp_path.replace(target_path)


def download_file(
    url: str,
    target_path: Path,
    expected_sha256: Optional[str] = None,
    show_progress: bool = True,
) -> None:
    """Download a file from *url* to *target_path* and verify its SHA256.

    If the file already exists its checksum is verified first.  A corrupt or
    tampered file is automatically deleted and re-downloaded exactly once
    before a final verification is performed.

    Args:
        url: Source URL to download from.
        target_path: Destination path on disk.
        expected_sha256: Lowercase hex SHA256 digest to verify against.
            Pass ``None`` to skip verification.
        show_progress: Whether to print download progress messages.
    """
    if target_path.exists():
        if expected_sha256 is None:
            return
        try:
            _verify_sha256(target_path, expected_sha256)
            return  # already present and valid
        except ValueError:
            # Corrupt / outdated cached file — delete and re-download
            target_path.unlink(missing_ok=True)

    # First download attempt
    _download_once(url, target_path, show_progress)

    if expected_sha256 is not None:
        try:
            _verify_sha256(target_path, expected_sha256)
        except ValueError:
            # Auto-recovery: wipe and retry once
            target_path.unlink(missing_ok=True)
            _download_once(url, target_path, show_progress)
            _verify_sha256(target_path, expected_sha256)  # hard fail if still wrong


class OnnxEmbedder:
    """ONNX-based text embedder using all-MiniLM-L6-v2.
    
    This embedder uses ONNX Runtime for fast CPU inference without
    heavy PyTorch dependencies. The model is automatically downloaded
    on first use.
    
    Attributes:
        batch_size: Number of texts to process in each batch.
        normalize: Whether to L2-normalize output embeddings.
        verbose: Whether to print progress messages.
    """

    def __init__(
        self,
        batch_size: int = 64,
        normalize: bool = True,
        verbose: bool = False,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the ONNX embedder.
        
        Args:
            batch_size: Batch size for inference.
            normalize: Whether to L2-normalize embeddings.
            verbose: Whether to print progress messages.
            cache_dir: Directory to cache model files. Defaults to ~/.cache/semantic_clusterer/
        """
        self.batch_size = batch_size
        self.normalize = normalize
        self.verbose = verbose
        self.cache_dir = cache_dir or get_cache_dir()

        self._session = None
        self._tokenizer = None

    def _ensure_model_loaded(self) -> None:
        """Ensure the ONNX model and tokenizer are loaded."""
        if self._session is not None:
            return

        # Download model if needed
        model_path = self.cache_dir / "all-MiniLM-L6-v2" / "model.onnx"
        tokenizer_path = self.cache_dir / "all-MiniLM-L6-v2" / "tokenizer.json"

        try:
            download_file(MODEL_URL, model_path, expected_sha256=MODEL_SHA256, show_progress=True)
            download_file(TOKENIZER_URL, tokenizer_path, expected_sha256=TOKENIZER_SHA256, show_progress=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download ONNX model files. Check network access or pre-populate the cache."
            ) from exc

        # Load ONNX model
        import onnxruntime as ort

        # Configure session and hardware acceleration
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = os.cpu_count() or 4
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Automatically use any available GPU/NPU execution provider, falling back to CPU
        available_providers = ort.get_available_providers()
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=available_providers
        )

        # Load tokenizer
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self._tokenizer.enable_padding(length=MAX_SEQ_LENGTH)

    def _mean_pooling(
        self,
        token_embeddings: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """Apply mean pooling to token embeddings.
        
        Args:
            token_embeddings: Token-level embeddings of shape (batch, seq_len, hidden_dim).
            attention_mask: Attention mask of shape (batch, seq_len).
            
        Returns:
            Sentence embeddings of shape (batch, hidden_dim).
        """
        # Expand attention mask to match embedding dimensions
        mask_expanded = np.expand_dims(attention_mask, axis=-1)
        mask_expanded = np.broadcast_to(mask_expanded, token_embeddings.shape)

        # Sum embeddings and divide by token count
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)

        return sum_embeddings / sum_mask

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def embed(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        *,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> np.ndarray:
        """Generate embeddings for texts.

        Args:
            texts: Text strings to embed.
            batch_size: Override instance batch_size if provided.
            progress_callback: Optional callable invoked once per batch with
                the batch's text count. Used by the pipeline to drive a
                single, smooth progress bar instead of opening a competing
                bar inside the embedder.

        Returns:
            Numpy array of shape (n_texts, 384).
        """
        if not texts:
            return np.array([]).reshape(0, EMBEDDING_DIM)

        self._ensure_model_loaded()

        effective_batch_size = batch_size if batch_size is not None else self.batch_size

        # Internal fallback bar — only shown when the caller did not provide
        # a progress_callback AND we have multiple batches to process AND
        # tqdm is importable. The pipeline always supplies a callback, so
        # this branch only fires for direct/manual OnnxEmbedder use.
        use_internal_bar = progress_callback is None and len(texts) > effective_batch_size
        internal_bar = None
        if use_internal_bar:
            try:
                from tqdm import tqdm  # local import — avoid hard dep at module load
                n_batches = (len(texts) + effective_batch_size - 1) // effective_batch_size
                internal_bar = tqdm(
                    total=len(texts),
                    desc="  embedding",
                    ncols=80,
                    unit="",
                    bar_format=(
                        "  {desc:<11} {bar} {percentage:3.0f}%  "
                        "{n_fmt}/{total_fmt}  [{elapsed}]"
                    ),
                    ascii=" ▏▎▍▌▋▊▉█",
                    file=sys.stderr,
                    leave=False,
                    mininterval=0.05,
                    miniters=1,
                )
            except ImportError:
                internal_bar = None

        all_embeddings = []

        try:
            for i in range(0, len(texts), effective_batch_size):
                batch_texts = texts[i:i + effective_batch_size]

                # Tokenize
                encoded = self._tokenizer.encode_batch(batch_texts)

                input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
                attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
                token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

                # Run inference
                outputs = self._session.run(
                    None,
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                    }
                )

                # Mean pooling
                token_embeddings = outputs[0]  # Shape: (batch, seq_len, hidden_dim)
                embeddings = self._mean_pooling(token_embeddings, attention_mask)
                all_embeddings.append(embeddings)

                # Report this batch's progress
                if progress_callback is not None:
                    try:
                        progress_callback(len(batch_texts))
                    except Exception:
                        # Never let a UI hiccup break embedding
                        pass
                if internal_bar is not None:
                    internal_bar.update(len(batch_texts))
        finally:
            if internal_bar is not None:
                try:
                    internal_bar.close()
                except Exception:
                    pass

        embeddings = np.vstack(all_embeddings)

        if self.normalize:
            embeddings = self._normalize_embeddings(embeddings)

        return embeddings.astype(np.float32)
