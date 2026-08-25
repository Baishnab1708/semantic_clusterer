"""Tests for download_file / SHA256 verification in onnx_model.py.

All tests use pytest's tmp_path fixture and monkeypatching so:
- No real network calls are made.
- No files are left in the real cache directory.
"""

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from semantic_clusterer.embedding.onnx_model import (
    MODEL_SHA256,
    TOKENIZER_SHA256,
    _sha256,
    _verify_sha256,
    download_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


GOOD_DATA = b"hello semantic clusterer"
GOOD_HASH = _sha256_of(GOOD_DATA)
BAD_DATA  = b"corrupted payload"


# ---------------------------------------------------------------------------
# _sha256 helper
# ---------------------------------------------------------------------------

class TestSha256Helper:
    def test_correct_digest(self, tmp_path):
        f = tmp_path / "file.bin"
        _write(f, GOOD_DATA)
        assert _sha256(f) == GOOD_HASH

    def test_different_content_gives_different_digest(self, tmp_path):
        f = tmp_path / "file.bin"
        _write(f, BAD_DATA)
        assert _sha256(f) != GOOD_HASH


# ---------------------------------------------------------------------------
# _verify_sha256
# ---------------------------------------------------------------------------

class TestVerifySha256:
    def test_passes_on_correct_hash(self, tmp_path):
        f = tmp_path / "model.onnx"
        _write(f, GOOD_DATA)
        _verify_sha256(f, GOOD_HASH)          # must not raise

    def test_raises_on_wrong_hash(self, tmp_path):
        f = tmp_path / "model.onnx"
        _write(f, BAD_DATA)
        with pytest.raises(ValueError, match="Checksum mismatch"):
            _verify_sha256(f, GOOD_HASH)

    def test_error_message_contains_expected_and_got(self, tmp_path):
        f = tmp_path / "model.onnx"
        _write(f, BAD_DATA)
        with pytest.raises(ValueError) as exc_info:
            _verify_sha256(f, GOOD_HASH)
        msg = str(exc_info.value)
        assert "expected" in msg
        assert "got" in msg
        assert GOOD_HASH in msg


# ---------------------------------------------------------------------------
# download_file — no network (monkeypatched _download_once)
# ---------------------------------------------------------------------------

class TestDownloadFile:
    """
    We patch `semantic_clusterer.embedding.onnx_model._download_once` so
    no real HTTP request is ever made.  Each fake download just writes the
    bytes we choose.
    """

    def _make_fake_download(self, data: bytes):
        """Return a callable that writes *data* when called as _download_once."""
        def fake_download(url, target_path, show_progress):          # noqa: ANN001
            _write(target_path, data)
        return fake_download

    # --- file does not exist yet ----------------------------------------

    def test_downloads_and_verifies_good_file(self, tmp_path):
        dest = tmp_path / "model.onnx"
        with patch(
            "semantic_clusterer.embedding.onnx_model._download_once",
            side_effect=self._make_fake_download(GOOD_DATA),
        ):
            download_file("http://fake/model.onnx", dest, expected_sha256=GOOD_HASH)
        assert dest.exists()

    def test_raises_if_download_gives_wrong_hash_after_retry(self, tmp_path):
        """Both attempts serve bad data → must raise ValueError."""
        dest = tmp_path / "model.onnx"
        with patch(
            "semantic_clusterer.embedding.onnx_model._download_once",
            side_effect=self._make_fake_download(BAD_DATA),
        ):
            with pytest.raises(ValueError, match="Checksum mismatch"):
                download_file("http://fake/model.onnx", dest, expected_sha256=GOOD_HASH)

    def test_auto_recovery_on_first_bad_then_good(self, tmp_path):
        """First download returns bad bytes, second returns good → should succeed."""
        dest = tmp_path / "model.onnx"
        call_count = {"n": 0}

        def fake_download(url, target_path, show_progress):          # noqa: ANN001
            call_count["n"] += 1
            data = GOOD_DATA if call_count["n"] > 1 else BAD_DATA
            _write(target_path, data)

        with patch(
            "semantic_clusterer.embedding.onnx_model._download_once",
            side_effect=fake_download,
        ):
            download_file("http://fake/model.onnx", dest, expected_sha256=GOOD_HASH)

        assert call_count["n"] == 2, "Should have retried exactly once"
        assert dest.exists()

    def test_no_network_call_when_expected_sha256_is_none(self, tmp_path):
        """No hash → return immediately if file exists, no download."""
        dest = tmp_path / "model.onnx"
        _write(dest, BAD_DATA)            # any content, hash not checked

        with patch(
            "semantic_clusterer.embedding.onnx_model._download_once"
        ) as mock_dl:
            download_file("http://fake/model.onnx", dest, expected_sha256=None)
            mock_dl.assert_not_called()

    # --- file already exists -------------------------------------------

    def test_skips_download_if_file_already_valid(self, tmp_path):
        """File present + correct hash → no network call."""
        dest = tmp_path / "model.onnx"
        _write(dest, GOOD_DATA)

        with patch(
            "semantic_clusterer.embedding.onnx_model._download_once"
        ) as mock_dl:
            download_file("http://fake/model.onnx", dest, expected_sha256=GOOD_HASH)
            mock_dl.assert_not_called()

    def test_redownloads_corrupt_cached_file(self, tmp_path):
        """Cached file has wrong hash → deleted and re-downloaded."""
        dest = tmp_path / "model.onnx"
        _write(dest, BAD_DATA)           # simulate corrupt cache

        with patch(
            "semantic_clusterer.embedding.onnx_model._download_once",
            side_effect=self._make_fake_download(GOOD_DATA),
        ) as mock_dl:
            download_file("http://fake/model.onnx", dest, expected_sha256=GOOD_HASH)
            mock_dl.assert_called_once()

        assert _sha256(dest) == GOOD_HASH

    def test_corrupt_cache_file_is_replaced_not_left_behind(self, tmp_path):
        """After recovery the on-disk file must contain the good payload."""
        dest = tmp_path / "model.onnx"
        _write(dest, BAD_DATA)

        with patch(
            "semantic_clusterer.embedding.onnx_model._download_once",
            side_effect=self._make_fake_download(GOOD_DATA),
        ):
            download_file("http://fake/model.onnx", dest, expected_sha256=GOOD_HASH)

        assert dest.read_bytes() == GOOD_DATA


# ---------------------------------------------------------------------------
# Hardcoded constants sanity checks (not network-dependent)
# ---------------------------------------------------------------------------

class TestHardcodedConstants:
    def test_model_sha256_is_set(self):
        assert MODEL_SHA256 is not None, "MODEL_SHA256 must not be None"

    def test_tokenizer_sha256_is_set(self):
        assert TOKENIZER_SHA256 is not None, "TOKENIZER_SHA256 must not be None"

    def test_model_sha256_is_64_hex_chars(self):
        assert len(MODEL_SHA256) == 64
        assert all(c in "0123456789abcdef" for c in MODEL_SHA256)

    def test_tokenizer_sha256_is_64_hex_chars(self):
        assert len(TOKENIZER_SHA256) == 64
        assert all(c in "0123456789abcdef" for c in TOKENIZER_SHA256)

    def test_model_and_tokenizer_hashes_differ(self):
        assert MODEL_SHA256 != TOKENIZER_SHA256
