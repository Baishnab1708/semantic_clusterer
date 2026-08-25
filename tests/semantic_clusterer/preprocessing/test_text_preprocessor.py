"""Unit tests for ``semantic_clusterer.preprocessing.clean.TextPreprocessor``.

These tests cover the public preprocessing contract end-to-end so we can
trust the index-mapping invariants the rest of the pipeline depends on:

  * Unicode NFKC normalisation
  * lowercase / punctuation toggles
  * ``min_length`` filtering
  * deduplication with stable ``original_to_processed`` mapping
  * non-deduplicated mode
  * missing-value sentinels (None / NaN / pandas.NA)
  * type errors for non-string objects
  * ``preprocess_simple`` round-trip

The release-level smoke checks live in ``test_release_v010.py``; this file
holds the focused, behaviour-by-behaviour tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from semantic_clusterer.preprocessing.clean import TextPreprocessor


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------

class TestConstructionDefaults:
    """Default flags match the documented behaviour."""

    def test_defaults(self):
        p = TextPreprocessor()
        assert p.lowercase is True
        assert p.remove_punctuation is True
        assert p.min_length == 1

    def test_custom_flags(self):
        p = TextPreprocessor(lowercase=False, remove_punctuation=False, min_length=3)
        assert p.lowercase is False
        assert p.remove_punctuation is False
        assert p.min_length == 3


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------

class TestCleaning:
    """The configurable cleaning steps behave as documented."""

    def test_lowercase_default(self):
        p = TextPreprocessor()
        out = p.preprocess_simple(["Hello World"])
        assert out == ["hello world"]

    def test_lowercase_disabled_preserves_case(self):
        p = TextPreprocessor(lowercase=False, remove_punctuation=False)
        out = p.preprocess_simple(["Hello World"])
        assert out == ["Hello World"]

    def test_punctuation_removed_by_default(self):
        p = TextPreprocessor()
        out = p.preprocess_simple(["hello, world!"])
        assert out == ["hello world"]

    def test_punctuation_kept_when_disabled(self):
        p = TextPreprocessor(remove_punctuation=False, lowercase=False)
        out = p.preprocess_simple(["hello, world!"])
        assert out == ["hello, world!"]

    def test_whitespace_collapsed(self):
        p = TextPreprocessor()
        out = p.preprocess_simple(["hello\t  \nworld   "])
        assert out == ["hello world"]

    def test_unicode_nfkc_normalisation(self):
        # Full-width "Ｈｅｌｌｏ" → "Hello" → "hello"
        p = TextPreprocessor()
        out = p.preprocess_simple(["Ｈｅｌｌｏ"])
        assert out == ["hello"]

    def test_combining_characters_normalised(self):
        # "café" written as "cafe\u0301" (e + combining acute) should be
        # NFKC-collapsed to a single composed code point.
        p = TextPreprocessor(remove_punctuation=False)
        out = p.preprocess_simple(["cafe\u0301"])
        assert out == ["café"]

    def test_punctuation_only_becomes_empty_string(self):
        p = TextPreprocessor()
        out = p.preprocess_simple(["..."])
        assert out == [""]


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

class TestMissingValues:
    """``_is_missing`` plus its handling in ``preprocess``."""

    def test_none_detected_as_missing(self):
        p = TextPreprocessor()
        assert p._is_missing(None) is True

    def test_python_nan_detected(self):
        p = TextPreprocessor()
        assert p._is_missing(float("nan")) is True

    def test_numpy_nan_detected(self):
        p = TextPreprocessor()
        assert p._is_missing(np.nan) is True
        assert p._is_missing(np.float32("nan")) is True

    def test_pandas_na_detected(self):
        pd = pytest.importorskip("pandas")
        p = TextPreprocessor()
        assert p._is_missing(pd.NA) is True
        assert p._is_missing(pd.NaT) is True

    def test_finite_floats_not_missing(self):
        p = TextPreprocessor()
        assert p._is_missing(1.5) is False
        assert p._is_missing(0.0) is False

    def test_strings_not_missing(self):
        p = TextPreprocessor()
        assert p._is_missing("hello") is False
        assert p._is_missing("") is False

    def test_clean_returns_none_for_missing(self):
        p = TextPreprocessor()
        assert p._clean_text(None) is None
        assert p._clean_text(float("nan")) is None

    def test_preprocess_simple_returns_none_for_missing(self):
        """``preprocess_simple`` does not filter — None survives in the output."""
        p = TextPreprocessor()
        out = p.preprocess_simple(["hello", None, "world"])
        assert out == ["hello", None, "world"]


# ---------------------------------------------------------------------------
# Type errors for invalid containers
# ---------------------------------------------------------------------------

class TestTypeErrors:
    """Non-string non-missing inputs raise ``TypeError`` with a clear message."""

    @pytest.mark.parametrize(
        "value",
        [
            {"a": 1},
            [1, 2, 3],
            {1, 2, 3},
            (1, 2),
            object(),
        ],
    )
    def test_non_string_raises_type_error(self, value):
        p = TextPreprocessor()
        with pytest.raises(TypeError, match="Expected str"):
            p._clean_text(value)

    def test_int_raises_type_error(self):
        p = TextPreprocessor()
        with pytest.raises(TypeError, match="Expected str"):
            p.preprocess([42])

    def test_bytes_raises_type_error(self):
        p = TextPreprocessor()
        with pytest.raises(TypeError, match="Expected str"):
            p.preprocess([b"bytes"])


# ---------------------------------------------------------------------------
# preprocess() with deduplication
# ---------------------------------------------------------------------------

class TestPreprocessDedup:
    """Deduplicating preprocess returns stable, monotonic index maps."""

    def test_basic_dedup(self):
        p = TextPreprocessor()
        texts = ["apple", "Apple", "banana", "APPLE"]
        processed, o2p, p2o = p.preprocess(texts)

        # Three duplicates collapse to one
        assert processed == ["apple", "banana"]
        assert o2p == {0: 0, 1: 0, 2: 1, 3: 0}
        assert p2o == [0, 2]

    def test_missing_maps_to_minus_one(self):
        p = TextPreprocessor()
        texts = ["hello", None, "world", float("nan")]
        processed, o2p, _ = p.preprocess(texts)

        assert processed == ["hello", "world"]
        assert o2p[1] == -1
        assert o2p[3] == -1
        assert o2p[0] == 0
        assert o2p[2] == 1

    def test_filtered_short_texts_map_to_minus_one(self):
        p = TextPreprocessor(min_length=3)
        texts = ["hello", "ok", "hi", "world"]
        processed, o2p, _ = p.preprocess(texts)

        assert processed == ["hello", "world"]
        assert o2p[0] == 0
        assert o2p[1] == -1   # "ok" too short
        assert o2p[2] == -1   # "hi" too short
        assert o2p[3] == 1

    def test_punctuation_only_maps_to_minus_one(self):
        p = TextPreprocessor()
        texts = ["hello", "...", "world"]
        processed, o2p, _ = p.preprocess(texts)
        assert processed == ["hello", "world"]
        assert o2p[1] == -1

    def test_processed_to_original_points_at_first_occurrence(self):
        p = TextPreprocessor()
        texts = ["apple", "apple", "banana"]
        _, _, p2o = p.preprocess(texts)
        # Both "apple"s share processed index 0; p2o records the FIRST original idx.
        assert p2o == [0, 2]

    def test_o2p_covers_every_input_index(self):
        p = TextPreprocessor()
        texts = ["a", "b", None, "a", ""]
        _, o2p, _ = p.preprocess(texts)
        assert set(o2p.keys()) == set(range(len(texts)))

    def test_empty_input(self):
        p = TextPreprocessor()
        processed, o2p, p2o = p.preprocess([])
        assert processed == []
        assert o2p == {}
        assert p2o == []

    def test_all_missing_returns_empty_processed(self):
        p = TextPreprocessor()
        processed, o2p, p2o = p.preprocess([None, float("nan"), None])
        assert processed == []
        assert all(v == -1 for v in o2p.values())
        assert p2o == []


# ---------------------------------------------------------------------------
# preprocess() without deduplication
# ---------------------------------------------------------------------------

class TestPreprocessNoDedup:
    """``deduplicate=False`` keeps duplicates but still filters missing/short rows."""

    def test_keeps_duplicates(self):
        p = TextPreprocessor()
        texts = ["apple", "apple", "banana"]
        processed, o2p, _ = p.preprocess(texts, deduplicate=False)
        assert processed == ["apple", "apple", "banana"]
        assert o2p == {0: 0, 1: 1, 2: 2}

    def test_filters_missing_with_minus_one(self):
        p = TextPreprocessor()
        texts = ["a", None, "b", float("nan"), "c"]
        processed, o2p, _ = p.preprocess(texts, deduplicate=False)
        assert processed == ["a", "b", "c"]
        assert o2p[1] == -1
        assert o2p[3] == -1
        # Surviving positions are reindexed 0..len-1
        assert o2p[0] == 0
        assert o2p[2] == 1
        assert o2p[4] == 2

    def test_filters_short(self):
        p = TextPreprocessor(min_length=3)
        texts = ["yes", "no", "ok", "okay"]
        processed, o2p, _ = p.preprocess(texts, deduplicate=False)
        assert processed == ["yes", "okay"]
        assert o2p == {0: 0, 1: -1, 2: -1, 3: 1}


# ---------------------------------------------------------------------------
# Index-mapping invariants — round-trip correctness
# ---------------------------------------------------------------------------

class TestIndexMappingInvariants:
    """Index maps are self-consistent end-to-end."""

    def test_dedup_round_trip(self):
        p = TextPreprocessor()
        texts = ["Apple Pie", "apple pie", "banana split", "BANANA SPLIT", None, ""]
        processed, o2p, p2o = p.preprocess(texts)

        # For every surviving original index, processed[o2p[i]] equals
        # the cleaned form.
        for orig_idx, proc_idx in o2p.items():
            if proc_idx == -1:
                continue
            assert 0 <= proc_idx < len(processed)
            cleaned = p._clean_text(texts[orig_idx])
            assert processed[proc_idx] == cleaned

        # p2o always points at the FIRST original index for each processed row.
        for proc_idx, orig_idx in enumerate(p2o):
            assert o2p[orig_idx] == proc_idx
            # No earlier original index also maps here.
            for j in range(orig_idx):
                assert o2p[j] != proc_idx

    def test_no_dedup_processed_count_equals_surviving_inputs(self):
        p = TextPreprocessor()
        texts = ["a", None, "", "b", "c"]
        processed, o2p, _ = p.preprocess(texts, deduplicate=False)
        surviving = sum(1 for v in o2p.values() if v != -1)
        assert len(processed) == surviving

    def test_dedup_processed_count_equals_unique_surviving(self):
        p = TextPreprocessor()
        texts = ["A", "a", "B", "b", "B", None, ""]
        processed, _, _ = p.preprocess(texts)
        # After lowercasing + dedup the unique surviving rows are {"a", "b"}.
        assert sorted(processed) == ["a", "b"]
