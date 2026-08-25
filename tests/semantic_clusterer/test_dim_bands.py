"""Unit tests for dim_bands.py — band boundaries, resolve_dim_band edge cases,
SUPPORTED_DIM_BANDS read-only contract, and BandGrid non-empty contract.

"""

import warnings

import pytest

from semantic_clusterer.dim_bands import (
    SUPPORTED_DIM_BANDS,
    BandGrid,
    resolve_dim_band,
)
from semantic_clusterer.pipeline.tuning import get_band_grid


# ---------------------------------------------------------------------------
# 1. Four bands' integer ranges
# ---------------------------------------------------------------------------

class TestBandRanges:
    """resolve_dim_band returns the correct band for every in-range value."""

    # --- low band: [256, 511] ---
    @pytest.mark.parametrize("D", [256, 300, 384, 511])
    def test_low_band_in_range(self, D):
        assert resolve_dim_band(D) == "low"

    def test_low_band_lower_boundary(self):
        assert resolve_dim_band(256) == "low"

    def test_low_band_upper_boundary(self):
        assert resolve_dim_band(511) == "low"

    # --- mid band: [512, 1023] ---
    @pytest.mark.parametrize("D", [512, 768, 1023])
    def test_mid_band_in_range(self, D):
        assert resolve_dim_band(D) == "mid"

    def test_mid_band_lower_boundary(self):
        assert resolve_dim_band(512) == "mid"

    def test_mid_band_upper_boundary(self):
        assert resolve_dim_band(1023) == "mid"

    # --- high band: [1024, 2047] ---
    @pytest.mark.parametrize("D", [1024, 1536, 2047])
    def test_high_band_in_range(self, D):
        assert resolve_dim_band(D) == "high"

    def test_high_band_lower_boundary(self):
        assert resolve_dim_band(1024) == "high"

    def test_high_band_upper_boundary(self):
        assert resolve_dim_band(2047) == "high"

    # --- xhigh band: [2048, 16384] ---
    @pytest.mark.parametrize("D", [2048, 3072, 16384])
    def test_xhigh_band_in_range(self, D):
        assert resolve_dim_band(D) == "xhigh"

    def test_xhigh_band_lower_boundary(self):
        assert resolve_dim_band(2048) == "xhigh"

    def test_xhigh_band_upper_boundary(self):
        assert resolve_dim_band(16384) == "xhigh"

    # --- adjacent boundaries do not bleed into each other ---
    def test_low_mid_boundary(self):
        """511 is low, 512 is mid — no overlap."""
        assert resolve_dim_band(511) == "low"
        assert resolve_dim_band(512) == "mid"

    def test_mid_high_boundary(self):
        """1023 is mid, 1024 is high — no overlap."""
        assert resolve_dim_band(1023) == "mid"
        assert resolve_dim_band(1024) == "high"

    def test_high_xhigh_boundary(self):
        """2047 is high, 2048 is xhigh — no overlap."""
        assert resolve_dim_band(2047) == "high"
        assert resolve_dim_band(2048) == "xhigh"


# ---------------------------------------------------------------------------
# 2. D < 1 raises ValueError
# ---------------------------------------------------------------------------

class TestInvalidDimRaisesValueError:
    """resolve_dim_band raises ValueError for D < 1."""

    @pytest.mark.parametrize("D", [0, -1, -100])
    def test_zero_and_negative_raise(self, D):
        with pytest.raises(ValueError):
            resolve_dim_band(D)

    def test_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="embedding_dim must be >= 1"):
            resolve_dim_band(0)

    def test_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="embedding_dim must be >= 1"):
            resolve_dim_band(-5)


# ---------------------------------------------------------------------------
# 3. 1 <= D < 256 warns once and returns "low"
# ---------------------------------------------------------------------------

class TestBelowLowBandWarning:
    """1 <= D < 256 emits exactly one UserWarning and returns 'low'."""

    @pytest.mark.parametrize("D", [1, 100, 128, 255])
    def test_returns_low(self, D):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            result = resolve_dim_band(D)
        assert result == "low"

    @pytest.mark.parametrize("D", [1, 100, 128, 255])
    def test_emits_exactly_one_warning(self, D):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_dim_band(D)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1

    def test_warning_message_contains_dim_value(self):
        D = 128
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_dim_band(D)
        msg = str(caught[0].message)
        assert str(D) in msg

    def test_warning_message_contains_low(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_dim_band(200)
        msg = str(caught[0].message)
        assert "low" in msg


# ---------------------------------------------------------------------------
# 4. D > 16384 warns once and returns "xhigh"
# ---------------------------------------------------------------------------

class TestAboveXHighBandWarning:
    """D > 16384 emits exactly one UserWarning and returns 'xhigh'."""

    @pytest.mark.parametrize("D", [16385, 20000, 32768])
    def test_returns_xhigh(self, D):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            result = resolve_dim_band(D)
        assert result == "xhigh"

    @pytest.mark.parametrize("D", [16385, 20000, 32768])
    def test_emits_exactly_one_warning(self, D):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_dim_band(D)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1

    def test_warning_message_contains_dim_value(self):
        D = 20000
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_dim_band(D)
        msg = str(caught[0].message)
        assert str(D) in msg

    def test_warning_message_contains_xhigh(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_dim_band(16385)
        msg = str(caught[0].message)
        assert "xhigh" in msg


# ---------------------------------------------------------------------------
# 5. SUPPORTED_DIM_BANDS is read-only
# ---------------------------------------------------------------------------

class TestSupportedDimBandsReadOnly:
    """SUPPORTED_DIM_BANDS is a MappingProxyType — mutations raise TypeError."""

    def test_contains_exactly_four_bands(self):
        assert set(SUPPORTED_DIM_BANDS.keys()) == {"low", "mid", "high", "xhigh"}

    def test_band_ranges_are_correct(self):
        assert SUPPORTED_DIM_BANDS["low"] == (256, 511)
        assert SUPPORTED_DIM_BANDS["mid"] == (512, 1023)
        assert SUPPORTED_DIM_BANDS["high"] == (1024, 2047)
        assert SUPPORTED_DIM_BANDS["xhigh"] == (2048, 16384)

    def test_assigning_new_key_raises(self):
        with pytest.raises(TypeError):
            SUPPORTED_DIM_BANDS["new_band"] = (0, 100)  # type: ignore[index]

    def test_overwriting_existing_key_raises(self):
        with pytest.raises(TypeError):
            SUPPORTED_DIM_BANDS["low"] = (0, 511)  # type: ignore[index]

    def test_deleting_key_raises(self):
        with pytest.raises(TypeError):
            del SUPPORTED_DIM_BANDS["low"]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 6. get_band_grid non-empty for all 4×4 (band, tier) combinations
# ---------------------------------------------------------------------------

_ALL_BANDS = ["low", "mid", "high", "xhigh"]
_ALL_TIERS = ["small", "medium", "large", "tiny"]


class TestBandGridNonEmpty:
    """Every (band, tier) pair returns a BandGrid with non-empty list fields."""

    @pytest.mark.parametrize("band", _ALL_BANDS)
    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_get_band_grid_returns_band_grid_instance(self, band, tier):
        grid = get_band_grid(band, tier)
        assert isinstance(grid, BandGrid)

    @pytest.mark.parametrize("band", _ALL_BANDS)
    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_pca_targets_non_empty(self, band, tier):
        grid = get_band_grid(band, tier)
        assert len(grid.pca_targets) >= 1, (
            f"pca_targets is empty for band={band!r}, tier={tier!r}"
        )

    @pytest.mark.parametrize("band", _ALL_BANDS)
    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_umap_n_neighbors_non_empty(self, band, tier):
        grid = get_band_grid(band, tier)
        assert len(grid.umap_n_neighbors) >= 1, (
            f"umap_n_neighbors is empty for band={band!r}, tier={tier!r}"
        )

    @pytest.mark.parametrize("band", _ALL_BANDS)
    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_umap_n_components_non_empty(self, band, tier):
        grid = get_band_grid(band, tier)
        assert len(grid.umap_n_components) >= 1, (
            f"umap_n_components is empty for band={band!r}, tier={tier!r}"
        )

    @pytest.mark.parametrize("band", _ALL_BANDS)
    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_hdbscan_min_cluster_size_ratios_non_empty(self, band, tier):
        grid = get_band_grid(band, tier)
        assert len(grid.hdbscan_min_cluster_size_ratios) >= 1, (
            f"hdbscan_min_cluster_size_ratios is empty for band={band!r}, tier={tier!r}"
        )

    @pytest.mark.parametrize("band", _ALL_BANDS)
    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_hdbscan_min_samples_non_empty(self, band, tier):
        grid = get_band_grid(band, tier)
        assert len(grid.hdbscan_min_samples) >= 1, (
            f"hdbscan_min_samples is empty for band={band!r}, tier={tier!r}"
        )

    @pytest.mark.parametrize("band", _ALL_BANDS)
    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_hdbscan_methods_non_empty(self, band, tier):
        grid = get_band_grid(band, tier)
        assert len(grid.hdbscan_methods) >= 1, (
            f"hdbscan_methods is empty for band={band!r}, tier={tier!r}"
        )

    @pytest.mark.parametrize("band", _ALL_BANDS)
    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_tiny_k_grid_non_empty(self, band, tier):
        """tiny_k_grid must be non-empty for all combinations (used by tiny tier)."""
        grid = get_band_grid(band, tier)
        assert len(grid.tiny_k_grid) >= 1, (
            f"tiny_k_grid is empty for band={band!r}, tier={tier!r}"
        )

    def test_all_16_combinations_covered(self):
        """Smoke test: all 16 (band, tier) pairs return without error."""
        for band in _ALL_BANDS:
            for tier in _ALL_TIERS:
                grid = get_band_grid(band, tier)
                assert grid is not None

    def test_invalid_band_raises_value_error(self):
        with pytest.raises(ValueError):
            get_band_grid("ultra", "small")

    def test_invalid_tier_raises_value_error(self):
        with pytest.raises(ValueError):
            get_band_grid("low", "xlarge")
