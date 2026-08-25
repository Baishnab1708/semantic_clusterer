"""Text preprocessing pipeline."""

import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


class TextPreprocessor:
    """Text preprocessing pipeline for semantic clustering.
    
    Steps:
    1. Unicode normalization (NFKC)
    2. Lowercasing
    3. Punctuation cleanup (optional)
    4. Whitespace normalization
    5. Deduplication with index mapping
    
    Attributes:
        lowercase: Whether to convert text to lowercase.
        remove_punctuation: Whether to remove punctuation.
        min_length: Minimum text length after preprocessing.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        min_length: int = 1,
    ):
        """Initialize the text preprocessor.
        
        Args:
            lowercase: Convert to lowercase.
            remove_punctuation: Remove punctuation characters.
            min_length: Minimum text length to keep.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_length = min_length

        # Compile regex patterns
        self._punctuation_pattern = re.compile(r'[^\w\s]', re.UNICODE)
        self._whitespace_pattern = re.compile(r'\s+')

    def _is_missing(self, value: Any) -> bool:
        """Check if value is missing-like (None, NaN, etc.)."""
        if value is None:
            return True
        # Check for float NaN
        if isinstance(value, float) and math.isnan(value):
            return True
        # Check for numpy NaN (if numpy is available)
        try:
            import numpy as np
            if isinstance(value, (np.floating, float)) and np.isnan(value):
                return True
        except (ImportError, TypeError, ValueError):
            pass
        # Check for pandas NA/NaT (if pandas is available)
        try:
            import pandas as pd
            if pd.isna(value):
                return True
        except (ImportError, TypeError, ValueError):
            pass
        return False

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        return unicodedata.normalize("NFKC", text)

    def _clean_text(self, text: Any) -> Optional[str]:
        """Apply all cleaning steps to a single text.
        
        Returns:
            Cleaned text string, or None for missing/invalid inputs.
            
        Raises:
            TypeError: For non-string objects that are not missing-like
                       (e.g., dict, list, set).
        """
        # Handle missing values first (None, NaN, pandas NA)
        if self._is_missing(text):
            return None

        # Type check - must be string after missing check
        if not isinstance(text, str):
            # Raise for clearly invalid types (dict, list, set, etc.)
            raise TypeError(
                f"Expected str or missing value (None/NaN), got {type(text).__name__}"
            )

        # Unicode normalization
        text = self._normalize_unicode(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = self._punctuation_pattern.sub(" ", text)

        text = self._whitespace_pattern.sub(" ", text).strip()

        return text

    def preprocess(
        self,
        texts: List[Any],
        deduplicate: bool = True,
    ) -> Tuple[List[str], Dict[int, int], List[int]]:
        """Preprocess a list of texts.
        
        Args:
            texts: List of raw text strings (may contain None/NaN).
            deduplicate: Whether to remove duplicates.
            
        Returns:
            A tuple of:
            - processed_texts: List of cleaned, unique texts.
            - original_to_processed: Mapping from original index to processed index.
              -1 indicates missing/invalid/too-short input.
            - processed_to_original: Mapping from processed index to first original index.
        """
        cleaned = [self._clean_text(text) for text in texts]

        if not deduplicate:
            # For non-dedup mode, still need to handle None values
            valid_cleaned = []
            original_to_processed: Dict[int, int] = {}
            for i, text in enumerate(cleaned):
                if text is None or len(text) < self.min_length:
                    original_to_processed[i] = -1
                else:
                    original_to_processed[i] = len(valid_cleaned)
                    valid_cleaned.append(text)
            processed_to_original = [i for i, idx in original_to_processed.items() if idx >= 0]
            return valid_cleaned, original_to_processed, processed_to_original

        # Deduplicate while preserving order and tracking indices
        seen: Dict[str, int] = {}  # cleaned_text -> processed_index
        processed_texts: List[str] = []
        original_to_processed: Dict[int, int] = {}
        processed_to_original: List[int] = []

        for orig_idx, text in enumerate(cleaned):
            # Skip None (missing values) and empty/short texts
            if text is None or len(text) < self.min_length:
                # Map to -1 to indicate filtered out
                original_to_processed[orig_idx] = -1
                continue

            if text in seen:
                # Duplicate: map to existing processed index
                original_to_processed[orig_idx] = seen[text]
            else:
                # New unique text
                proc_idx = len(processed_texts)
                seen[text] = proc_idx
                processed_texts.append(text)
                original_to_processed[orig_idx] = proc_idx
                processed_to_original.append(orig_idx)

        return processed_texts, original_to_processed, processed_to_original

    def preprocess_simple(self, texts: List[str]) -> List[str]:
        """Preprocess texts without deduplication or index tracking.
        
        Args:
            texts: List of raw text strings.
            
        Returns:
            List of cleaned texts.
        """
        return [self._clean_text(text) for text in texts]
