"""Unit tests for tokenizer singleton caching."""
import pytest
from pathlib import Path

from ocr.domains.recognition.data.tokenizer import KoreanOCRTokenizer, _TOKENIZER_CACHE


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear tokenizer cache before each test."""
    _TOKENIZER_CACHE.clear()
    yield
    _TOKENIZER_CACHE.clear()


@pytest.fixture
def charset_path():
    """Return path to test charset file."""
    # Use the actual charset file in the project
    return str(Path(__file__).parents[4] / "ocr" / "data" / "charset.json")


class TestTokenizerCaching:
    """Test suite for tokenizer singleton pattern."""

    def test_get_or_create_returns_same_instance_for_same_params(self, charset_path):
        """Test that same parameters return the same tokenizer instance."""
        tok1 = KoreanOCRTokenizer.get_or_create(charset_path, max_len=25)
        tok2 = KoreanOCRTokenizer.get_or_create(charset_path, max_len=25)

        assert tok1 is tok2, "Same parameters should return same instance"
        assert len(_TOKENIZER_CACHE) == 1, "Cache should contain exactly one entry"

    def test_get_or_create_returns_different_instance_for_different_max_len(self, charset_path):
        """Test that different max_len creates different instances."""
        tok1 = KoreanOCRTokenizer.get_or_create(charset_path, max_len=25)
        tok2 = KoreanOCRTokenizer.get_or_create(charset_path, max_len=30)

        assert tok1 is not tok2, "Different max_len should return different instances"
        assert tok1.max_len == 25
        assert tok2.max_len == 30
        assert len(_TOKENIZER_CACHE) == 2, "Cache should contain two entries"

    def test_get_or_create_returns_different_instance_for_different_charset(self, charset_path, tmp_path):
        """Test that different charset paths create different instances."""
        # Create a temporary charset file
        temp_charset = tmp_path / "temp_charset.json"
        temp_charset.write_text('{"charset": ["a", "b", "c"]}')

        tok1 = KoreanOCRTokenizer.get_or_create(charset_path, max_len=25)
        tok2 = KoreanOCRTokenizer.get_or_create(str(temp_charset), max_len=25)

        assert tok1 is not tok2, "Different charset paths should return different instances"
        assert tok1.vocab_size != tok2.vocab_size
        assert len(_TOKENIZER_CACHE) == 2, "Cache should contain two entries"

    def test_get_or_create_with_path_object(self, charset_path):
        """Test that Path objects are handled correctly."""
        path_obj = Path(charset_path)
        tok1 = KoreanOCRTokenizer.get_or_create(path_obj, max_len=25)
        tok2 = KoreanOCRTokenizer.get_or_create(str(path_obj), max_len=25)

        assert tok1 is tok2, "Path and str should resolve to same cache entry"

    def test_get_or_create_with_relative_paths(self, charset_path):
        """Test that relative and absolute paths resolve to same cache entry."""
        import os

        # Get the absolute path
        abs_path = Path(charset_path).resolve()

        # Get relative path from current directory
        try:
            rel_path = abs_path.relative_to(Path.cwd())
        except ValueError:
            # If charset is not relative to cwd, skip this test
            pytest.skip("Charset path is not relative to current directory")

        tok1 = KoreanOCRTokenizer.get_or_create(str(abs_path), max_len=25)
        tok2 = KoreanOCRTokenizer.get_or_create(str(rel_path), max_len=25)

        assert tok1 is tok2, "Relative and absolute paths should resolve to same instance"
        assert len(_TOKENIZER_CACHE) == 1

    def test_cache_key_immutability(self, charset_path):
        """Test that cache keys are properly formed from immutable tuples."""
        tok = KoreanOCRTokenizer.get_or_create(charset_path, max_len=25)

        # Check cache structure
        assert len(_TOKENIZER_CACHE) == 1
        cache_key = list(_TOKENIZER_CACHE.keys())[0]

        assert isinstance(cache_key, tuple), "Cache key should be a tuple"
        assert len(cache_key) == 2, "Cache key should have 2 elements"
        assert isinstance(cache_key[0], str), "First element should be resolved path string"
        assert isinstance(cache_key[1], int), "Second element should be max_len integer"

    def test_tokenizer_properties_preserved(self, charset_path):
        """Test that cached tokenizer preserves all properties."""
        tok1 = KoreanOCRTokenizer.get_or_create(charset_path, max_len=25)
        tok2 = KoreanOCRTokenizer.get_or_create(charset_path, max_len=25)

        # Both should work identically
        test_text = "테스트"
        assert tok1.encode(test_text) == tok2.encode(test_text)
        assert tok1.vocab_size == tok2.vocab_size
        assert tok1.max_len == tok2.max_len

    def test_multiple_retrievals_from_cache(self, charset_path):
        """Test that multiple retrievals work correctly."""
        # Create initial tokenizer
        tok_original = KoreanOCRTokenizer.get_or_create(charset_path, max_len=25)

        # Retrieve multiple times
        tokenizers = [
            KoreanOCRTokenizer.get_or_create(charset_path, max_len=25)
            for _ in range(10)
        ]

        # All should be the same instance
        assert all(tok is tok_original for tok in tokenizers)
        assert len(_TOKENIZER_CACHE) == 1
