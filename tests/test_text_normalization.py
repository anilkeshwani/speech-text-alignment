import pytest

from sardalign.text_normalization import text_normalize


class TestTextNormalizeDefaultIsoCode:
    """Tests for text_normalize with default iso_code '*'."""

    def test_basic_lowercasing(self):
        assert text_normalize("Hello World", iso_code="*") == "hello world"

    def test_punctuation_removal(self):
        assert text_normalize("hello, world!", iso_code="*") == "hello world"

    def test_number_removal_default(self):
        assert text_normalize("hello 123 world", iso_code="*") == "hello world"

    def test_whitespace_collapsing(self):
        assert text_normalize("hello   world", iso_code="*") == "hello world"


class TestTextNormalizeRemoveNumbersFalse:
    """Tests for text_normalize with remove_numbers=False."""

    def test_numbers_retained(self):
        result = text_normalize("hello 123 world", iso_code="*", remove_numbers=False)
        assert "123" in result


class TestTextNormalizeRemoveBrackets:
    """Tests for text_normalize with remove_brackets=True."""

    def test_brackets_removed(self):
        assert text_normalize("hello (note) world", iso_code="*", remove_brackets=True) == "hello world"


class TestTextNormalizeEdgeCases:
    """Edge case tests for text_normalize."""

    def test_empty_string(self):
        assert text_normalize("", iso_code="*") == ""

    def test_only_spaces(self):
        assert text_normalize("   ", iso_code="*") == ""
