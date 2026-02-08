"""Unit tests for receipt extraction module.

Tests the ReceiptData schema, normalizers, and field extractor.
"""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal

import pytest
from pydantic import ValidationError

from ocr.domains.kie.inference.extraction.field_extractor import (
    ExtractorConfig,
    ReceiptFieldExtractor,
)
from ocr.domains.kie.inference.extraction.normalizers import (
    normalize_business_number,
    normalize_currency,
    normalize_date,
    normalize_phone,
    normalize_time,
)
from ocr.domains.kie.inference.extraction.receipt_schema import (
    LineItem,
    ReceiptData,
    ReceiptMetadata,
)


class TestNormalizeCurrency:
    """Tests for currency normalization."""

    def test_simple_amount(self):
        """Simple number should be parsed."""
        assert normalize_currency("12500") == Decimal("12500")

    def test_amount_with_commas(self):
        """Comma-separated amount should be parsed."""
        assert normalize_currency("12,500") == Decimal("12500")

    def test_amount_with_won(self):
        """Amount with 원 suffix should be parsed."""
        assert normalize_currency("12,500원") == Decimal("12500")

    def test_amount_with_won_symbol(self):
        """Amount with ₩ prefix should be parsed."""
        assert normalize_currency("₩ 12,500") == Decimal("12500")

    def test_amount_with_decimal(self):
        """Amount with decimal should be parsed."""
        assert normalize_currency("12,500.00원") == Decimal("12500.00")

    def test_empty_string(self):
        """Empty string should return None."""
        assert normalize_currency("") is None

    def test_non_numeric(self):
        """Non-numeric text should return None."""
        assert normalize_currency("hello") is None


class TestNormalizeDate:
    """Tests for date normalization."""

    def test_iso_format(self):
        """ISO format YYYY-MM-DD should be parsed."""
        assert normalize_date("2024-12-25") == date(2024, 12, 25)

    def test_slash_format(self):
        """Slash format YYYY/MM/DD should be parsed."""
        assert normalize_date("2024/12/25") == date(2024, 12, 25)

    def test_dot_format(self):
        """Dot format YYYY.MM.DD should be parsed."""
        assert normalize_date("2024.12.25") == date(2024, 12, 25)

    def test_two_digit_year(self):
        """Two-digit year should be parsed."""
        result = normalize_date("24-12-25")
        assert result is not None
        assert result.month == 12
        assert result.day == 25

    def test_no_separator(self):
        """YYYYMMDD format should be parsed."""
        assert normalize_date("20241225") == date(2024, 12, 25)

    def test_empty_string(self):
        """Empty string should return None."""
        assert normalize_date("") is None

    def test_invalid_date(self):
        """Invalid date should return None."""
        assert normalize_date("2024-13-45") is None  # Invalid month/day


class TestNormalizeTime:
    """Tests for time normalization."""

    def test_simple_time(self):
        """Simple HH:MM format should be parsed."""
        assert normalize_time("14:30") == time(14, 30)

    def test_time_with_seconds(self):
        """HH:MM:SS format should be parsed."""
        assert normalize_time("14:30:45") == time(14, 30, 45)

    def test_korean_format(self):
        """Korean format HH시 MM분 should be parsed."""
        assert normalize_time("14시 30분") == time(14, 30)

    def test_am_pm(self):
        """AM/PM format should be handled."""
        result = normalize_time("2:30 PM")
        assert result is not None
        assert result.hour == 14

    def test_empty_string(self):
        """Empty string should return None."""
        assert normalize_time("") is None


class TestNormalizePhone:
    """Tests for phone number normalization."""

    def test_standard_format(self):
        """Standard Korean phone format should be normalized."""
        assert normalize_phone("010-1234-5678") == "010-1234-5678"

    def test_with_spaces(self):
        """Phone with spaces should be normalized."""
        assert normalize_phone("010 1234 5678") == "010-1234-5678"

    def test_with_parenthesis(self):
        """Phone with parenthesis should be normalized."""
        assert normalize_phone("02)1234-5678") == "02-1234-5678"

    def test_special_number(self):
        """Special numbers like 1588 should work."""
        assert normalize_phone("1588-1234") == "1588-1234"

    def test_empty_string(self):
        """Empty string should return None."""
        assert normalize_phone("") is None


class TestNormalizeBusinessNumber:
    """Tests for business number normalization."""

    def test_with_dashes(self):
        """Business number with dashes should be normalized."""
        assert normalize_business_number("123-45-67890") == "123-45-67890"

    def test_without_dashes(self):
        """Business number without dashes should be formatted."""
        assert normalize_business_number("1234567890") == "123-45-67890"

    def test_wrong_length(self):
        """Wrong length should return None."""
        assert normalize_business_number("12345") is None

    def test_empty_string(self):
        """Empty string should return None."""
        assert normalize_business_number("") is None


class TestLineItem:
    """Tests for LineItem model."""

    def test_valid_item(self):
        """Valid item should be created."""
        item = LineItem(
            name="아메리카노",
            quantity=2,
            unit_price=Decimal("4500"),
            total_price=Decimal("9000"),
            confidence=0.9,
        )

        assert item.name == "아메리카노"
        assert item.quantity == 2
        assert item.total_price == Decimal("9000")

    def test_minimal_item(self):
        """Item with only name should be valid."""
        item = LineItem(name="카페라떼")

        assert item.name == "카페라떼"
        assert item.quantity is None
        assert item.confidence == 0.0

    def test_frozen_item(self):
        """Item should be immutable."""
        item = LineItem(name="Test", confidence=0.5)
        with pytest.raises(ValidationError):
            item.name = "Changed"


class TestReceiptData:
    """Tests for ReceiptData model."""

    def test_empty_receipt(self):
        """Empty receipt should be valid."""
        receipt = ReceiptData()

        assert receipt.store_name is None
        assert receipt.items == []
        assert receipt.is_complete is False

    def test_complete_receipt(self):
        """Receipt with store and total is complete."""
        receipt = ReceiptData(
            store_name="스타벅스",
            total=Decimal("5500"),
        )

        assert receipt.is_complete is True

    def test_has_items(self):
        """has_items property should work."""
        receipt = ReceiptData(items=[LineItem(name="Test")])

        assert receipt.has_items is True
        assert receipt.num_items == 1

    def test_items_total(self):
        """items_total should sum item prices."""
        receipt = ReceiptData(
            items=[
                LineItem(name="Item 1", total_price=Decimal("1000")),
                LineItem(name="Item 2", total_price=Decimal("2000")),
            ]
        )

        assert receipt.items_total == Decimal("3000")

    def test_to_dict_serialization(self):
        """to_dict should produce JSON-serializable output."""
        receipt = ReceiptData(
            store_name="Test",
            total=Decimal("5500"),
            transaction_date=date(2024, 12, 25),
            transaction_time=time(14, 30),
            items=[LineItem(name="Item", total_price=Decimal("5500"))],
        )

        result = receipt.to_dict()

        assert isinstance(result["total"], float)
        assert isinstance(result["transaction_date"], str)
        assert isinstance(result["transaction_time"], str)
        assert isinstance(result["items"][0]["total_price"], float)

    def test_card_last_four_validation(self):
        """Card last four must be exactly 4 digits."""
        # Valid
        receipt = ReceiptData(card_last_four="1234")
        assert receipt.card_last_four == "1234"

        # Invalid - too short
        with pytest.raises(ValueError):
            ReceiptData(card_last_four="123")

        # Invalid - too long
        with pytest.raises(ValueError):
            ReceiptData(card_last_four="12345")


class TestReceiptMetadata:
    """Tests for ReceiptMetadata model."""

    def test_default_values(self):
        """Default metadata should have expected values."""
        meta = ReceiptMetadata()

        assert meta.extraction_time_ms is None
        assert meta.model_version == "1.0.0"

    def test_custom_values(self):
        """Custom values should be preserved."""
        meta = ReceiptMetadata(
            extraction_time_ms=15.5,
            ocr_confidence=0.95,
            source_image_path="/path/to/image.jpg",
        )

        assert meta.extraction_time_ms == 15.5
        assert meta.ocr_confidence == 0.95


class TestExtractorConfig:
    """Tests for ExtractorConfig."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = ExtractorConfig()

        assert config.min_item_confidence == 0.5
        assert config.use_position_heuristics is True
        assert config.extract_items is True
        assert config.language == "ko"


class TestReceiptFieldExtractor:
    """Tests for ReceiptFieldExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create default extractor."""
        return ReceiptFieldExtractor()

    def test_empty_text(self, extractor):
        """Empty text should return empty receipt."""
        result = extractor.extract("")

        assert result.store_name is None
        assert result.total is None
        assert result.extraction_confidence == 0.0

    def test_extract_store_name(self, extractor):
        """Store name should be extracted."""
        text = "상호: 스타벅스 강남점"
        result = extractor.extract(text)

        assert result.store_name == "스타벅스 강남점"

    def test_extract_total(self, extractor):
        """Total should be extracted."""
        text = "합계: 12,500원"
        result = extractor.extract(text)

        assert result.total == Decimal("12500")

    def test_extract_date(self, extractor):
        """Date should be extracted."""
        text = "거래일: 2024-12-25"
        result = extractor.extract(text)

        assert result.transaction_date == date(2024, 12, 25)

    def test_extract_phone(self, extractor):
        """Phone should be extracted."""
        text = "전화: 02-1234-5678"
        result = extractor.extract(text)

        assert result.store_phone == "02-1234-5678"

    def test_detect_card_payment(self, extractor):
        """Card payment should be detected."""
        text = "신용카드 결제"
        result = extractor.extract(text)

        assert result.payment_method == "card"

    def test_detect_cash_payment(self, extractor):
        """Cash payment should be detected."""
        text = "현금 결제"
        result = extractor.extract(text)

        assert result.payment_method == "cash"

    def test_extract_items(self, extractor):
        """Items should be extracted."""
        text = """
스타벅스
아메리카노 4,500원
카페라떼 5,000원
합계: 9,500원
"""
        result = extractor.extract(text)

        assert len(result.items) == 2
        assert any(item.name == "아메리카노" for item in result.items)
        assert any(item.name == "카페라떼" for item in result.items)

    def test_full_receipt(self, extractor):
        """Full receipt should be extracted."""
        text = """
스타벅스 강남역점
서울시 강남구 강남대로 123
전화: 02-1234-5678
사업자번호: 123-45-67890
거래일: 2024-12-25 14:30:00

아메리카노 4,500원
카페라떼 5,000원

소계: 9,500원
부가세: 950원
합계: 10,450원

신용카드 ****1234
"""
        result = extractor.extract(text)

        # Check extracted fields
        assert "스타벅스" in (result.store_name or "")
        assert result.store_phone == "02-1234-5678"
        assert result.business_number == "123-45-67890"
        assert result.transaction_date == date(2024, 12, 25)
        assert result.subtotal == Decimal("9500")
        assert result.tax == Decimal("950")
        assert result.total == Decimal("10450")
        assert result.payment_method == "card"
        assert len(result.items) >= 2

        # Should be complete
        assert result.is_complete is True
        assert result.extraction_confidence > 0.5

    def test_extraction_confidence(self, extractor):
        """Extraction confidence should reflect quality."""
        # Minimal receipt
        minimal = extractor.extract("hello world")
        assert minimal.extraction_confidence < 0.5

        # Complete receipt
        complete_text = """
스타벅스
아메리카노 4,500원
합계: 4,500원
2024-12-25
"""
        complete = extractor.extract(complete_text)
        assert complete.extraction_confidence > minimal.extraction_confidence
