"""Field extractor for receipt data.

This module provides regex-based and heuristic extraction of
structured fields from OCR text.
"""

from __future__ import annotations

import logging
import re
import time as time_module
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from .normalizers import (
    normalize_business_number,
    normalize_currency,
    normalize_date,
    normalize_phone,
    normalize_time,
)
from .receipt_schema import LineItem, ReceiptData, ReceiptMetadata

if TYPE_CHECKING:
    from ..layout.contracts import LayoutResult

LOGGER = logging.getLogger(__name__)


@dataclass
class ExtractorConfig:
    """Configuration for field extraction.

    Attributes:
        min_item_confidence: Minimum confidence for item extraction
        use_position_heuristics: Use position-based heuristics
        extract_items: Whether to extract individual items
        language: Primary language (ko, en)
    """

    min_item_confidence: float = 0.5
    use_position_heuristics: bool = True
    extract_items: bool = True
    language: str = "ko"


# Korean receipt field patterns
KOREAN_PATTERNS = {
    # Store name patterns (often at top)
    "store_name": [
        re.compile(r"(?:상\s*호|매\s*장\s*명)[:\s]*(.+)", re.IGNORECASE),
    ],
    # Total amount patterns
    "total": [
        re.compile(r"(?:합\s*계|총\s*액|결\s*제\s*금\s*액)[:\s]*₩?\s*([0-9,]+)", re.IGNORECASE),
        re.compile(r"(?:TOTAL|합계)[:\s]*₩?\s*([0-9,]+)", re.IGNORECASE),
    ],
    # Subtotal patterns
    "subtotal": [
        re.compile(r"(?:소\s*계|과세\s*물품)[:\s]*₩?\s*([0-9,]+)", re.IGNORECASE),
    ],
    # Tax patterns
    "tax": [
        re.compile(r"(?:부가세|VAT|세금)[:\s]*₩?\s*([0-9,]+)", re.IGNORECASE),
    ],
    # Date patterns
    "date": [
        re.compile(r"(?:거래일|날짜|일시)[:\s]*(\d{4}[-./]\d{1,2}[-./]\d{1,2})", re.IGNORECASE),
        re.compile(r"(\d{4}[-./]\d{1,2}[-./]\d{1,2})"),
    ],
    # Time patterns
    "time": [
        re.compile(r"(?:거래일|일시)[:\s]*\d{4}[-./]\d{1,2}[-./]\d{1,2}\s+(\d{1,2}:\d{2}(?::\d{2})?)", re.IGNORECASE),
        re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?)"),
    ],
    # Phone patterns
    "phone": [
        re.compile(r"(?:전화|TEL|연락처)[:\s]*([\d\-)\s]+)"),
        re.compile(r"(\d{2,3}[-)\s]?\d{3,4}[-\s]?\d{4})"),
    ],
    # Business number patterns
    "business_number": [
        re.compile(r"(?:사업자\s*(?:등록\s*)?번호|사\s*업\s*자\s*번\s*호)[:\s]*([\d\-]+)"),
    ],
    # Receipt number patterns
    "receipt_number": [
        re.compile(r"(?:영수증\s*번호|거래\s*번호|NO)[:\s#]*(\d+)"),
    ],
    # Address patterns
    "address": [
        re.compile(r"(?:주\s*소)[:\s]*(.+)"),
    ],
    # Card payment indicators
    "card_payment": [
        re.compile(r"(?:카드|CARD|신용|체크)", re.IGNORECASE),
    ],
    # Cash payment indicators
    "cash_payment": [
        re.compile(r"(?:현금|CASH)", re.IGNORECASE),
    ],
    # Card last 4 digits
    "card_last_four": [
        re.compile(r"(?:카드|CARD).*?(\d{4})\s*$"),
        re.compile(r"\*{4,}(\d{4})"),
    ],
    # Item patterns (name followed by price)
    "item": [
        re.compile(r"(.+?)\s+₩?\s*([0-9,]+)\s*원?\s*$"),
        re.compile(r"(.+?)\s+([0-9,]+)\s*$"),
    ],
}


class ReceiptFieldExtractor:
    """Extract structured fields from OCR text.

    Uses regex patterns and position heuristics to extract
    receipt data fields from raw OCR output.

    Example:
        >>> extractor = ReceiptFieldExtractor()
        >>> result = extractor.extract(raw_text="상호: 스타벅스\n합계: 5,500원")
    """

    def __init__(self, config: ExtractorConfig | None = None) -> None:
        """Initialize field extractor.

        Args:
            config: Extraction configuration
        """
        self.config = config or ExtractorConfig()
        self._patterns = KOREAN_PATTERNS

        LOGGER.info("ReceiptFieldExtractor initialized | language=%s", self.config.language)

    def extract(
        self,
        raw_text: str | None = None,
        layout: LayoutResult | None = None,
    ) -> ReceiptData:
        """Extract receipt fields from OCR text.

        Uses regex patterns for known fields and position heuristics
        for field type inference.

        Args:
            raw_text: Raw OCR text (newline-separated lines)
            layout: Optional LayoutResult for position-aware extraction

        Returns:
            ReceiptData with extracted fields
        """
        start_time = time_module.perf_counter()

        # Get text from layout if not provided
        if raw_text is None and layout is not None:
            raw_text = layout.text

        if not raw_text:
            return ReceiptData(
                raw_text="",
                extraction_confidence=0.0,
                metadata=ReceiptMetadata(extraction_time_ms=0),
            )

        # Extract each field type
        store_name = self._extract_store_name(raw_text)
        store_address = self._extract_address(raw_text)
        store_phone = self._extract_phone(raw_text)
        business_number = self._extract_business_number(raw_text)

        transaction_date = self._extract_date(raw_text)
        transaction_time = self._extract_time(raw_text)
        receipt_number = self._extract_receipt_number(raw_text)

        subtotal = self._extract_subtotal(raw_text)
        tax = self._extract_tax(raw_text)
        total = self._extract_total(raw_text)

        payment_method = self._detect_payment_method(raw_text)
        card_last_four = self._extract_card_last_four(raw_text)

        items = self._extract_items(raw_text) if self.config.extract_items else []

        # Calculate confidence
        extraction_confidence = self._calculate_confidence(
            store_name=store_name,
            total=total,
            items=items,
            transaction_date=transaction_date,
        )

        elapsed_ms = (time_module.perf_counter() - start_time) * 1000

        return ReceiptData(
            store_name=store_name,
            store_address=store_address,
            store_phone=store_phone,
            business_number=business_number,
            transaction_date=transaction_date,
            transaction_time=transaction_time,
            receipt_number=receipt_number,
            items=items,
            subtotal=subtotal,
            tax=tax,
            total=total,
            payment_method=payment_method,
            card_last_four=card_last_four,
            extraction_confidence=extraction_confidence,
            raw_text=raw_text,
            metadata=ReceiptMetadata(extraction_time_ms=elapsed_ms),
        )

    def _extract_store_name(self, text: str) -> str | None:
        """Extract store name from text."""
        for pattern in self._patterns["store_name"]:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()

        # Heuristic: First non-empty line is often store name
        if self.config.use_position_heuristics:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if lines:
                first_line = lines[0]
                # Skip if it looks like a date, number, or pattern
                if not re.match(r"^\d+|^-+$|^=+$", first_line):
                    return first_line

        return None

    def _extract_address(self, text: str) -> str | None:
        """Extract store address from text."""
        for pattern in self._patterns["address"]:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return None

    def _extract_phone(self, text: str) -> str | None:
        """Extract phone number from text."""
        for pattern in self._patterns["phone"]:
            match = pattern.search(text)
            if match:
                raw_phone = match.group(1)
                return normalize_phone(raw_phone)
        return None

    def _extract_business_number(self, text: str) -> str | None:
        """Extract business registration number."""
        for pattern in self._patterns["business_number"]:
            match = pattern.search(text)
            if match:
                return normalize_business_number(match.group(1))
        return None

    def _extract_date(self, text: str):
        """Extract transaction date."""
        for pattern in self._patterns["date"]:
            match = pattern.search(text)
            if match:
                raw_date = match.group(1)
                result = normalize_date(raw_date)
                if result:
                    return result
        return None

    def _extract_time(self, text: str):
        """Extract transaction time."""
        for pattern in self._patterns["time"]:
            match = pattern.search(text)
            if match:
                raw_time = match.group(1)
                result = normalize_time(raw_time)
                if result:
                    return result
        return None

    def _extract_receipt_number(self, text: str) -> str | None:
        """Extract receipt number."""
        for pattern in self._patterns["receipt_number"]:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    def _extract_total(self, text: str) -> Decimal | None:
        """Extract total amount."""
        for pattern in self._patterns["total"]:
            match = pattern.search(text)
            if match:
                return normalize_currency(match.group(1))
        return None

    def _extract_subtotal(self, text: str) -> Decimal | None:
        """Extract subtotal amount."""
        for pattern in self._patterns["subtotal"]:
            match = pattern.search(text)
            if match:
                return normalize_currency(match.group(1))
        return None

    def _extract_tax(self, text: str) -> Decimal | None:
        """Extract tax amount."""
        for pattern in self._patterns["tax"]:
            match = pattern.search(text)
            if match:
                return normalize_currency(match.group(1))
        return None

    def _detect_payment_method(self, text: str) -> str:
        """Detect payment method from text."""
        for pattern in self._patterns["card_payment"]:
            if pattern.search(text):
                return "card"

        for pattern in self._patterns["cash_payment"]:
            if pattern.search(text):
                return "cash"

        return "unknown"

    def _extract_card_last_four(self, text: str) -> str | None:
        """Extract last 4 digits of card number."""
        for pattern in self._patterns["card_last_four"]:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    def _extract_items(self, text: str) -> list[LineItem]:
        """Extract line items from text.

        Uses heuristics to identify product lines with prices.
        """
        items: list[LineItem] = []
        lines = text.split("\n")

        # Skip header and footer lines
        skip_keywords = [
            "상호", "주소", "전화", "사업자", "합계", "총액", "결제",
            "부가세", "소계", "거스름", "카드", "현금", "영수증",
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip lines with known keywords
            if any(kw in line for kw in skip_keywords):
                continue

            # Try to match item pattern (name + price)
            for pattern in self._patterns["item"]:
                match = pattern.match(line)
                if match:
                    name = match.group(1).strip()
                    price_str = match.group(2)

                    # Skip very short names or number-only names
                    if len(name) < 2 or name.isdigit():
                        continue

                    price = normalize_currency(price_str)
                    if price and price > 0:
                        items.append(
                            LineItem(
                                name=name,
                                total_price=price,
                                confidence=0.7,  # Heuristic confidence
                            )
                        )
                    break

        return items

    def _calculate_confidence(
        self,
        store_name: str | None,
        total: Decimal | None,
        items: list[LineItem],
        transaction_date,
    ) -> float:
        """Calculate overall extraction confidence.

        Based on which fields were successfully extracted.
        """
        score = 0.0
        max_score = 5.0

        if store_name:
            score += 1.0
        if total:
            score += 1.5
        if items:
            score += 1.0
        if transaction_date:
            score += 1.0
        if len(items) >= 2:
            score += 0.5

        return min(1.0, score / max_score)


__all__ = [
    "ExtractorConfig",
    "ReceiptFieldExtractor",
]
