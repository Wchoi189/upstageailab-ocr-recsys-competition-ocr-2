"""Pydantic schema for receipt data extraction.

This module defines the structured output format for receipt data,
including store information, line items, totals, and payment details.
"""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LineItem(BaseModel):
    """Single item on a receipt.

    Represents a purchased product or service with quantity and price.

    Attributes:
        name: Product or service name
        quantity: Number of items purchased
        unit_price: Price per unit
        total_price: Total price for this line (quantity * unit_price)
        unit: Unit of measurement (e.g., "개", "kg")
        confidence: Extraction confidence (0-1)
    """

    model_config = ConfigDict(frozen=True)

    name: str
    quantity: int | float | None = None
    unit_price: Decimal | None = None
    total_price: Decimal | None = None
    unit: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


PaymentMethod = Literal[
    "cash",
    "card",
    "mobile",
    "gift_card",
    "unknown",
]


class ReceiptMetadata(BaseModel):
    """Metadata about the extraction process.

    Attributes:
        extraction_time_ms: Time taken for extraction in milliseconds
        ocr_confidence: Average OCR confidence
        model_version: Version of extraction model/logic
        source_image_path: Path to source image (if available)
    """

    extraction_time_ms: float | None = None
    ocr_confidence: float | None = None
    model_version: str = "1.0.0"
    source_image_path: str | None = None


class ReceiptData(BaseModel):
    """Structured receipt extraction result.

    The complete structured output from receipt extraction, containing
    all recognized fields from a receipt image.

    Attributes:
        # Store Information
        store_name: Name of the store/merchant
        store_address: Store address
        store_phone: Store phone number
        business_number: Korean business registration number (사업자번호)

        # Transaction Details
        transaction_date: Date of the transaction
        transaction_time: Time of the transaction
        receipt_number: Receipt or transaction number

        # Items
        items: List of purchased items

        # Totals
        subtotal: Sum before tax
        tax: Tax amount
        total: Final total amount

        # Payment
        payment_method: How payment was made
        card_last_four: Last 4 digits of card (if card payment)

        # Metadata
        extraction_confidence: Overall extraction confidence (0-1)
        raw_text: Raw OCR text for debugging
        metadata: Additional extraction metadata
    """

    # Store Information
    store_name: str | None = None
    store_address: str | None = None
    store_phone: str | None = None
    business_number: str | None = None

    # Transaction Details
    transaction_date: date | None = None
    transaction_time: time | None = None
    receipt_number: str | None = None

    # Items
    items: list[LineItem] = Field(default_factory=list)

    # Totals
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal | None = None

    # Payment
    payment_method: PaymentMethod = "unknown"
    card_last_four: str | None = Field(default=None, min_length=4, max_length=4)

    # Metadata
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_text: str = ""
    metadata: ReceiptMetadata = Field(default_factory=ReceiptMetadata)

    @property
    def has_items(self) -> bool:
        """Check if any items were extracted."""
        return len(self.items) > 0

    @property
    def num_items(self) -> int:
        """Number of extracted items."""
        return len(self.items)

    @property
    def items_total(self) -> Decimal:
        """Sum of all item prices."""
        return sum(((item.total_price or Decimal(0)) for item in self.items), start=Decimal(0))

    @property
    def is_complete(self) -> bool:
        """Check if essential fields are extracted.

        A receipt is considered complete if it has:
        - Store name OR total
        - At least one item OR a total
        """
        has_store = self.store_name is not None
        has_total = self.total is not None
        has_items = len(self.items) > 0

        return (has_store or has_total) and (has_items or has_total)

    def to_dict(self) -> dict:
        """Convert to dictionary with JSON-serializable types.

        Converts Decimal to float and date/time to ISO strings.
        """
        result = self.model_dump()

        # Convert Decimal fields
        if result.get("subtotal"):
            result["subtotal"] = float(result["subtotal"])
        if result.get("tax"):
            result["tax"] = float(result["tax"])
        if result.get("total"):
            result["total"] = float(result["total"])

        # Convert items
        for item in result.get("items", []):
            if item.get("unit_price"):
                item["unit_price"] = float(item["unit_price"])
            if item.get("total_price"):
                item["total_price"] = float(item["total_price"])

        # Convert date/time
        if result.get("transaction_date"):
            result["transaction_date"] = result["transaction_date"].isoformat()
        if result.get("transaction_time"):
            result["transaction_time"] = result["transaction_time"].isoformat()

        return result


__all__ = [
    "LineItem",
    "PaymentMethod",
    "ReceiptData",
    "ReceiptMetadata",
]
