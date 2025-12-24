# Golden Receipt Dataset

This directory contains annotated receipt samples for regression testing
the extraction module.

## Purpose

These samples provide ground-truth annotations for:
1. Validating extraction accuracy
2. Regression testing when modifying extraction logic
3. Collecting edge cases and failure modes

## File Format

Each receipt sample consists of:
- `receipt_XXX.txt` - Raw OCR text (simulated or real)
- `receipt_XXX.json` - Expected `ReceiptData` output

## Sample Coverage

The dataset should include:
- [ ] Standard Korean convenience store receipts
- [ ] Coffee shop receipts (Starbucks, etc.)
- [ ] Restaurant receipts
- [ ] Online purchase receipts
- [ ] Edge cases (missing fields, OCR errors)

## Adding New Samples

1. Create `receipt_XXX.txt` with raw OCR text
2. Create `receipt_XXX.json` with expected output
3. Add to regression test in `tests/regression/test_golden_receipts.py`

## JSON Schema

```json
{
  "store_name": "스타벅스 강남점",
  "store_phone": "02-1234-5678",
  "transaction_date": "2024-12-25",
  "items": [
    {"name": "아메리카노", "total_price": 4500}
  ],
  "total": 4500,
  "payment_method": "card"
}
```
