#!/usr/bin/env python3
"""Extract charset from LMDB label data for Korean OCR tokenizer."""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import lmdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Basic safety charset: digits, punctuation, space, common Latin
SAFETY_CHARSET = set("0123456789" "abcdefghijklmnopqrstuvwxyz" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" " .,!?:;-()[]{}'\"/\\@#$%&*+=<>~`")


def extract_charset(lmdb_path: str, sample_limit: int | None = None) -> dict:
    """
    Extract all unique characters from LMDB labels.

    Args:
        lmdb_path: Path to LMDB directory
        sample_limit: Optional limit on samples to scan (None = all)

    Returns:
        dict with charset, vocab_size, and stats
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn:
        num_samples_raw = txn.get(b"num-samples")
        if num_samples_raw:
            num_samples = int(num_samples_raw.decode())
        else:
            # Fallback: count entries
            num_samples = env.stat()["entries"] // 2

    logger.info("LMDB has %d samples", num_samples)

    if sample_limit:
        num_samples = min(num_samples, sample_limit)
        logger.info("Limiting scan to %d samples", num_samples)

    char_counter: Counter = Counter()
    label_lengths: list[int] = []

    with env.begin() as txn:
        for idx in range(1, num_samples + 1):
            label_key = f"label-{idx:09d}".encode()
            label_raw = txn.get(label_key)

            if label_raw:
                label = label_raw.decode("utf-8")
                label_lengths.append(len(label))
                char_counter.update(label)

            if idx % 100000 == 0:
                logger.info("Scanned %d / %d", idx, num_samples)

    env.close()

    # Build charset: all unique chars from data + safety set
    data_chars = set(char_counter.keys())
    all_chars = data_chars | SAFETY_CHARSET

    # Sort for determinism: special order (Korean by Unicode, then others)
    charset_list = sorted(all_chars)

    # vocab_size = 3 specials (PAD, BOS, EOS) + len(charset)
    vocab_size = 3 + len(charset_list)

    stats = {
        "total_samples": num_samples,
        "unique_chars_from_data": len(data_chars),
        "safety_chars_added": len(SAFETY_CHARSET - data_chars),
        "final_charset_size": len(charset_list),
        "vocab_size": vocab_size,
        "avg_label_length": sum(label_lengths) / len(label_lengths) if label_lengths else 0,
        "max_label_length": max(label_lengths) if label_lengths else 0,
        "top_10_chars": char_counter.most_common(10),
    }

    return {
        "charset": charset_list,
        "vocab_size": vocab_size,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract charset from LMDB for OCR tokenizer")
    parser.add_argument("--lmdb-path", required=True, help="Path to LMDB directory")
    parser.add_argument("--output", default="ocr/data/charset.json", help="Output JSON path")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit samples to scan")
    args = parser.parse_args()

    logger.info("Extracting charset from %s", args.lmdb_path)
    result = extract_charset(args.lmdb_path, args.sample_limit)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Saved charset to %s", output_path)
    logger.info("Stats: %s", json.dumps(result["stats"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
