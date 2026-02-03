# Context Bundle Token Budget Report

**Generated**: measure_bundle_tokens.py
**Budget Constraints**: <3000 tokens, ≤6 files per bundle

## Summary

- **Total Bundles**: 17
- **Compliant**: 4 / 17 (23.5%)
- **Non-Compliant**: 13

## Bundle Details

| Bundle | Total Tokens | Total Files | Status |

|:-------|-------------:|------------:|:------:|
| ocr-debugging                  |       17862 |          10 |   ❌    |
| documentation-update           |       15437 |           9 |   ❌    |
| compliance-check               |       14803 |          12 |   ❌    |
| v5-domains-standard            |        7693 |           7 |   ❌    |
| security-review                |        6797 |           5 |   ❌    |
| pipeline-development           |        6644 |           4 |   ❌    |
| hydra-configuration            |        6335 |           4 |   ❌    |
| project-compass                |        5288 |           6 |   ❌    |
| ocr-experiment                 |        5228 |           7 |   ❌    |
| rec-historical-context         |        5148 |           6 |   ❌    |
| rec-audit-foundation           |        4693 |           5 |   ❌    |
| ast-debugging-tools            |        3472 |           3 |   ❌    |
| agent-configuration            |        3121 |           6 |   ❌    |
| ocr-text-recognition           |        1800 |           3 |   ✅    |
| ocr-text-detection             |         483 |           2 |   ✅    |
| ocr-layout-analysis            |         424 |           2 |   ✅    |
| ocr-information-extraction     |         283 |           1 |   ✅    |

## Non-Compliant Bundles (Detailed)

### agent-configuration
- **Total Tokens**: 3121 (limit: 3000)
- **Total Files**: 6 (limit: 6)
- **⚠️ Token Excess**: +121 tokens (4.0% over)

**Per-Tier Breakdown**:
- `tier1`: 1449 tokens, 2 files
- `tier2`: 825 tokens, 2 files
- `tier3`: 847 tokens, 2 files

### ast-debugging-tools
- **Total Tokens**: 3472 (limit: 3000)
- **Total Files**: 3 (limit: 6)
- **⚠️ Token Excess**: +472 tokens (15.7% over)

**Per-Tier Breakdown**:
- `tier1`: 3472 tokens, 3 files

### compliance-check
- **Total Tokens**: 14803 (limit: 3000)
- **Total Files**: 12 (limit: 6)
- **⚠️ Token Excess**: +11803 tokens (393.4% over)
- **⚠️ File Excess**: +6 files

**Per-Tier Breakdown**:
- `tier1`: 1874 tokens, 3 files
- `tier2`: 8292 tokens, 4 files
- `tier3`: 4637 tokens, 5 files

### documentation-update
- **Total Tokens**: 15437 (limit: 3000)
- **Total Files**: 9 (limit: 6)
- **⚠️ Token Excess**: +12437 tokens (414.6% over)
- **⚠️ File Excess**: +3 files

**Per-Tier Breakdown**:
- `tier1`: 2387 tokens, 4 files
- `tier2`: 7082 tokens, 2 files
- `tier3`: 5968 tokens, 3 files

### hydra-configuration
- **Total Tokens**: 6335 (limit: 3000)
- **Total Files**: 4 (limit: 6)
- **⚠️ Token Excess**: +3335 tokens (111.2% over)

**Per-Tier Breakdown**:
- `tier1`: 746 tokens, 2 files
- `tier2`: 3295 tokens, 1 files
- `tier3`: 2294 tokens, 1 files

### ocr-debugging
- **Total Tokens**: 17862 (limit: 3000)
- **Total Files**: 10 (limit: 6)
- **⚠️ Token Excess**: +14862 tokens (495.4% over)
- **⚠️ File Excess**: +4 files

**Per-Tier Breakdown**:
- `tier1`: 5045 tokens, 3 files
- `tier2`: 5089 tokens, 3 files
- `tier3`: 7728 tokens, 4 files

### ocr-experiment
- **Total Tokens**: 5228 (limit: 3000)
- **Total Files**: 7 (limit: 6)
- **⚠️ Token Excess**: +2228 tokens (74.3% over)
- **⚠️ File Excess**: +1 files

**Per-Tier Breakdown**:
- `tier1`: 1174 tokens, 3 files
- `tier2`: 379 tokens, 1 files
- `tier3`: 3675 tokens, 3 files

### pipeline-development
- **Total Tokens**: 6644 (limit: 3000)
- **Total Files**: 4 (limit: 6)
- **⚠️ Token Excess**: +3644 tokens (121.5% over)

**Per-Tier Breakdown**:
- `tier1`: 3091 tokens, 1 files
- `tier2`: 3174 tokens, 2 files
- `tier3`: 379 tokens, 1 files

### project-compass
- **Total Tokens**: 5288 (limit: 3000)
- **Total Files**: 6 (limit: 6)
- **⚠️ Token Excess**: +2288 tokens (76.3% over)

**Per-Tier Breakdown**:
- `tier1`: 3226 tokens, 2 files
- `tier2`: 827 tokens, 3 files
- `tier3`: 1235 tokens, 1 files

### rec-audit-foundation
- **Total Tokens**: 4693 (limit: 3000)
- **Total Files**: 5 (limit: 6)
- **⚠️ Token Excess**: +1693 tokens (56.4% over)

**Per-Tier Breakdown**:
- `tier1`: 1494 tokens, 3 files
- `tier2`: 3199 tokens, 2 files

### rec-historical-context
- **Total Tokens**: 5148 (limit: 3000)
- **Total Files**: 6 (limit: 6)
- **⚠️ Token Excess**: +2148 tokens (71.6% over)

**Per-Tier Breakdown**:
- `tier1`: 633 tokens, 2 files
- `tier2`: 4515 tokens, 4 files

### security-review
- **Total Tokens**: 6797 (limit: 3000)
- **Total Files**: 5 (limit: 6)
- **⚠️ Token Excess**: +3797 tokens (126.6% over)

**Per-Tier Breakdown**:
- `tier1`: 800 tokens, 2 files
- `tier2`: 693 tokens, 1 files
- `tier3`: 5304 tokens, 2 files

### v5-domains-standard
- **Total Tokens**: 7693 (limit: 3000)
- **Total Files**: 7 (limit: 6)
- **⚠️ Token Excess**: +4693 tokens (156.4% over)
- **⚠️ File Excess**: +1 files

**Per-Tier Breakdown**:
- `tier1`: 4019 tokens, 5 files
- `tier2`: 3674 tokens, 2 files
