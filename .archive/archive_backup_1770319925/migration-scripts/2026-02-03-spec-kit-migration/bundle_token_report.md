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
| ocr-debugging                  |       18435 |          12 |   ❌    |
| documentation-update           |       15365 |           9 |   ❌    |
| compliance-check               |       14731 |          12 |   ❌    |
| ocr-experiment                 |       11303 |           9 |   ❌    |
| agent-configuration            |        8953 |           9 |   ❌    |
| v5-domains-standard            |        7693 |           7 |   ❌    |
| security-review                |        6797 |           5 |   ❌    |
| pipeline-development           |        6644 |           4 |   ❌    |
| hydra-configuration            |        6335 |           4 |   ❌    |
| project-compass                |        5288 |           6 |   ❌    |
| rec-historical-context         |        5148 |           6 |   ❌    |
| rec-audit-foundation           |        4693 |           5 |   ❌    |
| ast-debugging-tools            |        3400 |           3 |   ❌    |
| ocr-text-recognition           |        1800 |           3 |   ✅    |
| ocr-text-detection             |         483 |           2 |   ✅    |
| ocr-layout-analysis            |         424 |           2 |   ✅    |
| ocr-information-extraction     |         283 |           1 |   ✅    |

## Non-Compliant Bundles (Detailed)

### agent-configuration
- **Total Tokens**: 8953 (limit: 3000)
- **Total Files**: 9 (limit: 6)
- **⚠️ Token Excess**: +5953 tokens (198.4% over)
- **⚠️ File Excess**: +3 files

**Per-Tier Breakdown**:
- `tier1`: 1449 tokens, 2 files
- `tier2`: 1600 tokens, 4 files
- `tier3`: 5904 tokens, 3 files

### ast-debugging-tools
- **Total Tokens**: 3400 (limit: 3000)
- **Total Files**: 3 (limit: 6)
- **⚠️ Token Excess**: +400 tokens (13.3% over)

**Per-Tier Breakdown**:
- `tier1`: 3400 tokens, 3 files

### compliance-check
- **Total Tokens**: 14731 (limit: 3000)
- **Total Files**: 12 (limit: 6)
- **⚠️ Token Excess**: +11731 tokens (391.0% over)
- **⚠️ File Excess**: +6 files

**Per-Tier Breakdown**:
- `tier1`: 1874 tokens, 3 files
- `tier2`: 8220 tokens, 4 files
- `tier3`: 4637 tokens, 5 files

### documentation-update
- **Total Tokens**: 15365 (limit: 3000)
- **Total Files**: 9 (limit: 6)
- **⚠️ Token Excess**: +12365 tokens (412.2% over)
- **⚠️ File Excess**: +3 files

**Per-Tier Breakdown**:
- `tier1`: 2315 tokens, 4 files
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
- **Total Tokens**: 18435 (limit: 3000)
- **Total Files**: 12 (limit: 6)
- **⚠️ Token Excess**: +15435 tokens (514.5% over)
- **⚠️ File Excess**: +6 files

**Per-Tier Breakdown**:
- `tier1`: 5936 tokens, 4 files
- `tier2`: 8180 tokens, 4 files
- `tier3`: 4319 tokens, 4 files

### ocr-experiment
- **Total Tokens**: 11303 (limit: 3000)
- **Total Files**: 9 (limit: 6)
- **⚠️ Token Excess**: +8303 tokens (276.8% over)
- **⚠️ File Excess**: +3 files

**Per-Tier Breakdown**:
- `tier1`: 3599 tokens, 5 files
- `tier2`: 379 tokens, 1 files
- `tier3`: 7325 tokens, 3 files

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
