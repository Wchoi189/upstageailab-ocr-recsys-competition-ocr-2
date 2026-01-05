# Translation Strategy for Korean Documentation

## Overview

This document outlines the strategy for translating the README and linked technical documents to Korean using the GitHub Actions workflow with Upstage Solar API.

## Current Setup

### Workflow Configuration
- **File:** `.github/workflows/translation-action.yml`
- **Trigger:** Automatic on README.md changes + Manual dispatch
- **Provider:** Upstage Solar Pro 2 API
- **Output:** `README.ko.md` (standard Korean README)

### Language Selector
Added to `README.md` header:
```markdown
**Languages:** [English](README.md) · [한국어](README.ko.md)
```

## Translation Workflow Usage

### Automatic Translation (README only)
Triggers automatically when `README.md` is pushed to main branch.

### Manual Translation
1. Go to **Actions** tab in GitHub
2. Select **"Translate Documentation"** workflow
3. Click **"Run workflow"**
4. Enter files to translate (comma-separated):
   ```
   README.md
   ```
5. Click **"Run workflow"**

## Documents to Translate

### Priority 1: README (✅ Configured)
- **Source:** `README.md`
- **Output:** `README.ko.md`
- **Status:** Automated

### Priority 2: Key Technical Documents

The following documents are referenced in the README and should have Korean translations:

#### Bug Reports
1. `docs/artifacts/bug_reports/2026-01-04_1730_BUG_003_config-precedence-leak.md`
   - **Output:** `docs/artifacts/bug_reports/2026-01-04_1730_BUG_003_config-precedence-leak.ko.md`
   - **Reason:** Referenced in "Key Findings" section

#### Assessments
2. `docs/artifacts/assessments/2026-01-03_1200_assessment-layoutlmv3-kie-receipt-failure.md`
   - **Output:** `docs/artifacts/assessments/2026-01-03_1200_assessment-layoutlmv3-kie-receipt-failure.ko.md`
   - **Reason:** Referenced in "Key Findings" section

#### Experiment Reports
3. `experiment_manager/experiments/20251122_172313_perspective_correction/artifacts/20251122_1723_assessment_200-image-test-results.md`
   - **Output:** `experiment_manager/experiments/20251122_172313_perspective_correction/artifacts/20251122_1723_assessment_200-image-test-results.ko.md`
   - **Reason:** Referenced in "Experimental Findings"

4. `experiment_manager/experiments/20251220_154834_zero_prediction_images_debug/README.md`
   - **Output:** `experiment_manager/experiments/20251220_154834_zero_prediction_images_debug/README.ko.md`
   - **Reason:** Referenced in "Experimental Findings" (Sepia Discovery)

5. `experiment_manager/experiments/20251129_173500_perspective_correction_implementation/.metadata/assessments/20251129_1735_assessment_test-results-analysis.md`
   - **Output:** `experiment_manager/experiments/20251129_173500_perspective_correction_implementation/.metadata/assessments/20251129_1735_assessment_test-results-analysis.ko.md`
   - **Reason:** Referenced in "Experimental Findings" (Hybrid DL Approach)

## Implementation Strategy

### Option 1: Batch Translation (Recommended)

Create a separate workflow for batch translating all documents:

**File:** `.github/workflows/translate-docs-batch.yml`

```yaml
name: Translate All Documentation

on:
  workflow_dispatch:

jobs:
  translate-docs:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        file:
          - 'docs/artifacts/bug_reports/2026-01-04_1730_BUG_003_config-precedence-leak.md'
          - 'docs/artifacts/assessments/2026-01-03_1200_assessment-layoutlmv3-kie-receipt-failure.md'
          - 'experiment_manager/experiments/20251122_172313_perspective_correction/artifacts/20251122_1723_assessment_200-image-test-results.md'
          - 'experiment_manager/experiments/20251220_154834_zero_prediction_images_debug/README.md'
          - 'experiment_manager/experiments/20251129_173500_perspective_correction_implementation/.metadata/assessments/20251129_1735_assessment_test-results-analysis.md'

    steps:
      - name: Checkout repository
        uses: actions/checkout@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Translate document
        id: translate
        uses: wchoi189/translation-action@solar
        with:
          provider: upstage
          lang: en-ko
          source: ${{ matrix.file }}
          api_key: ${{ secrets.UPSTAGE_API_KEY }}
          api_additional_parameter: solar-pro2

      - name: Save translation
        run: |
          dir=$(dirname "${{ matrix.file }}")
          base=$(basename "${{ matrix.file }}" .md)
          output="${dir}/${base}.ko.md"
          echo "${{ steps.translate.outputs.text }}" > "$output"

      - name: Commit translation
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add "*.ko.md"
          git diff --staged --quiet || git commit -m "docs: add Korean translation for ${{ matrix.file }} [skip ci]"
          git push
```

### Option 2: On-Demand Translation

Use the existing workflow with manual file input:

1. Go to Actions → "Translate Documentation"
2. Click "Run workflow"
3. Enter comma-separated file paths:
   ```
   docs/artifacts/bug_reports/2026-01-04_1730_BUG_003_config-precedence-leak.md,docs/artifacts/assessments/2026-01-03_1200_assessment-layoutlmv3-kie-receipt-failure.md
   ```

**Note:** Current workflow needs enhancement to support multiple files properly.

### Option 3: Local Script

Create a local script for one-time batch translation:

**File:** `scripts/translate_all_docs.sh`

```bash
#!/bin/bash

FILES=(
  "docs/artifacts/bug_reports/2026-01-04_1730_BUG_003_config-precedence-leak.md"
  "docs/artifacts/assessments/2026-01-03_1200_assessment-layoutlmv3-kie-receipt-failure.md"
  "experiment_manager/experiments/20251122_172313_perspective_correction/artifacts/20251122_1723_assessment_200-image-test-results.md"
  "experiment_manager/experiments/20251220_154834_zero_prediction_images_debug/README.md"
  "experiment_manager/experiments/20251129_173500_perspective_correction_implementation/.metadata/assessments/20251129_1735_assessment_test-results-analysis.md"
)

for file in "${FILES[@]}"; do
  echo "Translating $file..."
  gh workflow run translation-action.yml -f files="$file"
  sleep 5  # Rate limiting
done
```

## Updating README Links

After translations are complete, update README.md to link to Korean versions:

### Current Format
```markdown
**Reference:** [200-Image Test Results](experiment_manager/experiments/20251122_172313_perspective_correction/artifacts/20251122_1723_assessment_200-image-test-results.md)
```

### Bilingual Format
```markdown
**Reference:** [200-Image Test Results](experiment_manager/experiments/20251122_172313_perspective_correction/artifacts/20251122_1723_assessment_200-image-test-results.md) ([한국어](experiment_manager/experiments/20251122_172313_perspective_correction/artifacts/20251122_1723_assessment_200-image-test-results.ko.md))
```

## Cost Estimation

**Upstage Solar Pro 2 Pricing:**
- Typical cost: ~$0.001-0.005 per 1K tokens
- Average document: ~5K-10K tokens
- **Total for 6 documents:** ~$0.05-0.30

## Maintenance Strategy

### Keeping Translations Updated

1. **README.md:** Auto-updates on every push to main
2. **Technical Docs:** Manual trigger when documents are significantly updated
3. **Version Control:** Use git to track when English source was last translated

### Translation Metadata

Add to each translated document:

```markdown
---
translation_date: 2026-01-06
source_file: docs/artifacts/bug_reports/2026-01-04_1730_BUG_003_config-precedence-leak.md
translator: Upstage Solar Pro 2
---
```

## Next Steps

1. ✅ **README translation:** Already configured and working
2. **Create batch workflow:** Implement Option 1 for all technical documents
3. **Test translation:** Run manual workflow to verify output quality
4. **Update README links:** Add bilingual references
5. **Document process:** Add translation guidelines to CONTRIBUTING.md

## Troubleshooting

### Common Issues

1. **API Rate Limits:**
   - Solution: Add delays between translations in batch workflow
   - Use `sleep 10` between API calls

2. **Large Files:**
   - Solar Pro 2 has token limits
   - Solution: Split large documents or use chunking

3. **Special Characters:**
   - Markdown formatting may break
   - Solution: Review and manually fix formatting after translation

4. **Technical Terms:**
   - Some terms should remain in English (e.g., "RemBG", "LMDB")
   - Solution: Post-process to preserve technical terminology

## References

- Translation Action Repo: `wchoi189/translation-action@solar`
- Upstage Solar API Docs: https://console.upstage.ai/docs
- GitHub Actions Secrets: Repository Settings → Secrets and variables → Actions
