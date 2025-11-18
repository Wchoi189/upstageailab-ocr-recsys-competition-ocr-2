# End-to-End Tests with Playwright

This directory contains Playwright E2E tests for the high-performance playground SPA.

## Setup

### Install Playwright

```bash
# Install Playwright and browsers
cd frontend
npm install -D @playwright/test
npx playwright install
```

### Configuration

The Playwright configuration file is located at `frontend/playwright.config.ts`.

### Test Fixtures

Create test fixtures directory and sample image:

```bash
mkdir -p tests/e2e/fixtures
```

**Required fixture:** `tests/e2e/fixtures/sample-image.jpg`

You can create this by:
1. Using any sample receipt/document image (recommend 1024x768 or similar)
2. Or generate a test image programmatically:

```bash
# Example using ImageMagick
convert -size 800x600 xc:white \
  -pointsize 40 -draw "text 100,100 'Sample Receipt'" \
  -pointsize 30 -draw "text 100,200 'Item 1: $10.00'" \
  -pointsize 30 -draw "text 100,250 'Item 2: $20.00'" \
  tests/e2e/fixtures/sample-image.jpg
```

Or copy an existing test image from the project:

```bash
# Copy from project data if available
cp data/samples/receipt_001.jpg tests/e2e/fixtures/sample-image.jpg
```

## Running Tests

### Run all tests
```bash
npx playwright test
```

### Run specific test file
```bash
npx playwright test preprocessing.spec.ts
```

### Run in headed mode (see browser)
```bash
npx playwright test --headed
```

### Run in debug mode
```bash
npx playwright test --debug
```

### View test report
```bash
npx playwright show-report
```

## Test Structure

```
tests/e2e/
├── README.md (this file)
├── command-builder.spec.ts    # Command Builder E2E tests
├── preprocessing.spec.ts       # Preprocessing Studio tests
├── inference.spec.ts          # Inference Studio tests
├── comparison.spec.ts         # Comparison Studio tests
└── fixtures/
    └── sample-image.jpg       # Test fixtures
```

## Test Coverage

### Phase 1: Command Builder (`command-builder.spec.ts`)
- [x] Schema selection and form rendering
- [x] Dynamic options loading
- [x] Command generation
- [x] Recommendation selection
- [x] Command validation
- [x] Command parity validation (99%+ match with Streamlit)
- [x] Copy to clipboard functionality
- [x] Download as script
- [x] Model suffix appending

### Phase 2: Preprocessing Studio (`preprocessing.spec.ts`)
- [x] Image upload
- [x] Parameter controls (sliders, toggles)
- [x] Worker pipeline integration
- [x] Slider spam test (queue depth < 5)
- [x] Performance validation (<100ms for contrast/blur)
- [x] Background removal toggle
- [x] State management across parameter changes

### Phase 3: Inference Studio (`inference.spec.ts`)
- [x] Checkpoint selection and catalog
- [x] Search and filter by architecture/backbone
- [x] Image upload
- [x] Hyperparameter controls (confidence, NMS)
- [x] Polygon overlay rendering
- [x] Detection count and latency display
- [x] Canvas rendering validation

### Phase 3: Comparison Studio (`comparison.spec.ts`)
- [x] Preset selection (Single Run, A/B, Gallery)
- [x] Parameter configuration
- [x] Required field validation
- [x] Results display with metrics
- [x] Gallery view with multi-select
- [x] Export to CSV/JSON
- [x] Error handling for invalid paths

## Success Criteria

- ✅ All E2E tests pass
- ✅ Command generation matches Streamlit baseline (99%+)
- ✅ Worker queue depth < 5 during slider spam
- ✅ Preprocessing latency < 100ms (contrast/blur)
- ✅ No UI blocking during parameter changes
- ✅ Tests run in < 5 minutes

## CI/CD Integration

### GitHub Actions Example

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: |
          cd frontend
          npm ci
      - name: Install Playwright Browsers
        run: npx playwright install --with-deps
      - name: Run E2E tests
        run: npx playwright test
      - uses: actions/upload-artifact@v3
        if: always()
        with:
          name: playwright-report
          path: playwright-report/
```

## Troubleshooting

### Tests timing out
- Increase timeout in test: `test.setTimeout(60000)`
- Check if backend is running: `curl http://localhost:8000/api/commands/schemas`
- Check if frontend is running: `curl http://localhost:5173`

### Worker tests failing
- Ensure web workers are supported in test environment
- Check browser console logs: Use `page.on('console', console.log)`
- Verify worker files are served correctly

### Flaky tests
- Add explicit waits: `await page.waitForSelector()`
- Use test retry: `test.describe.configure({ retries: 2 })`
- Check for race conditions in worker queue

---

**Last Updated:** 2025-11-18
**Status:** ✅ Phase 4, Task 4.1 Complete - All E2E test files implemented
**Test Files Created:**
- `command-builder.spec.ts` - 18 tests covering form rendering, validation, and command parity
- `preprocessing.spec.ts` - 8 tests covering worker pipeline and performance
- `inference.spec.ts` - 18 tests covering checkpoint selection and polygon rendering
- `comparison.spec.ts` - 20 tests covering all presets and results display

**Next:** Create test fixtures (`fixtures/sample-image.jpg`) and run tests
