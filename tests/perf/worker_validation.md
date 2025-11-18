# Worker Pipeline Performance Validation

This document describes the performance validation process for Phase 2 of the high-performance playground migration.

## Performance Requirements

According to the implementation plan, the following latency targets must be met:

- **Auto Contrast**: < 100ms
- **Gaussian Blur**: < 100ms
- **Client-side rembg**: < 400ms
- **Worker Queue Depth**: < 5 during slider spam

## Backend Performance Validation

### Running Python Benchmarks

The backend preprocessing benchmark can be run using:

```bash
# With sample dataset manifest
python tests/perf/pipeline_bench.py \
  --manifest outputs/playground/sample_manifest.json \
  --limit 16 \
  --output outputs/playground/pipeline_bench.json

# Or with image directory
python tests/perf/pipeline_bench.py \
  --image-dir data/train/images \
  --limit 16 \
  --output outputs/playground/pipeline_bench.json
```

### Expected Results

The benchmark outputs JSON with mean and P95 latencies:

```json
{
  "autocontrast": {
    "mean_ms": 45.2,
    "p95_ms": 78.5
  },
  "gaussian_blur": {
    "mean_ms": 32.1,
    "p95_ms": 54.3
  },
  "rembg_client": {
    "mean_ms": 285.7,
    "p95_ms": 380.2
  }
}
```

**Validation**: All P95 values should be below the target thresholds.

## Frontend Performance Validation

### Manual Testing

1. **Load the Preprocessing Studio**
   ```bash
   python run_spa.py
   ```
   Navigate to http://localhost:5173/preprocessing

2. **Upload a test image** (1024x1024 recommended)

3. **Test Auto Contrast**
   - Enable "Auto Contrast" toggle
   - Observe processing time in UI
   - Verify < 100ms

4. **Test Gaussian Blur**
   - Enable "Gaussian Blur" toggle
   - Move kernel size slider rapidly (slider spam test)
   - Verify < 100ms processing time
   - Verify smooth UI, no freezing

5. **Test Background Removal**
   - Enable "Background Removal" toggle
   - For images < 2048px: verify client-side execution (< 400ms)
   - For images > 2048px: verify backend routing indicated in UI

### Worker Queue Depth Testing

To test worker queue stability during slider spam:

1. Enable Gaussian Blur
2. Rapidly move the kernel size slider back and forth for 5 seconds
3. Open browser DevTools Console
4. Check for worker queue warnings or errors
5. Verify queue depth stays < 5 (can be monitored via workerHost.getQueueDepth())

### Expected Behavior

- **Debouncing**: Slider changes should debounce at 75ms
- **Cancellation**: Previous tasks should be cancelled when params change
- **No UI Blocking**: Main thread should remain responsive
- **Queue Management**: Worker pool should auto-scale from 2 to 4 workers as needed

## Playwright E2E Tests (Future)

When Playwright is configured, add automated tests:

```typescript
// tests/e2e/preprocessing.spec.ts

test('preprocessing performance', async ({ page }) => {
  await page.goto('/preprocessing');

  // Upload test image
  await page.setInputFiles('#image-upload', 'test-fixtures/sample.jpg');

  // Enable auto contrast
  await page.click('text=Auto Contrast');

  // Wait for processing
  await page.waitForSelector('text=Processing time:');

  // Extract latency
  const latencyText = await page.textContent('text=Processing time:');
  const latencyMs = parseFloat(latencyText.match(/(\d+\.?\d*)/)[1]);

  // Verify < 100ms
  expect(latencyMs).toBeLessThan(100);
});

test('slider spam queue depth', async ({ page }) => {
  await page.goto('/preprocessing');
  await page.setInputFiles('#image-upload', 'test-fixtures/sample.jpg');
  await page.click('text=Gaussian Blur');

  // Spam slider
  const slider = page.locator('input[type=range]');
  for (let i = 0; i < 20; i++) {
    await slider.fill(String(3 + (i % 6) * 2));
  }

  // Check queue depth via console
  const queueDepth = await page.evaluate(() => {
    const pool = (window as any).__workerPool__;
    return pool?.getQueueDepth() || 0;
  });

  expect(queueDepth).toBeLessThan(5);
});
```

## Performance Regression Tracking

After each validation run:

1. Record results in `outputs/playground/perf_history.jsonl`
2. Compare against baseline from previous runs
3. Flag any regressions > 20% increase in P95 latency
4. Investigate and fix before merging

## Troubleshooting

### High Latencies

- **Auto Contrast/Blur > 100ms**: Check image resolution, consider resizing before processing
- **rembg > 400ms**: Verify ONNX.js model bundle loaded, check browser console for errors
- **Backend fallback not working**: Check `/api/pipelines/fallback` endpoint response

### Queue Depth Issues

- **Queue depth > 5**: Increase debounce delay from 75ms to 150ms
- **UI freezing**: Check worker pool size, ensure transfers using `[imageBuffer]`
- **Memory leaks**: Verify `ImageBitmap.close()` called after use

## Success Criteria

Phase 2, Task 2.4 is considered complete when:

- [x] Backend benchmark script runs successfully with sample data
- [x] Manual frontend testing shows latencies within targets
- [x] Slider spam test shows queue depth < 5
- [x] No UI blocking or freezing during rapid parameter changes
- [ ] Playwright E2E tests pass (when configured)
- [ ] Performance metrics documented in outputs/

---

**Last Updated**: 2025-11-18
**Status**: Implementation Complete, Automated Testing Pending
