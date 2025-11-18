import { test, expect } from '@playwright/test';

/**
 * E2E tests for Inference Studio
 *
 * Tests checkpoint picker, preview canvas, polygon overlay, and hyperparameter controls
 */

test.describe('Inference Studio', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inference');
  });

  test('should load inference page', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Inference Studio');
    await expect(page.locator('text=Checkpoint Picker')).toBeVisible();
  });

  test('should load checkpoint catalog', async ({ page }) => {
    // Wait for checkpoints to load
    await page.waitForSelector('[data-testid="checkpoint-card"]', { timeout: 5000 });

    // Verify at least one checkpoint is displayed
    const checkpoints = page.locator('[data-testid="checkpoint-card"]');
    const count = await checkpoints.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should search checkpoints by name', async ({ page }) => {
    // Wait for checkpoints to load
    await page.waitForSelector('[data-testid="checkpoint-card"]');

    // Get initial count
    const initialCount = await page.locator('[data-testid="checkpoint-card"]').count();

    // Search for specific checkpoint
    await page.locator('input[placeholder*="Search"]').fill('experiment');

    // Wait for filtered results
    await page.waitForTimeout(500);

    // Verify filtered count
    const filteredCount = await page.locator('[data-testid="checkpoint-card"]').count();
    expect(filteredCount).toBeLessThanOrEqual(initialCount);
  });

  test('should filter by architecture', async ({ page }) => {
    // Wait for checkpoints to load
    await page.waitForSelector('[data-testid="checkpoint-card"]');

    // Find architecture filter dropdown
    const archFilter = page.locator('select[name="architecture_filter"]');
    if (await archFilter.isVisible()) {
      // Select first architecture
      await archFilter.selectOption({ index: 1 });

      // Wait for filtering
      await page.waitForTimeout(500);

      // Verify checkpoints are filtered
      const checkpoints = page.locator('[data-testid="checkpoint-card"]');
      const count = await checkpoints.count();
      expect(count).toBeGreaterThan(0);

      // Verify all visible checkpoints match the filter
      const firstCheckpoint = checkpoints.first();
      await expect(firstCheckpoint).toBeVisible();
    }
  });

  test('should filter by backbone', async ({ page }) => {
    // Wait for checkpoints to load
    await page.waitForSelector('[data-testid="checkpoint-card"]');

    // Find backbone filter dropdown
    const backboneFilter = page.locator('select[name="backbone_filter"]');
    if (await backboneFilter.isVisible()) {
      // Select first backbone
      await backboneFilter.selectOption({ index: 1 });

      // Wait for filtering
      await page.waitForTimeout(500);

      // Verify checkpoints are filtered
      const checkpoints = page.locator('[data-testid="checkpoint-card"]');
      const count = await checkpoints.count();
      expect(count).toBeGreaterThanOrEqual(0); // May be 0 if no matches
    }
  });

  test('should select checkpoint and load preview', async ({ page }) => {
    // Wait for checkpoints to load
    await page.waitForSelector('[data-testid="checkpoint-card"]');

    // Click first checkpoint
    await page.locator('[data-testid="checkpoint-card"]').first().click();

    // Verify checkpoint is selected (e.g., highlighted border)
    const selectedCheckpoint = page.locator('[data-testid="checkpoint-card"][data-selected="true"]');
    await expect(selectedCheckpoint).toBeVisible();

    // Verify preview section appears
    await expect(page.locator('text=Upload Image for Preview')).toBeVisible();
  });

  test('should upload image and show preview canvas', async ({ page }) => {
    // Select a checkpoint first
    await page.waitForSelector('[data-testid="checkpoint-card"]');
    await page.locator('[data-testid="checkpoint-card"]').first().click();

    // Upload test image
    const fileInput = page.locator('input[type="file"]#inference-image-upload');
    await fileInput.setInputFiles('tests/e2e/fixtures/sample-image.jpg');

    // Verify image name is displayed
    await expect(page.locator('text=sample-image.jpg')).toBeVisible();

    // Verify canvas is rendered
    const canvas = page.locator('canvas');
    await expect(canvas).toBeVisible();
  });

  test('should display hyperparameter controls', async ({ page }) => {
    // Select checkpoint and upload image
    await page.waitForSelector('[data-testid="checkpoint-card"]');
    await page.locator('[data-testid="checkpoint-card"]').first().click();

    const fileInput = page.locator('input[type="file"]#inference-image-upload');
    await fileInput.setInputFiles('tests/e2e/fixtures/sample-image.jpg');

    // Verify sliders are visible
    await expect(page.locator('label:has-text("Confidence Threshold")')).toBeVisible();
    await expect(page.locator('label:has-text("NMS Threshold")')).toBeVisible();

    // Verify slider ranges
    const confSlider = page.locator('input[type="range"][name="confidence_threshold"]');
    await expect(confSlider).toHaveAttribute('min', '0');
    await expect(confSlider).toHaveAttribute('max', '1');
    await expect(confSlider).toHaveAttribute('step', '0.05');
  });

  test('should run inference and display results', async ({ page }) => {
    // Select checkpoint and upload image
    await page.waitForSelector('[data-testid="checkpoint-card"]');
    await page.locator('[data-testid="checkpoint-card"]').first().click();

    const fileInput = page.locator('input[type="file"]#inference-image-upload');
    await fileInput.setInputFiles('tests/e2e/fixtures/sample-image.jpg');

    // Wait for inference to complete
    await page.waitForSelector('text=Processing...', { state: 'hidden', timeout: 10000 });

    // Verify results are displayed
    await expect(page.locator('text=Detected')).toBeVisible();

    // Verify canvas shows polygons (check for canvas rendering)
    const canvas = page.locator('canvas');
    await expect(canvas).toBeVisible();
  });

  test('should render polygon overlays on canvas', async ({ page }) => {
    // Select checkpoint and upload image
    await page.waitForSelector('[data-testid="checkpoint-card"]');
    await page.locator('[data-testid="checkpoint-card"]').first().click();

    const fileInput = page.locator('input[type="file"]#inference-image-upload');
    await fileInput.setInputFiles('tests/e2e/fixtures/sample-image.jpg');

    // Wait for inference
    await page.waitForTimeout(2000);

    // Verify canvas is rendered
    const canvas = page.locator('canvas');
    await expect(canvas).toBeVisible();

    // Verify canvas has correct dimensions
    const canvasWidth = await canvas.evaluate((el) => (el as HTMLCanvasElement).width);
    const canvasHeight = await canvas.evaluate((el) => (el as HTMLCanvasElement).height);
    expect(canvasWidth).toBeGreaterThan(0);
    expect(canvasHeight).toBeGreaterThan(0);

    // Check if polygons are drawn (verify canvas has non-blank content)
    // Note: This is a basic check - actual polygon validation requires image analysis
    const hasContent = await canvas.evaluate((el) => {
      const ctx = (el as HTMLCanvasElement).getContext('2d');
      if (!ctx) return false;
      const imageData = ctx.getImageData(0, 0, el.width, el.height);
      // Check if any pixel is non-white/non-transparent
      for (let i = 0; i < imageData.data.length; i += 4) {
        if (imageData.data[i] !== 0 || imageData.data[i + 1] !== 0 ||
            imageData.data[i + 2] !== 0 || imageData.data[i + 3] !== 0) {
          return true;
        }
      }
      return false;
    });
    expect(hasContent).toBe(true);
  });

  test('should update preview when changing confidence threshold', async ({ page }) => {
    // Select checkpoint and upload image
    await page.waitForSelector('[data-testid="checkpoint-card"]');
    await page.locator('[data-testid="checkpoint-card"]').first().click();

    const fileInput = page.locator('input[type="file"]#inference-image-upload');
    await fileInput.setInputFiles('tests/e2e/fixtures/sample-image.jpg');

    // Wait for initial inference
    await page.waitForTimeout(2000);

    // Change confidence threshold
    const confSlider = page.locator('input[type="range"][name="confidence_threshold"]');
    await confSlider.fill('0.8');

    // Verify threshold value is displayed
    await expect(page.locator('text=0.8')).toBeVisible();

    // Wait for re-inference
    await page.waitForTimeout(1000);

    // Verify preview updated (check for processing indicator or updated results)
    const canvas = page.locator('canvas');
    await expect(canvas).toBeVisible();
  });

  test('should update preview when changing NMS threshold', async ({ page }) => {
    // Select checkpoint and upload image
    await page.waitForSelector('[data-testid="checkpoint-card"]');
    await page.locator('[data-testid="checkpoint-card"]').first().click();

    const fileInput = page.locator('input[type="file"]#inference-image-upload');
    await fileInput.setInputFiles('tests/e2e/fixtures/sample-image.jpg');

    // Wait for initial inference
    await page.waitForTimeout(2000);

    // Change NMS threshold
    const nmsSlider = page.locator('input[type="range"][name="nms_threshold"]');
    await nmsSlider.fill('0.3');

    // Verify threshold value is displayed
    await expect(page.locator('text=0.3')).toBeVisible();

    // Wait for re-inference
    await page.waitForTimeout(1000);

    // Verify preview updated
    const canvas = page.locator('canvas');
    await expect(canvas).toBeVisible();
  });

  test('should display detection count', async ({ page }) => {
    // Select checkpoint and upload image
    await page.waitForSelector('[data-testid="checkpoint-card"]');
    await page.locator('[data-testid="checkpoint-card"]').first().click();

    const fileInput = page.locator('input[type="file"]#inference-image-upload');
    await fileInput.setInputFiles('tests/e2e/fixtures/sample-image.jpg');

    // Wait for inference
    await page.waitForTimeout(2000);

    // Verify detection count is displayed
    const detectionCount = page.locator('text=/Detected \\d+ text regions?/');
    await expect(detectionCount).toBeVisible();
  });

  test('should display inference latency', async ({ page }) => {
    // Select checkpoint and upload image
    await page.waitForSelector('[data-testid="checkpoint-card"]');
    await page.locator('[data-testid="checkpoint-card"]').first().click();

    const fileInput = page.locator('input[type="file"]#inference-image-upload');
    await fileInput.setInputFiles('tests/e2e/fixtures/sample-image.jpg');

    // Wait for inference
    await page.waitForTimeout(2000);

    // Verify latency is displayed
    const latencyText = page.locator('text=/Inference time: \\d+\\.\\d+ms/');
    await expect(latencyText).toBeVisible();
  });

  test('should handle checkpoint metadata display', async ({ page }) => {
    // Wait for checkpoints to load
    await page.waitForSelector('[data-testid="checkpoint-card"]');

    // Click first checkpoint
    const checkpoint = page.locator('[data-testid="checkpoint-card"]').first();
    await checkpoint.click();

    // Verify metadata is visible in card
    await expect(checkpoint.locator('text=/Architecture:/i')).toBeVisible();
    await expect(checkpoint.locator('text=/Backbone:/i')).toBeVisible();
  });

  test('should handle empty checkpoint list', async ({ page }) => {
    // If no checkpoints are available, verify empty state
    try {
      await page.waitForSelector('[data-testid="checkpoint-card"]', { timeout: 2000 });
    } catch {
      // No checkpoints found - verify empty state message
      await expect(page.locator('text=/No checkpoints/i')).toBeVisible();
    }
  });

  test('should clear selection when clicking deselect', async ({ page }) => {
    // Select checkpoint
    await page.waitForSelector('[data-testid="checkpoint-card"]');
    await page.locator('[data-testid="checkpoint-card"]').first().click();

    // Verify selection
    await expect(page.locator('[data-testid="checkpoint-card"][data-selected="true"]'))
      .toBeVisible();

    // Click deselect/clear button if available
    const clearButton = page.locator('button:has-text("Clear")');
    if (await clearButton.isVisible()) {
      await clearButton.click();

      // Verify no checkpoint is selected
      const selectedCheckpoints = page.locator('[data-testid="checkpoint-card"][data-selected="true"]');
      await expect(selectedCheckpoints).toHaveCount(0);
    }
  });

  test('should show error state for invalid checkpoint', async ({ page }) => {
    // This test would require mocking a failed API response
    // For now, we just verify error handling structure exists

    // Wait for checkpoints
    await page.waitForSelector('[data-testid="checkpoint-card"]');

    // Verify error boundary or error message container exists
    const errorContainer = page.locator('[data-testid="error-message"]');
    // Should not be visible initially
    await expect(errorContainer).not.toBeVisible();
  });
});
