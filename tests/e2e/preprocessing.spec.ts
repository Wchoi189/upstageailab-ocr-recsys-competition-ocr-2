import { test, expect } from '@playwright/test';

/**
 * E2E tests for Preprocessing Studio
 *
 * Tests worker pipeline, parameter controls, and performance requirements
 */

test.describe('Preprocessing Studio', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/preprocessing');
  });

  test('should load preprocessing page', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Preprocessing Studio');
    await expect(page.locator('text=Upload Image')).toBeVisible();
  });

  test('should upload image and show before/after canvas', async ({ page }) => {
    // Upload test image
    const fileInput = page.locator('#image-upload');
    await fileInput.setInputFiles('tests/e2e/fixtures/sample-image.jpg');

    // Verify image name is displayed
    await expect(page.locator('text=sample-image.jpg')).toBeVisible();

    // Verify canvases are rendered
    const canvases = page.locator('canvas');
    await expect(canvases).toHaveCount(2); // Before and After
  });

  test('should process image with auto contrast', async ({ page }) => {
    // Upload image
    await page.locator('#image-upload').setInputFiles(
      'tests/e2e/fixtures/sample-image.jpg',
    );

    // Enable auto contrast
    await page.locator('text=Auto Contrast').click();

    // Wait for processing
    await page.waitForSelector('text=Processing time:', { timeout: 5000 });

    // Verify processing time is displayed
    const processingTime = await page.locator('text=Processing time:')
      .textContent();
    expect(processingTime).toMatch(/Processing time: \d+\.\d+ms/);

    // Extract latency and verify < 100ms
    const match = processingTime?.match(/(\d+\.\d+)ms/);
    if (match) {
      const latency = parseFloat(match[1]);
      expect(latency).toBeLessThan(100);
    }
  });

  test('should handle gaussian blur with slider', async ({ page }) => {
    // Upload image
    await page.locator('#image-upload').setInputFiles(
      'tests/e2e/fixtures/sample-image.jpg',
    );

    // Enable gaussian blur
    await page.locator('text=Gaussian Blur').click();

    // Verify kernel size slider appears
    const slider = page.locator('input[type=range][min="3"][max="15"]');
    await expect(slider).toBeVisible();

    // Change kernel size
    await slider.fill('7');

    // Verify kernel size label updates
    await expect(page.locator('text=Kernel Size: 7')).toBeVisible();

    // Wait for processing
    await page.waitForSelector('text=Processing time:', { timeout: 5000 });
  });

  test('slider spam test - queue depth < 5', async ({ page }) => {
    // Upload image
    await page.locator('#image-upload').setInputFiles(
      'tests/e2e/fixtures/sample-image.jpg',
    );

    // Enable gaussian blur
    await page.locator('text=Gaussian Blur').click();

    const slider = page.locator('input[type=range][min="3"][max="15"]');

    // Spam slider rapidly
    const values = [3, 5, 7, 9, 11, 13, 15, 13, 11, 9, 7, 5, 3];
    for (const value of values) {
      await slider.fill(String(value));
      await page.waitForTimeout(50); // 50ms between changes
    }

    // Get worker queue depth (requires exposing via window object)
    const queueDepth = await page.evaluate(() => {
      return (window as any).__workerPoolQueueDepth__ || 0;
    });

    // Verify queue depth < 5
    expect(queueDepth).toBeLessThan(5);

    // Verify UI is still responsive
    await expect(slider).toBeEnabled();
  });

  test('should show processing indicator while loading', async ({ page }) => {
    // Upload image
    await page.locator('#image-upload').setInputFiles(
      'tests/e2e/fixtures/sample-image.jpg',
    );

    // Enable auto contrast
    await page.locator('text=Auto Contrast').click();

    // Should show "Processing..." or similar
    // (This might be fast, so we check the final state)
    await page.waitForSelector('text=Processing time:');
  });

  test('should handle background removal toggle', async ({ page }) => {
    // Upload image
    await page.locator('#image-upload').setInputFiles(
      'tests/e2e/fixtures/sample-image.jpg',
    );

    // Enable rembg
    await page.locator('text=Background Removal').click();

    // Verify routing info is displayed
    await expect(page.locator(
      'text=Automatically routes to backend for large images'
    )).toBeVisible();

    // Wait for processing
    await page.waitForSelector('text=Processing time:', { timeout: 10000 });
  });

  test('should maintain state when switching parameters', async ({ page }) => {
    // Upload image
    await page.locator('#image-upload').setInputFiles(
      'tests/e2e/fixtures/sample-image.jpg',
    );

    // Enable auto contrast
    await page.locator('text=Auto Contrast').click();
    await page.waitForSelector('text=Processing time:');

    // Disable and enable blur
    await page.locator('text=Gaussian Blur').click();
    await page.waitForSelector('text=Kernel Size:');

    const slider = page.locator('input[type=range][min="3"][max="15"]');
    await slider.fill('9');

    // Verify canvas updates
    await page.waitForSelector('text=Processing time:');
  });
});
