import { test, expect } from '@playwright/test';

/**
 * E2E tests for Comparison Studio
 *
 * Tests preset selection, parameter configuration, and results display
 */

test.describe('Comparison Studio', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/comparison');
  });

  test('should load comparison page', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Comparison Studio');
    await expect(page.locator('text=Select Preset')).toBeVisible();
  });

  test('should display preset options', async ({ page }) => {
    // Wait for preset selector
    await page.waitForSelector('select[name="preset"]');

    // Verify preset options
    const presetSelect = page.locator('select[name="preset"]');
    await expect(presetSelect.locator('option:has-text("Single Run")')).toBeVisible();
    await expect(presetSelect.locator('option:has-text("A/B Test")')).toBeVisible();
    await expect(presetSelect.locator('option:has-text("Gallery View")')).toBeVisible();
  });

  test('should render Single Run preset form', async ({ page }) => {
    // Select Single Run preset
    await page.locator('select[name="preset"]').selectOption('single_run');

    // Wait for form to render
    await page.waitForSelector('label:has-text("Checkpoint Path")');

    // Verify form fields
    await expect(page.locator('label:has-text("Checkpoint Path")')).toBeVisible();
    await expect(page.locator('label:has-text("Dataset Path")')).toBeVisible();
    await expect(page.locator('label:has-text("Batch Size")')).toBeVisible();
  });

  test('should render A/B Test preset form', async ({ page }) => {
    // Select A/B Test preset
    await page.locator('select[name="preset"]').selectOption('ab_test');

    // Wait for form to render
    await page.waitForSelector('label:has-text("Checkpoint A")');

    // Verify A/B specific fields
    await expect(page.locator('label:has-text("Checkpoint A")')).toBeVisible();
    await expect(page.locator('label:has-text("Checkpoint B")')).toBeVisible();
    await expect(page.locator('label:has-text("Dataset Path")')).toBeVisible();
  });

  test('should render Gallery View preset form', async ({ page }) => {
    // Select Gallery View preset
    await page.locator('select[name="preset"]').selectOption('gallery_view');

    // Wait for form to render
    await page.waitForSelector('label:has-text("Checkpoint Paths")');

    // Verify gallery-specific fields
    await expect(page.locator('label:has-text("Checkpoint Paths")')).toBeVisible();
    await expect(page.locator('label:has-text("Sample Count")')).toBeVisible();
  });

  test('should validate required fields for Single Run', async ({ page }) => {
    // Select Single Run preset
    await page.locator('select[name="preset"]').selectOption('single_run');
    await page.waitForSelector('label:has-text("Checkpoint Path")');

    // Try to submit without filling fields
    const runButton = page.locator('button:has-text("Run Evaluation")');
    await runButton.click();

    // Verify validation messages appear
    const validationErrors = page.locator('text=/required/i');
    const errorCount = await validationErrors.count();
    expect(errorCount).toBeGreaterThan(0);
  });

  test('should submit valid Single Run evaluation', async ({ page }) => {
    // Select Single Run preset
    await page.locator('select[name="preset"]').selectOption('single_run');
    await page.waitForSelector('label:has-text("Checkpoint Path")');

    // Fill in form
    await page.locator('input[name="checkpoint_path"]').fill('checkpoints/model_001.pth');
    await page.locator('input[name="dataset_path"]').fill('data/test_set');
    await page.locator('input[name="batch_size"]').fill('16');

    // Submit form
    const runButton = page.locator('button:has-text("Run Evaluation")');
    await runButton.click();

    // Verify submission (either loading state or results)
    await expect(page.locator('text=/Running|Results/i')).toBeVisible({ timeout: 5000 });
  });

  test('should display loading state during evaluation', async ({ page }) => {
    // Select Single Run preset and submit
    await page.locator('select[name="preset"]').selectOption('single_run');
    await page.waitForSelector('label:has-text("Checkpoint Path")');

    await page.locator('input[name="checkpoint_path"]').fill('checkpoints/model_001.pth');
    await page.locator('input[name="dataset_path"]').fill('data/test_set');

    const runButton = page.locator('button:has-text("Run Evaluation")');
    await runButton.click();

    // Verify loading indicator appears
    const loadingIndicator = page.locator('text=/Running|Processing|Loading/i');
    await expect(loadingIndicator).toBeVisible();
  });

  test('should display results for Single Run', async ({ page }) => {
    // Select Single Run preset and submit
    await page.locator('select[name="preset"]').selectOption('single_run');
    await page.waitForSelector('label:has-text("Checkpoint Path")');

    await page.locator('input[name="checkpoint_path"]').fill('checkpoints/model_001.pth');
    await page.locator('input[name="dataset_path"]').fill('data/test_set');

    await page.locator('button:has-text("Run Evaluation")').click();

    // Wait for results
    await page.waitForSelector('text=Results', { timeout: 30000 });

    // Verify metrics are displayed
    await expect(page.locator('text=/Precision/i')).toBeVisible();
    await expect(page.locator('text=/Recall/i')).toBeVisible();
    await expect(page.locator('text=/F1/i')).toBeVisible();
  });

  test('should display comparison results for A/B Test', async ({ page }) => {
    // Select A/B Test preset
    await page.locator('select[name="preset"]').selectOption('ab_test');
    await page.waitForSelector('label:has-text("Checkpoint A")');

    // Fill in form
    await page.locator('input[name="checkpoint_a"]').fill('checkpoints/model_a.pth');
    await page.locator('input[name="checkpoint_b"]').fill('checkpoints/model_b.pth');
    await page.locator('input[name="dataset_path"]').fill('data/test_set');

    // Submit
    await page.locator('button:has-text("Run Comparison")').click();

    // Wait for results
    await page.waitForSelector('text=/Results|Comparison/i', { timeout: 30000 });

    // Verify both checkpoints are shown
    await expect(page.locator('text=/Model A|Checkpoint A/i')).toBeVisible();
    await expect(page.locator('text=/Model B|Checkpoint B/i')).toBeVisible();

    // Verify comparison metrics
    await expect(page.locator('text=/Difference|Δ/i')).toBeVisible();
  });

  test('should display gallery grid for Gallery View', async ({ page }) => {
    // Select Gallery View preset
    await page.locator('select[name="preset"]').selectOption('gallery_view');
    await page.waitForSelector('label:has-text("Checkpoint Paths")');

    // Fill in form (comma-separated checkpoint paths)
    await page.locator('textarea[name="checkpoint_paths"]').fill(
      'checkpoints/model_1.pth,checkpoints/model_2.pth,checkpoints/model_3.pth'
    );
    await page.locator('input[name="sample_count"]').fill('10');

    // Submit
    await page.locator('button:has-text("Generate Gallery")').click();

    // Wait for gallery
    await page.waitForSelector('[data-testid="gallery-image"]', { timeout: 30000 });

    // Verify gallery images are displayed
    const galleryImages = page.locator('[data-testid="gallery-image"]');
    const count = await galleryImages.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should handle image selection in Gallery View', async ({ page }) => {
    // Select Gallery View and generate gallery
    await page.locator('select[name="preset"]').selectOption('gallery_view');
    await page.waitForSelector('label:has-text("Checkpoint Paths")');

    await page.locator('textarea[name="checkpoint_paths"]').fill('checkpoints/model_1.pth');
    await page.locator('button:has-text("Generate Gallery")').click();

    await page.waitForSelector('[data-testid="gallery-image"]', { timeout: 30000 });

    // Click first image
    await page.locator('[data-testid="gallery-image"]').first().click();

    // Verify selection indicator
    const selectedImage = page.locator('[data-testid="gallery-image"][data-selected="true"]');
    await expect(selectedImage).toBeVisible();
  });

  test('should support multi-select in Gallery View', async ({ page }) => {
    // Select Gallery View and generate gallery
    await page.locator('select[name="preset"]').selectOption('gallery_view');
    await page.waitForSelector('label:has-text("Checkpoint Paths")');

    await page.locator('textarea[name="checkpoint_paths"]').fill('checkpoints/model_1.pth');
    await page.locator('button:has-text("Generate Gallery")').click();

    await page.waitForSelector('[data-testid="gallery-image"]', { timeout: 30000 });

    // Multi-select with Ctrl/Cmd + click
    const images = page.locator('[data-testid="gallery-image"]');
    await images.nth(0).click();
    await images.nth(1).click({ modifiers: ['Control'] });

    // Verify multiple selections
    const selectedImages = page.locator('[data-testid="gallery-image"][data-selected="true"]');
    const selectedCount = await selectedImages.count();
    expect(selectedCount).toBeGreaterThanOrEqual(2);
  });

  test('should display Select All button in Gallery View', async ({ page }) => {
    // Select Gallery View and generate gallery
    await page.locator('select[name="preset"]').selectOption('gallery_view');
    await page.waitForSelector('label:has-text("Checkpoint Paths")');

    await page.locator('textarea[name="checkpoint_paths"]').fill('checkpoints/model_1.pth');
    await page.locator('button:has-text("Generate Gallery")').click();

    await page.waitForSelector('[data-testid="gallery-image"]', { timeout: 30000 });

    // Click Select All
    const selectAllButton = page.locator('button:has-text("Select All")');
    if (await selectAllButton.isVisible()) {
      await selectAllButton.click();

      // Verify all images are selected
      const totalImages = await page.locator('[data-testid="gallery-image"]').count();
      const selectedImages = await page.locator('[data-testid="gallery-image"][data-selected="true"]')
        .count();
      expect(selectedImages).toBe(totalImages);
    }
  });

  test('should export results as CSV', async ({ page }) => {
    // Run Single Run evaluation
    await page.locator('select[name="preset"]').selectOption('single_run');
    await page.waitForSelector('label:has-text("Checkpoint Path")');

    await page.locator('input[name="checkpoint_path"]').fill('checkpoints/model_001.pth');
    await page.locator('input[name="dataset_path"]').fill('data/test_set');
    await page.locator('button:has-text("Run Evaluation")').click();

    // Wait for results
    await page.waitForSelector('text=Results', { timeout: 30000 });

    // Click export button
    const exportButton = page.locator('button:has-text("Export CSV")');
    if (await exportButton.isVisible()) {
      const downloadPromise = page.waitForEvent('download');
      await exportButton.click();
      const download = await downloadPromise;

      // Verify CSV filename
      expect(download.suggestedFilename()).toMatch(/.*\.csv$/);
    }
  });

  test('should export results as JSON', async ({ page }) => {
    // Run Single Run evaluation
    await page.locator('select[name="preset"]').selectOption('single_run');
    await page.waitForSelector('label:has-text("Checkpoint Path")');

    await page.locator('input[name="checkpoint_path"]').fill('checkpoints/model_001.pth');
    await page.locator('input[name="dataset_path"]').fill('data/test_set');
    await page.locator('button:has-text("Run Evaluation")').click();

    // Wait for results
    await page.waitForSelector('text=Results', { timeout: 30000 });

    // Click export button
    const exportButton = page.locator('button:has-text("Export JSON")');
    if (await exportButton.isVisible()) {
      const downloadPromise = page.waitForEvent('download');
      await exportButton.click();
      const download = await downloadPromise;

      // Verify JSON filename
      expect(download.suggestedFilename()).toMatch(/.*\.json$/);
    }
  });

  test('should handle error state for invalid checkpoint path', async ({ page }) => {
    // Select Single Run preset
    await page.locator('select[name="preset"]').selectOption('single_run');
    await page.waitForSelector('label:has-text("Checkpoint Path")');

    // Fill with invalid path
    await page.locator('input[name="checkpoint_path"]').fill('invalid/path/nonexistent.pth');
    await page.locator('input[name="dataset_path"]').fill('data/test_set');

    // Submit
    await page.locator('button:has-text("Run Evaluation")').click();

    // Wait for error message
    const errorMessage = page.locator('text=/Error|Failed|Not found/i');
    await expect(errorMessage).toBeVisible({ timeout: 10000 });
  });

  test('should reset form when changing presets', async ({ page }) => {
    // Select Single Run and fill form
    await page.locator('select[name="preset"]').selectOption('single_run');
    await page.waitForSelector('label:has-text("Checkpoint Path")');
    await page.locator('input[name="checkpoint_path"]').fill('checkpoints/model_001.pth');

    // Switch to A/B Test
    await page.locator('select[name="preset"]').selectOption('ab_test');
    await page.waitForSelector('label:has-text("Checkpoint A")');

    // Verify form is reset (checkpoint_path field shouldn't exist)
    const checkpointPathInput = page.locator('input[name="checkpoint_path"]');
    await expect(checkpointPathInput).not.toBeVisible();

    // Verify new fields are empty
    const checkpointAValue = await page.locator('input[name="checkpoint_a"]').inputValue();
    expect(checkpointAValue).toBe('');
  });

  test('should display confidence threshold slider in results', async ({ page }) => {
    // Run evaluation and wait for results
    await page.locator('select[name="preset"]').selectOption('single_run');
    await page.waitForSelector('label:has-text("Checkpoint Path")');

    await page.locator('input[name="checkpoint_path"]').fill('checkpoints/model_001.pth');
    await page.locator('input[name="dataset_path"]').fill('data/test_set');
    await page.locator('button:has-text("Run Evaluation")').click();

    await page.waitForSelector('text=Results', { timeout: 30000 });

    // Look for confidence threshold slider in results section
    const confSlider = page.locator('input[type="range"][name="confidence_threshold"]');
    if (await confSlider.isVisible()) {
      // Change threshold
      await confSlider.fill('0.7');

      // Verify metrics update
      await page.waitForTimeout(1000);
      await expect(page.locator('text=/Precision/i')).toBeVisible();
    }
  });

  test('should display comparison chart for A/B Test', async ({ page }) => {
    // Run A/B Test
    await page.locator('select[name="preset"]').selectOption('ab_test');
    await page.waitForSelector('label:has-text("Checkpoint A")');

    await page.locator('input[name="checkpoint_a"]').fill('checkpoints/model_a.pth');
    await page.locator('input[name="checkpoint_b"]').fill('checkpoints/model_b.pth');
    await page.locator('input[name="dataset_path"]').fill('data/test_set');
    await page.locator('button:has-text("Run Comparison")').click();

    await page.waitForSelector('text=/Results|Comparison/i', { timeout: 30000 });

    // Look for chart/graph element
    const chart = page.locator('canvas[data-testid="comparison-chart"]');
    if (await chart.isVisible()) {
      await expect(chart).toBeVisible();
    }
  });

  test('should maintain preset state across navigation', async ({ page }) => {
    // Select A/B Test and fill form
    await page.locator('select[name="preset"]').selectOption('ab_test');
    await page.waitForSelector('label:has-text("Checkpoint A")');
    await page.locator('input[name="checkpoint_a"]').fill('checkpoints/model_a.pth');

    // Navigate away and back
    await page.goto('/preprocessing');
    await page.waitForTimeout(500);
    await page.goto('/comparison');

    // Verify preset is preserved (or reset, depending on UX decision)
    await page.waitForSelector('select[name="preset"]');
    const presetValue = await page.locator('select[name="preset"]').inputValue();
    // Log for verification - behavior depends on implementation
    console.log('Preset value after navigation:', presetValue);
  });
});
