/**
 * Analytics tracking utilities for Google Tag Manager integration
 */

declare global {
  interface Window {
    dataLayer?: Array<Record<string, unknown>>;
  }
}

/**
 * Track a custom event in Google Tag Manager
 *
 * Only pushes events if:
 * 1. User has accepted consent
 * 2. GTM is loaded (dataLayer exists)
 *
 * @param event - Event name (e.g., "command_built", "image_uploaded")
 * @param properties - Additional event properties
 *
 * @example
 * ```ts
 * trackEvent("command_built", {
 *   schema_id: "donut-v2",
 *   command_length: 1234
 * });
 * ```
 */
export function trackEvent(
  event: string,
  properties?: Record<string, unknown>
): void {
  // Check if user has consented
  const consent = typeof window !== "undefined"
    ? localStorage.getItem("cookie-consent")
    : null;

  if (consent !== "accepted") {
    return; // Don't track if consent not given
  }

  // Check if GTM is loaded
  if (typeof window === "undefined" || !window.dataLayer) {
    return;
  }

  // Push event to dataLayer
  try {
    window.dataLayer.push({
      event,
      ...properties,
    });
  } catch (error) {
    console.error("Failed to track event:", event, error);
  }
}

/**
 * Track page view event
 *
 * @param path - Page path
 * @param title - Page title (optional)
 */
export function trackPageView(path: string, title?: string): void {
  trackEvent("page_view", {
    page_path: path,
    page_title: title,
  });
}

/**
 * Track command build event
 *
 * @param schemaId - Schema ID used for the command
 * @param commandLength - Length of the generated command
 */
export function trackCommandBuild(schemaId: string, commandLength: number): void {
  trackEvent("command_built", {
    schema_id: schemaId,
    command_length: commandLength,
  });
}

/**
 * Track image upload event
 *
 * @param fileSize - Size of uploaded file in bytes
 * @param fileType - MIME type of the file
 */
export function trackImageUpload(fileSize: number, fileType: string): void {
  trackEvent("image_uploaded", {
    file_size: fileSize,
    file_type: fileType,
  });
}

/**
 * Track inference run event
 *
 * @param checkpointId - Checkpoint ID used for inference
 * @param duration - Duration of inference in milliseconds (optional)
 */
export function trackInferenceRun(checkpointId: string, duration?: number): void {
  trackEvent("inference_run", {
    checkpoint_id: checkpointId,
    duration_ms: duration,
  });
}
