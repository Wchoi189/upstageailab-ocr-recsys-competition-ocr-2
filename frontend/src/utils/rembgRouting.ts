/**
 * Heuristics for rembg hybrid routing
 */

/**
 * Image size threshold for routing decision (2048px)
 */
const IMAGE_SIZE_THRESHOLD = 2048;

/**
 * Expected latency threshold for client-side rembg (400ms)
 */
const LATENCY_THRESHOLD_MS = 400;

/**
 * Result of routing decision
 */
export interface RoutingDecision {
  useBackend: boolean;
  reason: string;
}

/**
 * Decide whether to route rembg to backend or client
 *
 * Routes to backend if:
 * - Image width or height exceeds 2048px
 * - Previous client execution exceeded latency threshold
 *
 * @param imageWidth - Image width in pixels
 * @param imageHeight - Image height in pixels
 * @param previousLatencyMs - Previous execution latency (optional)
 * @returns Routing decision with reason
 */
export function decideRembgRouting(
  imageWidth: number,
  imageHeight: number,
  previousLatencyMs?: number,
): RoutingDecision {
  // Check image size
  if (imageWidth > IMAGE_SIZE_THRESHOLD || imageHeight > IMAGE_SIZE_THRESHOLD) {
    return {
      useBackend: true,
      reason: `Image size ${imageWidth}x${imageHeight} exceeds ${IMAGE_SIZE_THRESHOLD}px threshold`,
    };
  }

  // Check previous latency
  if (
    previousLatencyMs !== undefined &&
    previousLatencyMs > LATENCY_THRESHOLD_MS
  ) {
    return {
      useBackend: true,
      reason: `Previous latency ${previousLatencyMs}ms exceeded ${LATENCY_THRESHOLD_MS}ms threshold`,
    };
  }

  return {
    useBackend: false,
    reason: "Image size and latency within client-side thresholds",
  };
}

/**
 * Simple LRU cache for processed images
 */
export class ImageCache<T> {
  private cache = new Map<string, { value: T; timestamp: number }>();
  private maxSize: number;
  private maxAgeMs: number;

  constructor(maxSize = 50, maxAgeMs = 5 * 60 * 1000) {
    this.maxSize = maxSize;
    this.maxAgeMs = maxAgeMs;
  }

  /**
   * Get cached value if exists and not expired
   */
  get(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    const age = Date.now() - entry.timestamp;
    if (age > this.maxAgeMs) {
      this.cache.delete(key);
      return null;
    }

    return entry.value;
  }

  /**
   * Set cached value
   */
  set(key: string, value: T): void {
    // Evict oldest entry if cache is full
    if (this.cache.size >= this.maxSize) {
      const oldestKey = Array.from(this.cache.entries()).sort(
        (a, b) => a[1].timestamp - b[1].timestamp,
      )[0]?.[0];
      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }

    this.cache.set(key, { value, timestamp: Date.now() });
  }

  /**
   * Clear cache
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Get cache size
   */
  size(): number {
    return this.cache.size;
  }
}
