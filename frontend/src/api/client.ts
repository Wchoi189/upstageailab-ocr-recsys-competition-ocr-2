/**
 * API client for FastAPI backend communication.
 * Provides typed fetch wrappers for all playground endpoints.
 */

/**
 * Retry configuration
 */
interface RetryConfig {
  maxRetries?: number;
  retryDelay?: number;
  retryableStatuses?: number[];
}

const DEFAULT_RETRY_CONFIG: Required<RetryConfig> = {
  maxRetries: 2,
  retryDelay: 1000,
  retryableStatuses: [408, 429, 500, 502, 503, 504],
};

/**
 * Sleep utility for retry delays
 */
async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public response?: unknown
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface ApiResponse<T> {
  data: T;
  status: number;
}

/**
 * Retry wrapper for fetch operations
 */
async function withRetry<T>(
  fn: () => Promise<T>,
  config: RetryConfig = {}
): Promise<T> {
  const { maxRetries, retryDelay, retryableStatuses } = {
    ...DEFAULT_RETRY_CONFIG,
    ...config,
  };

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err as Error;

      // Don't retry on last attempt
      if (attempt === maxRetries) break;

      // Check if error is retryable
      if (err instanceof ApiError) {
        if (!retryableStatuses.includes(err.status)) {
          throw err;
        }
      } else if (
        !(err instanceof Error && err.name === "AbortError") &&
        !(err instanceof TypeError)
      ) {
        throw err;
      }

      // Wait before retrying (exponential backoff)
      await sleep(retryDelay * Math.pow(2, attempt));
    }
  }

  throw lastError;
}

async function handleResponse<T>(response: Response): Promise<ApiResponse<T>> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
      response.status,
      errorData
    );
  }

  const data = await response.json();
  return { data, status: response.status };
}

/**
 * GET request with typed response (with retry)
 */
export async function apiGet<T>(
  endpoint: string,
  retryConfig?: RetryConfig
): Promise<T> {
  return withRetry(async () => {
    try {
      // Add timeout to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      const response = await fetch(`/api${endpoint}`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      const { data } = await handleResponse<T>(response);
      return data;
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        throw new ApiError(
          "Request timed out. Server may not be running at http://127.0.0.1:8000",
          0,
          { networkError: true, timeout: true }
        );
      }
      if (
        err instanceof TypeError &&
        (err.message.includes("fetch") || err.message.includes("Failed to fetch"))
      ) {
        throw new ApiError(
          "Failed to connect to server at http://127.0.0.1:8000",
          0,
          { networkError: true }
        );
      }
      throw err;
    }
  }, retryConfig);
}

/**
 * POST request with typed request and response (with retry)
 */
export async function apiPost<TRequest, TResponse>(
  endpoint: string,
  body: TRequest,
  retryConfig?: RetryConfig
): Promise<TResponse> {
  return withRetry(async () => {
    try {
      // Add timeout to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      const response = await fetch(`/api${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      const { data } = await handleResponse<TResponse>(response);
      return data;
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        throw new ApiError(
          "Request timed out. Server may not be running at http://127.0.0.1:8000",
          0,
          { networkError: true, timeout: true }
        );
      }
      if (
        err instanceof TypeError &&
        (err.message.includes("fetch") || err.message.includes("Failed to fetch"))
      ) {
        throw new ApiError(
          "Failed to connect to server at http://127.0.0.1:8000",
          0,
          { networkError: true }
        );
      }
      throw err;
    }
  }, retryConfig);
}

/**
 * PUT request with typed request and response
 */
export async function apiPut<TRequest, TResponse>(
  endpoint: string,
  body: TRequest
): Promise<TResponse> {
  const response = await fetch(`/api${endpoint}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const { data } = await handleResponse<TResponse>(response);
  return data;
}

/**
 * DELETE request with typed response
 */
export async function apiDelete<T>(endpoint: string): Promise<T> {
  const response = await fetch(`/api${endpoint}`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
  });
  const { data } = await handleResponse<T>(response);
  return data;
}
