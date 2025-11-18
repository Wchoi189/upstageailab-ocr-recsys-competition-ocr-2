/**
 * API client for FastAPI backend communication.
 * Provides typed fetch wrappers for all playground endpoints.
 */

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
 * GET request with typed response
 */
export async function apiGet<T>(endpoint: string): Promise<T> {
  try {
    // Add timeout to prevent hanging
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

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
        "Request timed out. The API server may not be running. Make sure it's running at http://127.0.0.1:8000",
        0,
        { networkError: true, timeout: true }
      );
    }
    if (err instanceof TypeError && (err.message.includes("fetch") || err.message.includes("Failed to fetch"))) {
      throw new ApiError(
        "Failed to connect to API server. Make sure it's running at http://127.0.0.1:8000",
        0,
        { networkError: true }
      );
    }
    throw err;
  }
}

/**
 * POST request with typed request and response
 */
export async function apiPost<TRequest, TResponse>(
  endpoint: string,
  body: TRequest
): Promise<TResponse> {
  try {
    // Add timeout to prevent hanging
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

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
        "Request timed out. The API server may not be running. Make sure it's running at http://127.0.0.1:8000",
        0,
        { networkError: true, timeout: true }
      );
    }
    if (err instanceof TypeError && (err.message.includes("fetch") || err.message.includes("Failed to fetch"))) {
      throw new ApiError(
        "Failed to connect to API server. Make sure it's running at http://127.0.0.1:8000",
        0,
        { networkError: true }
      );
    }
    throw err;
  }
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
