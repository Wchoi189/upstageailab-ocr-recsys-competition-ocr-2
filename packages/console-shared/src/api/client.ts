/**
 * Base API client wrapper
 */
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export class ApiError extends Error {
    constructor(
        public status: number,
        message: string,
    ) {
        super(message);
        this.name = 'ApiError';
    }
}

interface RequestOptions extends RequestInit {
    maxRetries?: number;
    retryDelay?: number;
}

async function fetchWithRetry(url: string, options: RequestOptions = {}): Promise<Response> {
    const { maxRetries = 0, retryDelay = 1000, ...fetchOptions } = options;
    let lastError: Error | unknown;

    for (let i = 0; i <= maxRetries; i++) {
        try {
            const response = await fetch(url, fetchOptions);
            if (response.status === 503 && i < maxRetries) {
                // Retry on service unavailable (e.g. model loading)
                await new Promise((resolve) => setTimeout(resolve, retryDelay));
                continue;
            }
            return response;
        } catch (error) {
            lastError = error;
            if (i < maxRetries) {
                await new Promise((resolve) => setTimeout(resolve, retryDelay));
                continue;
            }
        }
    }

    throw lastError;
}

export async function apiRequest<T>(
    endpoint: string,
    options: RequestOptions = {},
): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;    let response: Response;
    try {
        response = await fetchWithRetry(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
        });    } catch (error: any) {        throw error;
    }

    if (!response.ok) {
        let errorMessage = `API request failed: ${response.statusText}`;
        try {
            const errorData = await response.json();
            errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch {
            // Ignore JSON parse error
        }        throw new ApiError(response.status, errorMessage);
    }

    return response.json();
}

export async function apiGet<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
    return apiRequest<T>(endpoint, { ...options, method: 'GET' });
}

export async function apiPost<Req, Res>(
    endpoint: string,
    body: Req,
    options: RequestOptions = {},
): Promise<Res> {
    return apiRequest<Res>(endpoint, {
        ...options,
        method: 'POST',
        body: JSON.stringify(body),
    });
}
