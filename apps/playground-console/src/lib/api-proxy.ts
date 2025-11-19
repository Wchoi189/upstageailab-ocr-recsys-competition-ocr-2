import { NextResponse } from "next/server";

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";

/**
 * Proxy a request to the FastAPI backend
 *
 * @param path - The API path (e.g., "/commands/schemas")
 * @param options - Fetch options (method, body, headers, etc.)
 * @returns NextResponse with the FastAPI response or error
 */
export async function proxyToFastAPI(
  path: string,
  options?: RequestInit
): Promise<NextResponse> {
  const url = `${FASTAPI_BASE_URL}${path}`;

  try {
    const response = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        { error: errorText || "Request failed" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error(`Proxy error for ${path}:`, error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Unknown proxy error",
        details: "Failed to connect to backend API"
      },
      { status: 500 }
    );
  }
}
