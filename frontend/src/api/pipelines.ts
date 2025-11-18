import { apiClient } from "./client";

/**
 * Pipeline preview request
 */
export interface PipelinePreviewRequest {
  pipeline_id: string;
  checkpoint_path?: string | null;
  image_base64?: string | null;
  image_path?: string | null;
  params: Record<string, unknown>;
}

/**
 * Pipeline preview response
 */
export interface PipelinePreviewResponse {
  status: string;
  job_id: string;
  routed_backend: string;
  cache_key: string;
  notes: string[];
}

/**
 * Pipeline fallback request
 */
export interface PipelineFallbackRequest {
  pipeline_id: string;
  image_path: string;
  params: Record<string, unknown>;
}

/**
 * Pipeline fallback response
 */
export interface PipelineFallbackResponse {
  status: string;
  routed_backend: string;
  result_path: string | null;
  notes: string[];
}

/**
 * Queue preview task (client-side execution)
 */
export async function queuePreview(
  request: PipelinePreviewRequest,
): Promise<PipelinePreviewResponse> {
  return apiClient<PipelinePreviewResponse>("/api/pipelines/preview", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

/**
 * Queue fallback task (backend execution)
 */
export async function queueFallback(
  request: PipelineFallbackRequest,
): Promise<PipelineFallbackResponse> {
  return apiClient<PipelineFallbackResponse>("/api/pipelines/fallback", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

/**
 * Generate cache key for image and parameters
 *
 * Uses SHA-1 hash of image bytes + params (matches backend logic)
 */
export async function generateCacheKey(
  pipelineId: string,
  params: Record<string, unknown>,
  imageBytes: ArrayBuffer,
): Promise<string> {
  const encoder = new TextEncoder();
  const paramsJson = JSON.stringify(params, Object.keys(params).sort());
  const paramsBytes = encoder.encode(paramsJson);

  // Concatenate image bytes and params bytes
  const combined = new Uint8Array(imageBytes.byteLength + paramsBytes.length);
  combined.set(new Uint8Array(imageBytes), 0);
  combined.set(paramsBytes, imageBytes.byteLength);

  // Compute SHA-1 hash
  const hashBuffer = await crypto.subtle.digest("SHA-1", combined);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");

  return `${pipelineId}:${hashHex.slice(0, 12)}`;
}
