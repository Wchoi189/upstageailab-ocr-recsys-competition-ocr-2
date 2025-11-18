import { apiClient } from "./client";

/**
 * Comparison preset descriptor
 */
export interface ComparisonPreset {
  id: string;
  label: string;
  required_inputs: string[];
  description: string | null;
}

/**
 * Comparison request
 */
export interface ComparisonRequest {
  preset_id: string;
  model_a_path?: string | null;
  model_b_path?: string | null;
  ground_truth_path?: string | null;
  image_dir?: string | null;
  extra_params?: Record<string, unknown>;
}

/**
 * Comparison response
 */
export interface ComparisonResponse {
  status: string;
  message: string;
  next_steps: string[];
}

/**
 * Gallery root response
 */
export interface GalleryRootResponse {
  gallery_root: string;
}

/**
 * List available comparison presets
 */
export async function listComparisonPresets(): Promise<ComparisonPreset[]> {
  return apiClient<ComparisonPreset[]>("/api/evaluation/presets");
}

/**
 * Queue comparison job
 */
export async function queueComparison(
  request: ComparisonRequest,
): Promise<ComparisonResponse> {
  return apiClient<ComparisonResponse>("/api/evaluation/compare", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

/**
 * Get gallery root directory
 */
export async function getGalleryRoot(): Promise<GalleryRootResponse> {
  return apiClient<GalleryRootResponse>("/api/evaluation/gallery-root");
}
