import { apiGet, apiPost } from "./client";

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
  return apiGet<ComparisonPreset[]>("/evaluation/presets");
}

/**
 * Queue comparison job
 */
export async function queueComparison(
  request: ComparisonRequest,
): Promise<ComparisonResponse> {
  return apiPost<ComparisonRequest, ComparisonResponse>(
    "/evaluation/compare",
    request,
  );
}

/**
 * Get gallery root directory
 */
export async function getGalleryRoot(): Promise<GalleryRootResponse> {
  return apiGet<GalleryRootResponse>("/evaluation/gallery-root");
}
