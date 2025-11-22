import { apiGet, apiPost } from "./client";

/**
 * Inference mode summary
 */
export interface InferenceModeSummary {
  id: string;
  description: string | null;
  supports_batch: boolean;
  supports_background_removal: boolean;
  notes: string[];
}

/**
 * Checkpoint summary for UI selection
 */
export interface CheckpointSummary {
  display_name: string;
  checkpoint_path: string;
  modified_at: string; // ISO datetime string
  size_mb: number;
  exp_name: string | null;
  architecture: string | null;
  backbone: string | null;
}

/**
 * Checkpoint with parsed metadata
 */
export interface CheckpointWithMetadata extends CheckpointSummary {
  modifiedDate: Date;
}

/**
 * List available inference modes
 */
export async function listInferenceModes(): Promise<InferenceModeSummary[]> {
  return apiGet<InferenceModeSummary[]>("/inference/modes");
}

/**
 * List available checkpoints with optional limit
 */
export async function listCheckpoints(
  limit = 50,
): Promise<CheckpointWithMetadata[]> {
  const checkpoints = await apiGet<CheckpointSummary[]>(
    `/inference/checkpoints?limit=${limit}`,
  );

  // Parse datetime strings
  return checkpoints.map((ckpt) => ({
    ...ckpt,
    modifiedDate: new Date(ckpt.modified_at),
  }));
}

/**
 * Filter checkpoints by search term
 */
export function filterCheckpoints(
  checkpoints: CheckpointWithMetadata[],
  searchTerm: string,
): CheckpointWithMetadata[] {
  if (!searchTerm.trim()) {
    return checkpoints;
  }

  const term = searchTerm.toLowerCase();
  return checkpoints.filter(
    (ckpt) =>
      ckpt.display_name.toLowerCase().includes(term) ||
      ckpt.exp_name?.toLowerCase().includes(term) ||
      ckpt.architecture?.toLowerCase().includes(term) ||
      ckpt.backbone?.toLowerCase().includes(term),
  );
}

/**
 * Filter checkpoints by architecture
 */
export function filterByArchitecture(
  checkpoints: CheckpointWithMetadata[],
  architecture: string | null,
): CheckpointWithMetadata[] {
  if (!architecture) {
    return checkpoints;
  }
  return checkpoints.filter((ckpt) => ckpt.architecture === architecture);
}

/**
 * Filter checkpoints by backbone
 */
export function filterByBackbone(
  checkpoints: CheckpointWithMetadata[],
  backbone: string | null,
): CheckpointWithMetadata[] {
  if (!backbone) {
    return checkpoints;
  }
  return checkpoints.filter((ckpt) => ckpt.backbone === backbone);
}

/**
 * Get unique architectures from checkpoints
 */
export function getUniqueArchitectures(
  checkpoints: CheckpointWithMetadata[],
): string[] {
  const architectures = new Set<string>();
  checkpoints.forEach((ckpt) => {
    if (ckpt.architecture) {
      architectures.add(ckpt.architecture);
    }
  });
  return Array.from(architectures).sort();
}

/**
 * Get unique backbones from checkpoints
 */
export function getUniqueBackbones(
  checkpoints: CheckpointWithMetadata[],
): string[] {
  const backbones = new Set<string>();
  checkpoints.forEach((ckpt) => {
    if (ckpt.backbone) {
      backbones.add(ckpt.backbone);
    }
  });
  return Array.from(backbones).sort();
}

/**
 * Text region detected by inference
 */
export interface TextRegion {
  polygon: number[][];  // [[x1, y1], [x2, y2], ...]
  text: string | null;
  confidence: number;
}

/**
 * Inference preview request
 */
export interface InferencePreviewRequest {
  checkpoint_path: string;
  image_base64?: string | null;
  image_path?: string | null;
  confidence_threshold?: number;
  nms_threshold?: number;
}

/**
 * Inference preview response
 */
export interface InferencePreviewResponse {
  status: string;
  regions: TextRegion[];
  processing_time_ms: number;
  notes: string[];
}

/**
 * Run inference preview on single image
 */
export async function runInferencePreview(
  request: InferencePreviewRequest,
): Promise<InferencePreviewResponse> {
  return apiPost<InferencePreviewRequest, InferencePreviewResponse>(
    "/inference/preview",
    request,
  );
}
