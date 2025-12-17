import { apiGet, apiPost } from "./client.js";

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
 *
 * Note: First load may be slow if checkpoint metadata files are missing.
 * Consider running 'make checkpoint-metadata' to pre-generate metadata.
 */
export async function listCheckpoints(
    limit = 50,
): Promise<CheckpointWithMetadata[]> {    try {
        // Use longer timeout for checkpoint loading (30 seconds)
        // as it can be slow on first load if metadata files are missing
        const checkpoints = await apiGet<CheckpointSummary[]>(
            `/inference/checkpoints?limit=${limit}`,
            { maxRetries: 3, retryDelay: 2000 } // More retries with longer delay
        );        // Parse datetime strings
        return checkpoints.map((ckpt) => ({
            ...ckpt,
            modifiedDate: new Date(ckpt.modified_at),
        }));
    } catch (error: any) {        throw error;
    }
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
 * Padding information for preprocessed images
 */
export interface Padding {
    top: number;
    bottom: number;
    left: number;
    right: number;
    // Bug fix: include original size if needed by frontend
}

/**
 * Metadata about image preprocessing and coordinate system
 */
export interface InferenceMetadata {
    original_size: [number, number];  // [width, height]
    processed_size: [number, number];  // [width, height]
    padding: Padding;
    scale: number;
    coordinate_system: "pixel" | "normalized";
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
    // BUG-001: optional base64-encoded PNG of the preprocessed image used for
    // polygon decoding so that the frontend can render coordinate-aligned overlays.
    preview_image_base64?: string | null;
    // Data contract: metadata about preprocessing and coordinate system
    meta?: InferenceMetadata | null;
}

/**
 * Run inference preview on single image
 */
export async function runInferencePreview(
    request: InferencePreviewRequest,
): Promise<InferencePreviewResponse> {
    const response = await apiPost<InferencePreviewRequest, InferencePreviewResponse>(
        "/inference/preview",
        request,
    );

    return response;
}
