// Removed shared package imports - using direct fetch workaround due to hanging issues
// import {
//     runInferencePreview,
//     listCheckpoints as sharedListCheckpoints
// } from '@upstage/console-shared';

export interface Point {
    x: number;
    y: number;
}

export interface Prediction {
    points: number[][]; // [x, y]
    confidence: number;
    label?: string; // Added label support
}

export interface PredictionMetadata {
    coordinate_system?: "pixel" | "normalized";
    processed_size?: [number, number]; // [width, height]
    original_size?: [number, number]; // [width, height]
    scale?: number;
    padding?: {
        top: number;
        bottom: number;
        left: number;
        right: number;
    };
}

export interface InferenceResponse {
    filename: string;
    predictions: Prediction[];
    meta?: PredictionMetadata; // Include metadata for coordinate transformation
    preview_image_base64?: string | null; // Preview image from backend (matches coordinate system)
}

export interface Checkpoint {
    path: string;
    name: string;
    size_mb: number;
    modified: number;
}

interface ApiRegion {
    polygon: number[][];
    confidence: number;
    text?: string;
}

export const ocrClient = {
    // Health check using backend /health endpoint
    healthCheck: async (): Promise<boolean> => {
        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8002/api';
            const response = await fetch(`${apiUrl}/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
            });
            return response.ok;
        } catch {
            return false;
        }
    },

    listCheckpoints: async (): Promise<Checkpoint[]> => {
        try {
            // WORKAROUND: Direct fetch since shared package fetch is hanging
            // This bypasses the shared package API client which seems to have issues
            // Try 127.0.0.1 instead of localhost to avoid DNS resolution issues
            const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8002/api';
            const url = `${apiUrl}/inference/checkpoints?limit=100`;

            // Log locally if needed; removed external ingest calls

            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            // Optional: console.debug('Checkpoints response', response.status)

            if (!response.ok) {
                throw new Error(`API request failed: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();

            const mapped = data.map((ckpt: any) => ({
                path: ckpt.checkpoint_path,
                name: ckpt.display_name,
                size_mb: ckpt.size_mb,
                modified: new Date(ckpt.modified_at).getTime()
            }));
            return mapped;
        } catch (e: any) {
            console.error('Failed to list checkpoints', e);
            return [];
        }
    },

    predict: async (
        file: File,
        checkpointPath?: string,
        enablePerspectiveCorrection?: boolean,
        perspectiveDisplayMode?: string,
        enableGrayscale?: boolean
    ): Promise<InferenceResponse> => {
        // Convert file to base64
        const toBase64 = (file: File) => new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = error => reject(error);
        });

        // Converting file to base64
        const base64Image = await toBase64(file);
        // Base64 conversion complete

        // WORKAROUND: Direct fetch since shared package fetch is hanging
        // This bypasses the shared package API client which has issues
        const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8002/api';
        const url = `${apiUrl}/inference/preview`;

        const requestBody = {
            checkpoint_path: checkpointPath || "",
            image_base64: base64Image,
            confidence_threshold: 0.1,
            nms_threshold: 0.4,
            enable_perspective_correction: enablePerspectiveCorrection || false,
            perspective_display_mode: perspectiveDisplayMode || "corrected",
            enable_grayscale: enableGrayscale || false
        };

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            const errorText = await response.text();
            // Optional: console.error('Inference error', response.status)
            throw new Error(`Inference API failed: ${response.status} ${response.statusText}: ${errorText}`);
        }

        const result = await response.json();

        // Optional: console.debug('Inference parsed', result.status)

        if (result.status !== 'success') {
            throw new Error('Inference failed');
        }

        // Map to legacy format
        const predictions: Prediction[] = result.regions.map((r: ApiRegion) => ({
            points: r.polygon,
            confidence: r.confidence,
            label: r.text || undefined
        }));

        // Extract metadata for coordinate transformation
        const meta: PredictionMetadata | undefined = result.meta ? {
            coordinate_system: result.meta.coordinate_system,
            processed_size: result.meta.processed_size,
            original_size: result.meta.original_size,
            scale: result.meta.scale,
            padding: result.meta.padding,
        } : undefined;

        // Optional: console.debug('Mapped predictions', predictions.length)

        return {
            filename: file.name,
            predictions: predictions,
            meta: meta,
            preview_image_base64: result.preview_image_base64 || null
        };
    }
};
