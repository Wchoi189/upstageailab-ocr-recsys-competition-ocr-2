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

export const ocrClient = {
    // Health check can be simple ping or re-implement if shared has one
    healthCheck: async (): Promise<boolean> => {
        try {
            // Shared package doesn't have explicit health check exported yet, so we can check checkpoints
            await sharedListCheckpoints(1);
            return true;
        } catch {
            return false;
        }
    },

    listCheckpoints: async (): Promise<Checkpoint[]> => {
        // #region agent log
        const startTime = Date.now();
        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:41',message:'listCheckpoints called - using direct fetch workaround',data:{timestamp:Date.now()},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
        // #endregion
        try {
            // WORKAROUND: Direct fetch since shared package fetch is hanging
            // This bypasses the shared package API client which seems to have issues
            // Try 127.0.0.1 instead of localhost to avoid DNS resolution issues
            const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8002/api';
            const url = `${apiUrl}/inference/checkpoints?limit=100`;

            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:47',message:'Direct fetch to API',data:{url,apiUrl},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
            // #endregion

            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:57',message:'Fetch response received',data:{status:response.status,statusText:response.statusText,ok:response.ok},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
            // #endregion

            if (!response.ok) {
                throw new Error(`API request failed: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            // #region agent log
            const duration = Date.now() - startTime;
            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:66',message:'Checkpoints data parsed',data:{dataLength:data.length,durationMs:duration,firstItem:data[0]||null},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
            // #endregion

            const mapped = data.map((ckpt: any) => ({
                path: ckpt.checkpoint_path,
                name: ckpt.display_name,
                size_mb: ckpt.size_mb,
                modified: new Date(ckpt.modified_at).getTime()
            }));
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:75',message:'Mapped checkpoints successfully',data:{mappedLength:mapped.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
            // #endregion
            return mapped;
        } catch (e: any) {
            // #region agent log
            const duration = Date.now() - startTime;
            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:79',message:'listCheckpoints error caught',data:{errorMessage:e?.message,errorName:e?.name,errorStack:e?.stack,durationMs:duration},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
            // #endregion
            console.error('Failed to list checkpoints', e);
            return [];
        }
    },

    predict: async (
        file: File,
        checkpointPath?: string,
        enablePerspectiveCorrection?: boolean,
        perspectiveDisplayMode?: string
    ): Promise<InferenceResponse> => {
        // #region agent log
        const predictStartTime = Date.now();
        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:98',message:'predict called',data:{fileName:file.name,fileSize:file.size,hasCheckpointPath:!!checkpointPath,checkpointPath:checkpointPath||'null'},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'G'})}).catch(()=>{});
        // #endregion

        // Convert file to base64
        const toBase64 = (file: File) => new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = error => reject(error);
        });

        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:107',message:'Converting file to base64',data:{timestamp:Date.now()},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'G'})}).catch(()=>{});
        // #endregion
        const base64Image = await toBase64(file);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:110',message:'Base64 conversion complete',data:{base64Length:base64Image.length,hasDataPrefix:base64Image.startsWith('data:')},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'G'})}).catch(()=>{});
        // #endregion

        // WORKAROUND: Direct fetch since shared package fetch is hanging
        // This bypasses the shared package API client which has issues
        const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8002/api';
        const url = `${apiUrl}/inference/preview`;

        // #region agent log
        const apiCallStartTime = Date.now();
        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:120',message:'Calling inference API directly',data:{url,checkpointPath:checkpointPath||'empty',hasCheckpoint:!!checkpointPath,imageBase64Length:base64Image.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'G'})}).catch(()=>{});
        // #endregion

        const requestBody = {
            checkpoint_path: checkpointPath || "",
            image_base64: base64Image,
            confidence_threshold: 0.1,
            nms_threshold: 0.4,
            enable_perspective_correction: enablePerspectiveCorrection || false,
            perspective_display_mode: perspectiveDisplayMode || "corrected"
        };

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        });

        // #region agent log
        const fetchDuration = Date.now() - apiCallStartTime;
        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:135',message:'Inference API response received',data:{status:response.status,statusText:response.statusText,ok:response.ok,durationMs:fetchDuration},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'G'})}).catch(()=>{});
        // #endregion

        if (!response.ok) {
            const errorText = await response.text();
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:142',message:'Inference API error response',data:{status:response.status,statusText:response.statusText,errorText},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'G'})}).catch(()=>{});
            // #endregion
            throw new Error(`Inference API failed: ${response.status} ${response.statusText}: ${errorText}`);
        }

        const result = await response.json();

        // #region agent log
        const apiCallDuration = Date.now() - apiCallStartTime;
        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:149',message:'Inference result parsed',data:{status:result.status,regionsCount:result.regions?.length||0,durationMs:apiCallDuration,hasMeta:!!result.meta},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'G'})}).catch(()=>{});
        // #endregion

        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:71',message:'API response received - raw inference result',data:{status:result.status,regionsCount:result.regions.length,hasMeta:!!result.meta,meta:result.meta,firstRegion:result.regions[0]?{polygon:result.regions[0].polygon,confidence:result.regions[0].confidence,text:result.regions[0].text}:null,hasPreviewImage:!!result.preview_image_base64},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
        // #endregion

        if (result.status !== 'success') {
            throw new Error('Inference failed');
        }

        // Map to legacy format
        const predictions: Prediction[] = result.regions.map(r => ({
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

        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ocrClient.ts:165',message:'Mapped predictions - coordinate system check',data:{predictionsCount:predictions.length,firstPredictionPoints:predictions[0]?.points,metaCoordinateSystem:result.meta?.coordinate_system,metaProcessedSize:result.meta?.processed_size,metaOriginalSize:result.meta?.original_size,metaPadding:result.meta?.padding,metaScale:result.meta?.scale},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
        // #endregion

        return {
            filename: file.name,
            predictions: predictions,
            meta: meta,
            preview_image_base64: result.preview_image_base64 || null
        };
    }
};
