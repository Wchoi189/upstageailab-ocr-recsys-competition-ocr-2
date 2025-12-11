export interface Point {
    x: number;
    y: number;
}

export interface Prediction {
    points: number[][]; // [x, y]
    confidence: number;
}

export interface InferenceResponse {
    filename: string;
    predictions: Prediction[];
}

const API_BASE_URL = 'http://localhost:8000/ocr';

export const ocrClient = {
    healthCheck: async (): Promise<boolean> => {
        try {
            const res = await fetch(`${API_BASE_URL}/health`);
            if (!res.ok) return false;
            const data = await res.json();
            return data.status === 'ok';
        } catch (e) {
            console.error('Health check failed', e);
            return false;
        }
    },

    predict: async (file: File): Promise<InferenceResponse> => {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Inference failed');
        }

        return res.json();
    }
};
