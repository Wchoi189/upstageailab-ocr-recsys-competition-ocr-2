import React from 'react';
import { type Prediction } from '../api/ocrClient';

export const PolygonOverlay: React.FC<{ imageUrl: string, predictions: Prediction[] }> = ({ imageUrl, predictions }) => {
    const [dimensions, setDimensions] = React.useState<{ w: number, h: number } | null>(null);

    // Creates a hidden image to load dimensions
    React.useEffect(() => {
        const img = new Image();
        img.onload = () => setDimensions({ w: img.naturalWidth, h: img.naturalHeight });
        img.src = imageUrl;
    }, [imageUrl]);

    if (!dimensions) {
        return <div className="flex items-center justify-center h-full text-gray-400">Loading image...</div>;
    }

    return (
        <div className="w-full h-full bg-gray-100 flex items-center justify-center p-4">
            {/* SVG container that preserves aspect ratio */}
            <svg
                viewBox={`0 0 ${dimensions.w} ${dimensions.h}`}
                className="max-w-full max-h-full shadow-lg bg-white"
            >
                <image href={imageUrl} width={dimensions.w} height={dimensions.h} />
                {predictions.map((pred, i) => (
                    <polygon
                        key={i}
                        points={pred.points.map(p => p.join(',')).join(' ')}
                        fill="rgba(59, 130, 246, 0.2)"
                        stroke="#2563EB"
                        strokeWidth={Math.max(2, dimensions.w / 500)} // Scale stroke slightly
                        vectorEffect="non-scaling-stroke" // Keep stroke consistent if we scaled via CSS (but we scale via viewBox)
                    />
                ))}
            </svg>
        </div>
    );
};
