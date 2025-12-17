import React from 'react';
import { type Prediction, type PredictionMetadata } from '../api/ocrClient';

export const PolygonOverlay: React.FC<{
    imageUrl: string,
    predictions: Prediction[],
    meta?: PredictionMetadata,
    previewImageBase64?: string | null
}> = ({ imageUrl, predictions, meta, previewImageBase64 }) => {    const [dimensions, setDimensions] = React.useState<{ w: number, h: number } | null>(null);
    const [displayImageUrl, setDisplayImageUrl] = React.useState<string>(imageUrl);
    const [contentArea, setContentArea] = React.useState<{ x: number, y: number, w: number, h: number } | null>(null);

    // BUG-001: Use preview_image_base64 if available (matches coordinate system)
    // This follows the same pattern as playground-console InferencePreviewCanvas
    // Trim black padding using content_area calculated from padding metadata
    React.useEffect(() => {
        if (previewImageBase64 && meta?.processed_size) {            try {
                const binary = atob(previewImageBase64);
                const len = binary.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) {
                    bytes[i] = binary.charCodeAt(i);
                }
                const blob = new Blob([bytes], { type: "image/jpeg" });
                const objectUrl = URL.createObjectURL(blob);

                // Set state synchronously (hooks must be called consistently)
                setDisplayImageUrl(objectUrl);

                // Calculate content area from padding (top-left padding: content at 0,0)
                const [processedW, processedH] = meta.processed_size;
                const padding = meta.padding || { top: 0, bottom: 0, left: 0, right: 0 };
                // For top-left padding: content starts at (0, 0) and ends before padding
                // Content area: (0, 0, processedW - padding.right, processedH - padding.bottom)
                const contentW = processedW - padding.right;
                const contentH = processedH - padding.bottom;
                const calculatedContentArea = { x: 0, y: 0, w: contentW, h: contentH };
                setContentArea(calculatedContentArea);                // Set display dimensions to content area (trimmed padding)
                setDimensions({ w: contentW, h: contentH });

                // Load preview image to verify dimensions (async, for logging only - doesn't affect hooks)
                const img = new Image();
                img.onload = () => {                };
                img.src = objectUrl;

                return () => {
                    URL.revokeObjectURL(objectUrl);
                };
            } catch (e) {                setDisplayImageUrl(imageUrl);
                setContentArea(null);
            }
        } else {
            setDisplayImageUrl(imageUrl);
            setContentArea(null);
        }
    }, [previewImageBase64, meta?.processed_size, meta?.padding, imageUrl]);

    // Load image dimensions (for original image fallback)
    React.useEffect(() => {
        if (!dimensions && displayImageUrl === imageUrl) {
            const img = new Image();
            img.onload = () => {                setDimensions({ w: img.naturalWidth, h: img.naturalHeight });
            };
            img.src = displayImageUrl;
        }
    }, [displayImageUrl, imageUrl, dimensions]);    // Early return AFTER all hooks are called (Rules of Hooks requirement)
    if (!dimensions) {
        return <div className="flex items-center justify-center h-full text-gray-400">Loading image...</div>;
    }    return (
        <div className="w-full h-full bg-gray-100 flex items-center justify-center p-4">
            {/* SVG container that preserves aspect ratio */}
            <svg
                viewBox={contentArea ? `${contentArea.x} ${contentArea.y} ${contentArea.w} ${contentArea.h}` : `0 0 ${dimensions.w} ${dimensions.h}`}
                className="max-w-full max-h-full shadow-lg bg-white"
                style={{ position: 'relative', zIndex: 1 }}
                preserveAspectRatio="xMidYMid meet"
            >
                {/* Display full preview image, but viewBox crops to content area (trimming black padding) */}
                {/* For top-left padding: content starts at (0,0), so viewBox shows content area */}
                {/* The image element shows the full processed_size image, viewBox crops to content */}
                {meta?.processed_size ? (
                    <image
                        href={displayImageUrl}
                        width={meta.processed_size[0]}
                        height={meta.processed_size[1]}
                        x="0"
                        y="0"
                        preserveAspectRatio="none"
                    />
                ) : (
                    <image href={displayImageUrl} width={dimensions.w} height={dimensions.h} x="0" y="0" />
                )}                {predictions.map((pred, i) => {
                    // Transform coordinates from processed_size to original_size if metadata available
                    let transformedPoints = pred.points;                    // BUG-001: Follow playground-console pattern - if preview_image_base64 is used,
                    // coordinates are already in processed_size space. If content area is trimmed,
                    // coordinates need to be offset by content area origin (for top-left: 0,0, so no offset needed).
                    // Only apply transformation if using original image (fallback case).
                    if (previewImageBase64 && meta?.processed_size) {
                        // Using preview image - coordinates are in processed_size space
                        // If we trimmed padding, coordinates are already correct (top-left padding means content at 0,0)
                        // No transformation needed, just use coordinates directly
                        transformedPoints = pred.points;                    } else if (meta && meta.processed_size) {
                        // Fallback: using original image, need to transform coordinates
                        console.log('Transforming coordinates (fallback - no preview image):', { processedSize: meta.processed_size, actualImageSize: [dimensions.w, dimensions.h], originalPoint: pred.points[0] });
                        const [processedW, processedH] = meta.processed_size;
                        // Use actual image dimensions for transformation target
                        const targetW = dimensions.w;
                        const targetH = dimensions.h;
                        const scaleX = targetW / processedW;
                        const scaleY = targetH / processedH;

                        transformedPoints = pred.points.map(([x, y]) => [
                            x * scaleX,
                            y * scaleY
                        ]);                    } else {                    }                    return (
                        <polygon
                            key={i}
                            points={transformedPoints.map(p => p.join(',')).join(' ')}
                            fill="rgba(255, 99, 71, 0.25)" // Darker red (tomato) fill with slightly higher opacity
                            stroke="rgba(220, 20, 60, 0.9)" // Darker red (crimson) stroke
                            strokeWidth="0.5" // Very thin border
                            vectorEffect="non-scaling-stroke" // Keep stroke consistent if we scaled via CSS (but we scale via viewBox)
                            style={{ pointerEvents: 'none' }} // Allow clicks to pass through
                        />
                    );
                })}
            </svg>
        </div>
    );
};
