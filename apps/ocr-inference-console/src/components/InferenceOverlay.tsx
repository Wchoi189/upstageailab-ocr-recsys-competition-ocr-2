import React from 'react';
import { type Prediction } from '../services/api';

interface InferenceOverlayProps {
    imageUrl: string;
    predictions: Prediction[];
    width?: number; // Natural width of image
    height?: number; // Natural height of image
}

export const InferenceOverlay: React.FC<InferenceOverlayProps> = ({ imageUrl, predictions }) => {
    // We render the image and an SVG overlay on top.
    // SVG viewBox should match the image's coordinate system (natural size).
    // But we don't know natural size until image loads, unless we get it from props or state.

    // Simple approach: Use a container, ref to get image dimensions, or use percentage if points were normalized (they are not, they are pixel absolute).

    // We will use an onload handler on the image to set the viewBox.
    const [dimensions, setDimensions] = React.useState<{ w: number, h: number } | null>(null);

    const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
        const img = e.currentTarget;
        setDimensions({ w: img.naturalWidth, h: img.naturalHeight });
    };

    return (
        <div className="relative inline-block w-full h-full">
            <img
                src={imageUrl}
                alt="Source"
                className="w-full h-full object-contain select-none"
                onLoad={handleImageLoad}
            />
            {dimensions && (
                <svg
                    className="absolute top-0 left-0 w-full h-full pointer-events-none"
                    viewBox={`0 0 ${dimensions.w} ${dimensions.h}`}
                    preserveAspectRatio="xMidYMid meet" // Must match object-contain behavior roughly, usually tricky with simple absolute
                // Actually, if img is object-contain, it might have empty space.
                // The SVG effectively covers the img element box, not the image content if there's letterboxing.
                // To solve this perfectly, we need to know the rendered position of the image content.

                // Simplify: We assume the parent container constrains the image, and we just want to draw on the image.
                // If we position SVG absolute over img, we rely on them matching.
                // A better way for object-contain is to use a wrapper div that fits the image aspect ratio, but UI layout might be fixed.

                // For now, let's assume we can overlay safely or use a simpler transform approach?
                // Let's stick to standard SVG overlay and hope 'object-contain' doesn't offset too much,
                // OR we force the image to be 'object-top-left' or similar?
                // Wait, object-contain centers the image.
                // If we use object-contain, we simply cannot map specific pixels 1:1 easily without JS calc.

                // Alternative: Don't use object-contain on the img tag directly if we want simple overlay.
                // Use a container that has the aspect ratio, or render the image inside SVG?
                // Rendering image inside SVG is robust for coordinates.
                >
                    {/* Re-render image in SVG? No, too slow maybe. */}
                </svg>
            )}

            {/*
               Alternative Robust Approach:
               Use SVG to render BOTH the image and the polygons.
               Then responsive scaling is automatic via viewBox.
            */}

            {dimensions && (
                <svg
                    viewBox={`0 0 ${dimensions.w} ${dimensions.h}`}
                    className="absolute inset-0 w-full h-full"
                    style={{
                        // This assumes the PARENT container enforces the aspect ratio or we want to fill?
                        // If we want 'object-contain' behavior, we can let SVG handle it.
                    }}
                >
                    {/* We don't render the image here to avoid double load, but we rely on exact alignment.
                         Actually, standard practice for canvas/svg annotation is to draw image on canvas or use relative coordinates.

                         Let's try the SVG-only approach for the content viewing area if possible?
                         Or just standard img + absolute svg and warn about aspect ratio.

                         Let's try: render image as <image> inside SVG. It guarantees alignment.
                     */}
                    <image href={imageUrl} width={dimensions.w} height={dimensions.h} />

                    {predictions.map((pred, i) => {
                        const pts = pred.points.map(p => p.join(',')).join(' ');
                        return (
                            <polygon
                                key={i}
                                points={pts}
                                fill="rgba(0, 255, 0, 0.2)"
                                stroke="#00FF00"
                                strokeWidth={2}
                            />
                        );
                    })}
                </svg>
            )}
            {/* Note: In the block above I rendered <img> AND <svg><image/></svg> ... that's redundant.
                 Let's switch to ONLY SVG for the viewer if we have an image.
             */}
        </div>
    );
};

// Refined Component
export const InferenceViewer: React.FC<{ imageUrl: string, predictions: Prediction[] }> = ({ imageUrl, predictions }) => {
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
}
