import React from 'react';
import { type Prediction, type PredictionMetadata } from '../api/ocrClient';

export const PolygonOverlay: React.FC<{
    imageUrl: string,
    predictions: Prediction[],
    meta?: PredictionMetadata,
    previewImageBase64?: string | null
}> = ({ imageUrl, predictions, meta, previewImageBase64 }) => {
    // #region agent log
    React.useEffect(() => {
        console.log('PolygonOverlay meta:', meta);
        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:6',message:'PolygonOverlay received props',data:{hasMeta:!!meta,hasProcessedSize:!!meta?.processed_size,hasOriginalSize:!!meta?.original_size,meta:meta,predictionsCount:predictions.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
    }, [meta, predictions]);
    // #endregion

    const [dimensions, setDimensions] = React.useState<{ w: number, h: number } | null>(null);
    const [displayImageUrl, setDisplayImageUrl] = React.useState<string>(imageUrl);
    const [contentArea, setContentArea] = React.useState<{ x: number, y: number, w: number, h: number } | null>(null);

    // BUG-001: Use preview_image_base64 if available (matches coordinate system)
    // This follows the same pattern as playground-console InferencePreviewCanvas
    // Trim black padding using content_area calculated from padding metadata
    React.useEffect(() => {
        if (previewImageBase64 && meta?.processed_size) {
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:18',message:'Loading preview image from base64',data:{hasPreview:!!previewImageBase64,previewLength:previewImageBase64.length,processedSize:meta.processed_size,padding:meta.padding},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
            // #endregion
            try {
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
                setContentArea(calculatedContentArea);

                // #region agent log
                fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:45',message:'Content area calculation for padding trim',data:{processedSize:[processedW,processedH],padding,contentArea:calculatedContentArea,hasPadding:padding.right>0||padding.bottom>0},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
                // #endregion

                // Set display dimensions to content area (trimmed padding)
                setDimensions({ w: contentW, h: contentH });

                // Load preview image to verify dimensions (async, for logging only - doesn't affect hooks)
                const img = new Image();
                img.onload = () => {
                    // #region agent log
                    fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:55',message:'Preview image loaded - actual dimensions',data:{naturalWidth:img.naturalWidth,naturalHeight:img.naturalHeight,processedSize:meta.processed_size,expectedSize:meta.processed_size},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
                    // #endregion
                };
                img.src = objectUrl;

                return () => {
                    URL.revokeObjectURL(objectUrl);
                };
            } catch (e) {
                // #region agent log
                fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:35',message:'Failed to load preview image, using original',data:{error:String(e)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
                // #endregion
                setDisplayImageUrl(imageUrl);
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
            img.onload = () => {
                // #region agent log
                fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:50',message:'Image loaded - dimensions',data:{naturalWidth:img.naturalWidth,naturalHeight:img.naturalHeight,imageUrl:displayImageUrl},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
                // #endregion
                setDimensions({ w: img.naturalWidth, h: img.naturalHeight });
            };
            img.src = displayImageUrl;
        }
    }, [displayImageUrl, imageUrl, dimensions]);

    // #region agent log
    React.useEffect(() => {
        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:22',message:'PolygonOverlay render - predictions data',data:{predictionsCount:predictions.length,hasDimensions:!!dimensions,imageDimensions:dimensions,firstPrediction:predictions[0]?{pointsCount:predictions[0].points.length,firstPoint:predictions[0].points[0],confidence:predictions[0].confidence}:null},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
    }, [predictions, dimensions]);
    // #endregion

    // #region agent log
    React.useEffect(() => {
        if (contentArea && meta?.processed_size) {
            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:108',message:'SVG viewBox and content area calculation',data:{contentArea,processedSize:meta.processed_size,padding:meta.padding,viewBox:`${contentArea.x} ${contentArea.y} ${contentArea.w} ${contentArea.h}`,dimensions},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
        }
    }, [contentArea, meta?.processed_size, meta?.padding, dimensions]);
    // #endregion

    // Early return AFTER all hooks are called (Rules of Hooks requirement)
    if (!dimensions) {
        return <div className="flex items-center justify-center h-full text-gray-400">Loading image...</div>;
    }

    // #region agent log
    const logPolygonRendering = () => {
        if (predictions.length > 0) {
            const firstPolygon = predictions[0];
            const viewBoxW = dimensions.w;
            const viewBoxH = dimensions.h;
            const firstPoint = firstPolygon.points[0];
            const maxX = Math.max(...firstPolygon.points.map(p => p[0]));
            const maxY = Math.max(...firstPolygon.points.map(p => p[1]));
            const minX = Math.min(...firstPolygon.points.map(p => p[0]));
            const minY = Math.min(...firstPolygon.points.map(p => p[1]));
            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:35',message:'Polygon coordinate analysis - coordinate mismatch check',data:{viewBoxWidth:viewBoxW,viewBoxHeight:viewBoxH,firstPointX:firstPoint[0],firstPointY:firstPoint[1],polygonMinX:minX,polygonMaxX:maxX,polygonMinY:minY,polygonMaxY:maxY,isOutsideViewBox:maxX>viewBoxW||maxY>viewBoxH||minX<0||minY<0},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
        }
    };
    logPolygonRendering();
    // #endregion

    return (
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
                )}
                {/* #region agent log */}
                {(() => {
                    if (predictions.length > 0) {
                        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:69',message:'Rendering polygons in SVG',data:{predictionsCount:predictions.length,hasMeta:!!meta,viewBox:`0 0 ${dimensions.w} ${dimensions.h}`},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
                    }
                    return null;
                })()}
                {/* #endregion */}
                {predictions.map((pred, i) => {
                    // Transform coordinates from processed_size to original_size if metadata available
                    let transformedPoints = pred.points;

                    // #region agent log
                    if (i === 0) {
                        console.log('Transformation check:', { hasMeta: !!meta, processedSize: meta?.processed_size, originalSize: meta?.original_size, firstPoint: pred.points[0] });
                        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:63',message:'Checking transformation condition',data:{hasMeta:!!meta,hasProcessedSize:!!meta?.processed_size,hasOriginalSize:!!meta?.original_size,processedSize:meta?.processed_size,originalSize:meta?.original_size,firstPoint:pred.points[0]},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
                    }
                    // #endregion

                    // BUG-001: Follow playground-console pattern - if preview_image_base64 is used,
                    // coordinates are already in processed_size space. If content area is trimmed,
                    // coordinates need to be offset by content area origin (for top-left: 0,0, so no offset needed).
                    // Only apply transformation if using original image (fallback case).
                    if (previewImageBase64 && meta?.processed_size) {
                        // Using preview image - coordinates are in processed_size space
                        // If we trimmed padding, coordinates are already correct (top-left padding means content at 0,0)
                        // No transformation needed, just use coordinates directly
                        transformedPoints = pred.points;
                        // #region agent log
                        if (i === 0) {
                            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:90',message:'Using preview image - no coordinate transformation needed',data:{usingPreview:true,processedSize:meta.processed_size,contentArea,firstPoint:pred.points[0]},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
                        }
                        // #endregion
                    } else if (meta && meta.processed_size) {
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
                        ]);

                        // #region agent log
                        if (i === 0) {
                            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:110',message:'Transforming coordinates (fallback)',data:{originalPoint:pred.points[0],transformedPoint:transformedPoints[0],scaleX,scaleY,processedSize:meta.processed_size,actualImageSize:[dimensions.w,dimensions.h]},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
                        }
                        // #endregion
                    } else {
                        // #region agent log
                        if (i === 0) {
                            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:85',message:'Skipping transformation - no metadata',data:{hasMeta:!!meta,meta:meta},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
                        }
                        // #endregion
                    }

                    // #region agent log
                    if (i === 0) {
                        const pointsStr = transformedPoints.map(p => p.join(',')).join(' ');
                        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'PolygonOverlay.tsx:70',message:'Rendering first polygon - SVG points string',data:{polygonIndex:i,pointsString:pointsStr,pointsArray:transformedPoints,viewBox:`0 0 ${dimensions.w} ${dimensions.h}`},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
                    }
                    // #endregion
                    return (
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
