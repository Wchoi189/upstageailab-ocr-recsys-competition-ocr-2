import { useState, useEffect, useRef, useMemo } from "react";
import type React from "react";
import type {
  CheckpointWithMetadata,
  TextRegion,
  InferencePreviewResponse,
  InferenceMetadata,
} from "../../api/inference";
import { runInferencePreview } from "../../api/inference";
import { Spinner } from "../ui/Spinner";

/**
 * Inference parameters
 */
export interface InferenceParams {
  confidenceThreshold: number;
  nmsThreshold: number;
}

/**
 * Props for InferencePreviewCanvas component
 */
interface InferencePreviewCanvasProps {
  imageFile: File | null;
  checkpoint: CheckpointWithMetadata | null;
  params: InferenceParams;
  onError?: (message: string) => void;
  onSuccess?: (message: string) => void;
}

/**
 * Inference preview canvas with polygon overlays
 *
 * Displays image with detected text regions as polygon overlays
 */
export function InferencePreviewCanvas({
  imageFile,
  checkpoint,
  params,
  onError,
  onSuccess,
}: InferencePreviewCanvasProps): React.JSX.Element {
  const [result, setResult] = useState<InferencePreviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageBitmap, setImageBitmap] = useState<ImageBitmap | null>(null);
  // BUG-001: separate display bitmap so we can switch to the backend's
  // preprocessed image (coordinate-aligned with polygons) when available.
  const [displayBitmap, setDisplayBitmap] = useState<ImageBitmap | null>(null);

  // Load image
  useEffect(() => {
    if (!imageFile) {
      setImageBitmap(null);
      setDisplayBitmap(null);
      return;
    }

    const loadImage = async (): Promise<void> => {
      try {
        const bitmap = await createImageBitmap(imageFile);
        setImageBitmap(bitmap);
        // BUG-001: Only set displayBitmap to original image if:
        // 1. We don't have a checkpoint selected (no inference will run)
        // 2. OR we don't have a result yet (inference hasn't completed)
        // This prevents race condition where original image renders before preview loads.
        // If inference will run, displayBitmap will be set when preview image loads.
        if (!checkpoint || !result?.preview_image_base64) {
          setDisplayBitmap(bitmap);
        }
      } catch (err) {
        console.error("Failed to load image:", err);
        setImageBitmap(null);
        setDisplayBitmap(null);
      }
    };

    loadImage();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageFile, checkpoint]); // Depend on checkpoint too - if checkpoint exists, inference will run and provide preview

  // Ref to track if inference should be cancelled
  const cancelledRef = useRef(false);

  // Ref to track last inference parameters to avoid duplicate calls
  const lastInferenceRef = useRef<string>("");

  // Memoize inference key to detect actual changes (not just object reference changes)
  // This prevents infinite loops when checkpoint or params objects are recreated with same values
  const inferenceKey = useMemo(() => {
    if (!imageFile || !checkpoint || !imageBitmap) return null;
    // Create a stable key from actual values, not object references
    return `${checkpoint.checkpoint_path}|${params.confidenceThreshold}|${params.nmsThreshold}|${imageFile.name}|${imageFile.size}`;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    checkpoint?.checkpoint_path,
    params.confidenceThreshold,
    params.nmsThreshold,
    imageFile?.name,
    imageFile?.size,
    imageBitmap,
  ]);

  // Run inference when actual values change (not just object references)
  // Note: onError and onSuccess are intentionally excluded from deps to prevent infinite loops
  // They are callback props that may be recreated on each render by the parent component
  useEffect(() => {
    if (!inferenceKey || !imageFile || !checkpoint || !imageBitmap) {
      setResult(null);
      lastInferenceRef.current = ""; // Reset when inputs are cleared
      return;
    }

    // Skip if this is the same inference request (prevent infinite loops)
    // This handles cases where objects are recreated but values haven't changed
    if (lastInferenceRef.current === inferenceKey) {
      return;
    }

    // Mark this inference key as processed before starting async operation
    lastInferenceRef.current = inferenceKey;

    // Reset cancellation flag for this effect run
    cancelledRef.current = false;

    const runInference = async (): Promise<void> => {
      setLoading(true);
      setError(null);
      // BUG-001: Clear displayBitmap when inference starts to prevent rendering with original image
      // dimensions during the race condition. It will be set to preview image when it loads.
      setDisplayBitmap(null);

      try {
        // Convert image to base64
        const arrayBuffer = await imageFile.arrayBuffer();
        const base64 = btoa(
          new Uint8Array(arrayBuffer).reduce(
            (data, byte) => data + String.fromCharCode(byte),
            "",
          ),
        );

        // Run inference
        const response = await runInferencePreview({
          checkpoint_path: checkpoint.checkpoint_path,
          image_base64: base64,
          confidence_threshold: params.confidenceThreshold,
          nms_threshold: params.nmsThreshold,
        });

        // Check if effect was cancelled before updating state
        if (cancelledRef.current) return;

        // Double-check inference key hasn't changed during async operation
        if (lastInferenceRef.current !== inferenceKey) return;

        setResult(response);
        onSuccess?.(
          `Inference completed: found ${response.regions.length} text regions in ${response.processing_time_ms.toFixed(0)}ms`
        );
      } catch (err) {
        // Check if effect was cancelled before updating state
        if (cancelledRef.current) return;

        // Double-check inference key hasn't changed during async operation
        if (lastInferenceRef.current !== inferenceKey) return;

        const errorMessage = err instanceof Error ? err.message : "Inference failed";
        setError(errorMessage);
        setResult(null);
        onError?.(errorMessage);
      } finally {
        if (!cancelledRef.current && lastInferenceRef.current === inferenceKey) {
          setLoading(false);
        }
      }
    };

    runInference();

    // Cleanup function to cancel if dependencies change
    return () => {
      cancelledRef.current = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inferenceKey]); // Only depend on inferenceKey - it already captures all necessary changes
  // imageFile, checkpoint, params are intentionally excluded to prevent infinite loops
  // The inferenceKey memoization captures their actual values, not object references

  // BUG-001: when backend provides a preprocessed preview image (aligned with
  // polygon coordinates), prefer that over the raw uploaded image.
  // IMPORTANT: Don't render with original image if we're expecting a preview image.
  useEffect(() => {
    const previewBase64 = result?.preview_image_base64;

    // If we have a result but no preview image yet, clear displayBitmap to prevent
    // rendering with wrong dimensions (race condition fix)
    if (result && !previewBase64) {
      // No backend preview; fall back to the client-side image only if we have no result yet.
      // If we have a result, wait for preview or use fallback after timeout.
      setDisplayBitmap(imageBitmap);
      return;
    }

    if (!previewBase64) {
      // No result yet or no preview image - use original image
      setDisplayBitmap(imageBitmap);
      return;
    }

    let objectUrl: string | null = null;

    const loadPreviewImage = async (): Promise<void> => {
      try {
        const binary = atob(previewBase64);
        const len = binary.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        // BUG-001: Backend now uses JPEG encoding (quality=85) instead of PNG to reduce file size (~10x smaller)
        const blob = new Blob([bytes], { type: "image/jpeg" });
        objectUrl = URL.createObjectURL(blob);
        const bitmap = await createImageBitmap(blob);

        // BUG-001: Verify preview image dimensions match expected processed_size
        if (result?.meta) {
          const [processedWidth, processedHeight] = result.meta.processed_size;
          if (bitmap.width !== processedWidth || bitmap.height !== processedHeight) {
            console.warn(
              `BUG-001: Preview image size mismatch - loaded: ${bitmap.width}x${bitmap.height}, ` +
              `expected: ${processedWidth}x${processedHeight}. Using preview anyway.`
            );
          }
        }

        setDisplayBitmap(bitmap);
      } catch (e) {
        console.error("BUG-001: Failed to load preview image from backend:", e);
        // On failure, continue using the local image bitmap.
        setDisplayBitmap(imageBitmap);
      }
    };

    void loadPreviewImage();

    return () => {
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [result?.preview_image_base64, result?.meta, imageBitmap, result]);


  // Draw image and polygons
  useEffect(() => {
    if (!canvasRef.current || !displayBitmap) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // BUG-001: Don't render if we have a result with preview_image_base64 but displayBitmap
    // dimensions don't match processed_size (prevents rendering with wrong image during race condition)
    // Also don't render if we're loading (inference in progress) and don't have preview yet
    if (result?.preview_image_base64 && result?.meta) {
      const [processedWidth, processedHeight] = result.meta.processed_size;
      if (displayBitmap.width !== processedWidth || displayBitmap.height !== processedHeight) {
        // Still loading preview image - don't render yet
        console.debug(
          `BUG-001: Waiting for preview image - current: ${displayBitmap.width}x${displayBitmap.height}, ` +
          `expected: ${processedWidth}x${processedHeight}`
        );
        return;
      }
    }

    // BUG-001: Don't render if we're loading and don't have a preview image yet
    // This prevents rendering with original image dimensions during inference
    if (loading && !result?.preview_image_base64) {
      console.debug("BUG-001: Inference in progress, waiting for preview image");
      return;
    }

    // BUG-001: Use fixed canvas size (720x720) to provide equal padding on all sides
    // This ensures images are perfectly centered with consistent left/right/top/bottom padding
    const imageWidth = displayBitmap.width;
    const imageHeight = displayBitmap.height;
    const CANVAS_SIZE = 720; // Fixed size for consistent padding and better visibility

    // Always use fixed 720x720 canvas size for consistent padding
    // This ensures equal padding on all sides regardless of image size
    const canvasSize = CANVAS_SIZE;

    // BUG-001: Data Contract verification - displayBitmap should match processed_size
    if (result?.meta) {
      const [processedWidth, processedHeight] = result.meta.processed_size;
      if (imageWidth !== processedWidth || imageHeight !== processedHeight) {
        console.warn(
          `BUG-001: Data Contract Mismatch - displayBitmap (${imageWidth}x${imageHeight}) ` +
          `does not match meta.processed_size (${processedWidth}x${processedHeight})`
        );
      } else {
        console.debug(
          `BUG-001: Data Contract Verified - displayBitmap matches processed_size: ${imageWidth}x${imageHeight}, ` +
          `coordinate_system=${result.meta.coordinate_system}, padding=${JSON.stringify(result.meta.padding)}`
        );
      }
    }

    // Set canvas to fixed square dimensions (720x720)
    canvas.width = canvasSize;
    canvas.height = canvasSize;

    // Calculate centering offsets to perfectly center the image within the square canvas
    // This provides equal padding on all sides: (720 - imageSize) / 2
    const dx = (canvasSize - imageWidth) / 2;
    const dy = (canvasSize - imageHeight) / 2;

    // BUG-001: Debug logging to verify centering calculation
    console.log(
      `BUG-001: Canvas centering calculation:\n` +
      `  Image dimensions: ${imageWidth}x${imageHeight}\n` +
      `  Canvas dimensions: ${canvasSize}x${canvasSize}\n` +
      `  Horizontal offset (dx): ${dx.toFixed(1)}px (should be equal left/right padding)\n` +
      `  Vertical offset (dy): ${dy.toFixed(1)}px (should be equal top/bottom padding)\n` +
      `  Expected padding: ${dx.toFixed(1)}px on all sides\n` +
      `  Meta padding (backend): ${JSON.stringify(result?.meta?.padding || {})}`
    );

    // Clear canvas with black background (for letterboxing)
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw image centered in the square canvas
    ctx.drawImage(displayBitmap, dx, dy);

    // Draw polygons if results available
    if (result?.regions) {
      result.regions.forEach((region: TextRegion) => {
        // Use metadata to determine coordinate handling
        drawPolygon(ctx, region, imageWidth, imageHeight, dx, dy, result.meta);
      });
    }
  }, [displayBitmap, result, loading]);

  /**
   * Draw polygon overlay for text region
   * BUG-001: Applies centering offsets (dx/dy) to align polygons with the centered image in square container.
   * Backend owns all geometry mapping - polygons are in absolute pixel coordinates relative to processed_size frame.
   * Viewer only handles display centering, no padding offsets or scaling.
   */
  const drawPolygon = (
    ctx: CanvasRenderingContext2D,
    region: TextRegion,
    displayWidth: number,
    displayHeight: number,
    dx: number = 0,
    dy: number = 0,
    meta: InferenceMetadata | null | undefined = undefined,
  ): void => {
    if (region.polygon.length < 3) return;

    // BUG-001: Data Contract enforcement - backend provides coordinate_system in meta.
    // Default to "pixel" if meta is missing (backward compatibility during transition).
    const coordinateSystem = meta?.coordinate_system || "pixel";
    const isNormalized = coordinateSystem === "normalized";

    // BUG-001: Debug logging for first polygon to verify coordinate handling
    if (region.polygon.length > 0 && Math.random() < 0.01) { // Log ~1% of polygons to avoid spam
      const [firstX, firstY] = region.polygon[0];
      console.debug(
        `BUG-001: Polygon coordinate handling - system=${coordinateSystem}, ` +
        `first_point=(${firstX}, ${firstY}), display_size=${displayWidth}x${displayHeight}, ` +
        `centering_offset=(${dx}, ${dy})`
      );
    }

    const offsetPolygon = region.polygon.map(([x, y]) => {
      if (isNormalized) {
        // Case 1: Normalized coordinates (0-1) - scale to display bitmap size
        // BUG-001: These are relative to the full processed_size frame (including padding).
        return [x * displayWidth + dx, y * displayHeight + dy];
      } else {
        // Case 2: Absolute pixel coordinates (coordinate_system="pixel")
        // BUG-001: Backend maps polygons to full processed_size frame (640x640).
        // With top_left padding, content is at [0-resized_w, 0-resized_h] within [0-640, 0-640].
        // Viewer only applies display centering (dx/dy) - no padding offsets needed.
        return [x + dx, y + dy];
      }
    });

    // Draw polygon stroke
    ctx.beginPath();
    ctx.moveTo(offsetPolygon[0][0], offsetPolygon[0][1]);
    for (let i = 1; i < offsetPolygon.length; i++) {
      ctx.lineTo(offsetPolygon[i][0], offsetPolygon[i][1]);
    }
    ctx.closePath();

    // Style based on confidence
    const alpha = Math.max(0.3, region.confidence);
    ctx.strokeStyle = `rgba(0, 255, 0, ${alpha})`;
    // BUG-001: Reduced line width from 2 to 1 for better visibility and cleaner appearance
    ctx.lineWidth = 1;
    ctx.stroke();

    // Fill with semi-transparent color (reduced opacity for less visual clutter)
    ctx.fillStyle = `rgba(0, 255, 0, ${alpha * 0.15})`;
    ctx.fill();

    // Draw text label with reduced font size for less clutter
    if (region.text) {
      const [x, y] = offsetPolygon[0];
      ctx.font = "12px Arial"; // Reduced from 14px to 12px
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      const textMetrics = ctx.measureText(region.text);
      ctx.fillRect(x, y - 16, textMetrics.width + 6, 16); // Reduced height from 20 to 16
      ctx.fillStyle = "white";
      ctx.fillText(region.text, x + 3, y - 4); // Adjusted padding
    }

    // BUG-001: Hide confidence label to reduce clutter (can be re-enabled if needed)
    // Confidence information is still visible through stroke alpha/color
    // Uncomment below to show confidence with smaller font:
    /*
    const confText = `${(region.confidence * 100).toFixed(0)}%`; // Removed decimal for shorter text
    const [x, y] = offsetPolygon[0];
    ctx.font = "10px Arial"; // Reduced from 12px to 10px
    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fillRect(x, y - 32, ctx.measureText(confText).width + 6, 14);
    ctx.fillStyle = "yellow";
    ctx.fillText(confText, x + 3, y - 20);
    */
  };

  if (!imageFile) {
    return (
      <div
        style={{
          padding: "2rem",
          textAlign: "center",
          color: "#666",
          border: "2px dashed #ccc",
          borderRadius: "8px",
          // @ts-ignore
          cursor: "pointer",
        }}
      >
        <p>Upload an image to run inference</p>
      </div>
    );
  }

  if (!checkpoint) {
    return (
      <div
        style={{
          padding: "2rem",
          textAlign: "center",
          color: "#666",
          border: "2px dashed #ccc",
          borderRadius: "8px",
        }}
      >
        <p>Select a checkpoint to run inference</p>
      </div>
    );
  }

  return (
    <div>
      {/* Canvas */}
      <div style={{ marginBottom: "1rem", position: "relative", display: "flex", justifyContent: "center" }}>
        <canvas
          ref={canvasRef}
          style={{
            // BUG-001: Fixed canvas size (720x720) with explicit aspect ratio to maintain square shape
            // This ensures equal padding on all sides even when scaled by CSS
            width: "720px",
            height: "720px",
            maxWidth: "100%",
            aspectRatio: "1 / 1", // Explicitly maintain 1:1 aspect ratio when scaled
            display: "block",
            margin: "0 auto", // Center the canvas horizontally in its container
            border: "1px solid #ddd",
            borderRadius: "4px",
          }}
        />
        {loading && (
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              backgroundColor: "rgba(0, 0, 0, 0.7)",
              color: "white",
              padding: "1.5rem",
              borderRadius: "8px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "0.75rem",
            }}
          >
            <Spinner size="large" color="#ffffff" />
            <span>Running inference...</span>
          </div>
        )}
      </div>

      {/* Results Info */}
      <div style={{ fontSize: "0.875rem", color: "#666" }}>
        {error && <div style={{ color: "red" }}>Error: {error}</div>}
        {result && !error && (
          <div>
            <div>
              <strong>Detected Regions:</strong> {result.regions.length}
            </div>
            <div>
              <strong>Processing Time:</strong>{" "}
              {result.processing_time_ms.toFixed(2)}ms
            </div>
            {result.notes.length > 0 && (
              <div style={{ marginTop: "0.5rem", color: "#999" }}>
                {result.notes.map((note, idx) => (
                  <div key={idx}>• {note}</div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
