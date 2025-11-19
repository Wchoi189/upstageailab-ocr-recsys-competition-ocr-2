import { useState, useEffect, useRef } from "react";
import type React from "react";
import type {
  CheckpointWithMetadata,
  TextRegion,
  InferencePreviewResponse,
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

  // Load image
  useEffect(() => {
    if (!imageFile) {
      setImageBitmap(null);
      return;
    }

    const loadImage = async (): Promise<void> => {
      try {
        const bitmap = await createImageBitmap(imageFile);
        setImageBitmap(bitmap);
      } catch (err) {
        console.error("Failed to load image:", err);
        setImageBitmap(null);
      }
    };

    loadImage();
  }, [imageFile]);

  // Run inference when image, checkpoint, or params change
  useEffect(() => {
    if (!imageFile || !checkpoint || !imageBitmap) {
      setResult(null);
      return;
    }

    const runInference = async (): Promise<void> => {
      setLoading(true);
      setError(null);

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

        setResult(response);
        onSuccess?.(
          `Inference completed: found ${response.regions.length} text regions in ${response.processing_time_ms.toFixed(0)}ms`
        );
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Inference failed";
        setError(errorMessage);
        setResult(null);
        onError?.(errorMessage);
      } finally {
        setLoading(false);
      }
    };

    runInference();
  }, [imageFile, checkpoint, params, imageBitmap, onError, onSuccess]);

  // Draw image and polygons
  useEffect(() => {
    if (!canvasRef.current || !imageBitmap) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size to image size
    canvas.width = imageBitmap.width;
    canvas.height = imageBitmap.height;

    // Draw image
    ctx.drawImage(imageBitmap, 0, 0);

    // Draw polygons if results available
    if (result?.regions) {
      result.regions.forEach((region: TextRegion, idx: number) => {
        drawPolygon(ctx, region, idx);
      });
    }
  }, [imageBitmap, result]);

  /**
   * Draw polygon overlay for text region
   */
  const drawPolygon = (
    ctx: CanvasRenderingContext2D,
    region: TextRegion,
    index: number,
  ): void => {
    if (region.polygon.length < 3) return;

    // Draw polygon stroke
    ctx.beginPath();
    ctx.moveTo(region.polygon[0][0], region.polygon[0][1]);
    for (let i = 1; i < region.polygon.length; i++) {
      ctx.lineTo(region.polygon[i][0], region.polygon[i][1]);
    }
    ctx.closePath();

    // Style based on confidence
    const alpha = Math.max(0.3, region.confidence);
    ctx.strokeStyle = `rgba(0, 255, 0, ${alpha})`;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Fill with semi-transparent color
    ctx.fillStyle = `rgba(0, 255, 0, ${alpha * 0.2})`;
    ctx.fill();

    // Draw text label
    if (region.text) {
      const [x, y] = region.polygon[0];
      ctx.font = "14px Arial";
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(x, y - 20, ctx.measureText(region.text).width + 8, 20);
      ctx.fillStyle = "white";
      ctx.fillText(region.text, x + 4, y - 6);
    }

    // Draw confidence label
    const confText = `${(region.confidence * 100).toFixed(1)}%`;
    const [x, y] = region.polygon[0];
    ctx.font = "12px Arial";
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fillRect(x, y - 40, ctx.measureText(confText).width + 8, 16);
    ctx.fillStyle = "yellow";
    ctx.fillText(confText, x + 4, y - 28);
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
      <div style={{ marginBottom: "1rem", position: "relative" }}>
        <canvas
          ref={canvasRef}
          style={{
            maxWidth: "100%",
            height: "auto",
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
