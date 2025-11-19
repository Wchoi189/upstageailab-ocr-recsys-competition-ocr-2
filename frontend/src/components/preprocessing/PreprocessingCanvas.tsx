import { useEffect, useRef, useState } from "react";
import type React from "react";
import type { WorkerTask } from "../../../workers/types";
import { useWorkerTask } from "../../hooks/useWorkerTask";
import type { PreprocessingParams } from "./ParameterControls";

/**
 * Props for PreprocessingCanvas component
 */
interface PreprocessingCanvasProps {
  imageFile: File | null;
  params: PreprocessingParams;
}

/**
 * Before/after image viewer with side-by-side layout
 *
 * Displays original and processed images using web worker pipeline
 */
export function PreprocessingCanvas({
  imageFile,
  params,
}: PreprocessingCanvasProps): React.JSX.Element {
  const [originalBitmap, setOriginalBitmap] = useState<ImageBitmap | null>(
    null,
  );
  const [workerTask, setWorkerTask] = useState<WorkerTask | null>(null);
  const beforeCanvasRef = useRef<HTMLCanvasElement>(null);
  const afterCanvasRef = useRef<HTMLCanvasElement>(null);

  // Use worker hook with 75ms debouncing for slider spam handling
  const { result, loading, error } = useWorkerTask(workerTask, {
    debounceMs: 75,
    priority: "normal",
  });

  // Load original image
  useEffect(() => {
    if (!imageFile) {
      setOriginalBitmap(null);
      setWorkerTask(null);
      return;
    }

    const loadImage = async (): Promise<void> => {
      try {
        const bitmap = await createImageBitmap(imageFile);
        setOriginalBitmap(bitmap);
      } catch (err) {
        console.error("Failed to load image:", err);
        setOriginalBitmap(null);
      }
    };

    loadImage();
  }, [imageFile]);

  // Create worker task when params or image changes
  useEffect(() => {
    if (!originalBitmap) {
      setWorkerTask(null);
      return;
    }

    const createTask = async (): Promise<void> => {
      try {
        // Convert ImageBitmap to ArrayBuffer
        const canvas = new OffscreenCanvas(
          originalBitmap.width,
          originalBitmap.height,
        );
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.drawImage(originalBitmap, 0, 0);
        const blob = await canvas.convertToBlob();
        const arrayBuffer = await blob.arrayBuffer();

        // Build task based on enabled parameters (priority order)
        let taskType: WorkerTask["type"] = "autocontrast";
        const taskParams: Record<string, unknown> = {};

        if (params.rembg) {
          taskType = "rembg-lite";
          // Note: Hybrid routing happens in worker or backend
        } else if (params.blur) {
          taskType = "gaussian-blur";
          taskParams.kernelSize = params.blurKernelSize;
        } else if (params.resize) {
          taskType = "resize";
          taskParams.width = params.resizeWidth;
          taskParams.height = params.resizeHeight;
        } else if (params.autocontrast) {
          taskType = "autocontrast";
        } else {
          // No transform enabled, skip worker task
          setWorkerTask(null);
          return;
        }

        const task: WorkerTask = {
          taskId: `task-${Date.now()}`,
          type: taskType,
          params: taskParams,
          imageBuffer: arrayBuffer,
        };

        setWorkerTask(task);
      } catch (err) {
        console.error("Failed to create worker task:", err);
      }
    };

    createTask();
  }, [originalBitmap, params]);

  // Draw original image on "before" canvas
  useEffect(() => {
    if (!originalBitmap || !beforeCanvasRef.current) return;

    const canvas = beforeCanvasRef.current;
    canvas.width = originalBitmap.width;
    canvas.height = originalBitmap.height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(originalBitmap, 0, 0);
  }, [originalBitmap]);

  // Draw processed image on "after" canvas
  useEffect(() => {
    if (!afterCanvasRef.current) return;

    const canvas = afterCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (result?.status === "success" && result.payload?.bitmap) {
      const bitmap = result.payload.bitmap as ImageBitmap;
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      ctx.drawImage(bitmap, 0, 0);
    } else if (!loading && !workerTask) {
      // No transform, show original
      if (originalBitmap) {
        canvas.width = originalBitmap.width;
        canvas.height = originalBitmap.height;
        ctx.drawImage(originalBitmap, 0, 0);
      }
    }
  }, [result, loading, workerTask, originalBitmap]);

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
        <p>Upload an image to start preprocessing</p>
      </div>
    );
  }

  return (
    <div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "1rem",
          marginBottom: "1rem",
        }}
      >
        {/* Before Canvas */}
        <div>
          <h4 style={{ marginTop: 0, marginBottom: "0.5rem" }}>Before</h4>
          <canvas
            ref={beforeCanvasRef}
            style={{
              maxWidth: "100%",
              height: "auto",
              border: "1px solid #ddd",
              borderRadius: "4px",
            }}
          />
        </div>

        {/* After Canvas */}
        <div>
          <h4 style={{ marginTop: 0, marginBottom: "0.5rem" }}>
            After
            {loading && (
              <span style={{ marginLeft: "0.5rem", color: "#666" }}>
                (Processing...)
              </span>
            )}
          </h4>
          <canvas
            ref={afterCanvasRef}
            style={{
              maxWidth: "100%",
              height: "auto",
              border: "1px solid #ddd",
              borderRadius: "4px",
            }}
          />
        </div>
      </div>

      {/* Status info */}
      <div style={{ fontSize: "0.875rem", color: "#666" }}>
        {error && (
          <div style={{ color: "red" }}>Error: {error}</div>
        )}
        {result && !error && (
          <div>
            Processing time: {result.durationMs.toFixed(2)}ms
            {result.routedBackend && ` (${result.routedBackend})`}
          </div>
        )}
      </div>
    </div>
  );
}
