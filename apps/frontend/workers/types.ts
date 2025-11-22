export type WorkerTaskType =
  | "autocontrast"
  | "gaussian-blur"
  | "resize"
  | "rembg-lite";

export interface WorkerTask<TParams = Record<string, unknown>> {
  taskId: string;
  type: WorkerTaskType;
  params: TParams;
  imageBuffer: ArrayBuffer;
  metadata?: Record<string, unknown>;
}

export interface WorkerResult<TPayload = Record<string, unknown>> {
  taskId: string;
  status: "success" | "error";
  payload?: TPayload;
  error?: string;
  durationMs: number;
  routedBackend?: "client" | "server-rembg";
  notes?: string[];
}

export interface TransformContext {
  canvas: OffscreenCanvas;
  ctx: OffscreenCanvasRenderingContext2D;
}

export type TransformHandler = (
  task: WorkerTask,
  ctx: TransformContext,
) => Promise<ImageBitmap>;
