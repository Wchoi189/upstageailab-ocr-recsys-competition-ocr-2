import type { TransformContext, TransformHandler, WorkerTask } from "./types";
import * as ort from "onnxruntime-web";

// Lazy-loaded ONNX session cache
let onnxSession: ort.InferenceSession | null = null;
const MODEL_URL = "/models/u2net.onnx";

/**
 * Load ONNX model session (lazy initialization)
 */
async function loadOnnxSession(): Promise<ort.InferenceSession> {
  if (!onnxSession) {
    try {
      console.log("[rembg] Loading ONNX model from:", MODEL_URL);

      // Configure ONNX Runtime for Web Workers
      // Set WASM file paths - use CDN to avoid path issues in workers
      // In workers, relative paths from node_modules don't work, so we use a CDN
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
      ort.env.wasm.simd = true;
      ort.env.wasm.proxy = false; // Disable proxy worker in worker context

      console.log("[rembg] ONNX environment configured:", {
        wasmPaths: ort.env.wasm.wasmPaths,
        simd: ort.env.wasm.simd,
        proxy: ort.env.wasm.proxy,
      });

      // Create session with WASM execution provider
      console.log("[rembg] Creating inference session...");
      onnxSession = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });

      console.log("[rembg] ONNX session created successfully", {
        inputNames: onnxSession.inputNames,
        outputNames: onnxSession.outputNames,
      });
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      console.error("[rembg] Failed to load ONNX model:", errorMsg);
      console.error("[rembg] Error stack:", error instanceof Error ? error.stack : "No stack");
      throw new Error(`Failed to load ONNX model: ${errorMsg}`);
    }
  }
  return onnxSession;
}

function decodeImage(task: WorkerTask, ctx: TransformContext): Promise<ImageData> {
  const blob = new Blob([task.imageBuffer]);
  const imageBitmapPromise = createImageBitmap(blob);

  return new Promise((resolve, reject) => {
    imageBitmapPromise
      .then((bitmap) => {
        ctx.canvas.width = bitmap.width;
        ctx.canvas.height = bitmap.height;
        ctx.ctx.drawImage(bitmap, 0, 0);
        const imageData = ctx.ctx.getImageData(0, 0, bitmap.width, bitmap.height);
        bitmap.close();
        resolve(imageData);
      })
      .catch(reject);
  });
}

export const runAutoContrast: TransformHandler = async (task, ctx) => {
  const imageData = await decodeImage(task, ctx);
  const data = imageData.data;

  let min = 255;
  let max = 0;
  for (let i = 0; i < data.length; i += 4) {
    const luminance = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
    min = Math.min(min, luminance);
    max = Math.max(max, luminance);
  }
  const scale = 255 / Math.max(1, max - min);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = (data[i] - min) * scale;
    data[i + 1] = (data[i + 1] - min) * scale;
    data[i + 2] = (data[i + 2] - min) * scale;
  }
  ctx.ctx.putImageData(imageData, 0, 0);
  return ctx.canvas.transferToImageBitmap();
};

export const runGaussianBlur: TransformHandler = async (task, ctx) => {
  const imageData = await decodeImage(task, ctx);
  const kernelSize = Number((task.params.kernelSize ?? 3) as number);
  const radius = Math.max(1, Math.floor(kernelSize / 2));
  const temp = ctx.ctx.createImageData(imageData.width, imageData.height);
  temp.data.set(imageData.data);

  for (let y = radius; y < imageData.height - radius; y += 1) {
    for (let x = radius; x < imageData.width - radius; x += 1) {
      const offset = (y * imageData.width + x) * 4;
      let r = 0;
      let g = 0;
      let b = 0;
      let count = 0;
      for (let ky = -radius; ky <= radius; ky += 1) {
        for (let kx = -radius; kx <= radius; kx += 1) {
          const sample = offset + (ky * imageData.width + kx) * 4;
          r += temp.data[sample];
          g += temp.data[sample + 1];
          b += temp.data[sample + 2];
          count += 1;
        }
      }
      imageData.data[offset] = r / count;
      imageData.data[offset + 1] = g / count;
      imageData.data[offset + 2] = b / count;
    }
  }
  ctx.ctx.putImageData(imageData, 0, 0);
  return ctx.canvas.transferToImageBitmap();
};

export const runResize: TransformHandler = async (task, _ctx) => {
  const targetWidth = Number(task.params.width ?? 1024);
  const targetHeight = Number(task.params.height ?? 1024);
  const blob = new Blob([task.imageBuffer]);
  const bitmap = await createImageBitmap(blob);
  const offscreen = new OffscreenCanvas(targetWidth, targetHeight);
  const offCtx = offscreen.getContext("2d");
  if (!offCtx) {
    throw new Error("Failed to acquire OffscreenCanvas context for resize");
  }
  offCtx.drawImage(bitmap, 0, 0, targetWidth, targetHeight);
  bitmap.close();
  return offscreen.transferToImageBitmap();
};

/**
 * Preprocess image for u2net model
 * - Resize to 320x320
 * - Normalize to [0, 1] range
 * - Convert RGB to tensor format (1, 3, 320, 320)
 */
function preprocessImage(imageData: ImageData): Float32Array {
  const { width, height } = imageData;
  const targetSize = 320;

  // Create temporary canvas for resizing
  const tempCanvas = new OffscreenCanvas(targetSize, targetSize);
  const tempCtx = tempCanvas.getContext("2d");
  if (!tempCtx) {
    throw new Error("Failed to create temporary canvas context");
  }

  // Draw and resize
  const sourceCanvas = new OffscreenCanvas(width, height);
  const sourceCtx = sourceCanvas.getContext("2d");
  if (!sourceCtx) {
    throw new Error("Failed to create source canvas context");
  }
  sourceCtx.putImageData(imageData, 0, 0);
  tempCtx.drawImage(sourceCanvas, 0, 0, targetSize, targetSize);

  // Get resized image data
  const resizedData = tempCtx.getImageData(0, 0, targetSize, targetSize).data;

  // Convert to normalized tensor [1, 3, 320, 320]
  const tensor = new Float32Array(1 * 3 * targetSize * targetSize);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < targetSize * targetSize; i++) {
    const r = resizedData[i * 4] / 255.0;
    const g = resizedData[i * 4 + 1] / 255.0;
    const b = resizedData[i * 4 + 2] / 255.0;

    // Normalize and store in CHW format
    tensor[i] = (r - mean[0]) / std[0]; // R channel
    tensor[targetSize * targetSize + i] = (g - mean[1]) / std[1]; // G channel
    tensor[2 * targetSize * targetSize + i] = (b - mean[2]) / std[2]; // B channel
  }

  return tensor;
}

/**
 * Postprocess u2net output mask and apply to original image
 */
function postprocessMask(
  maskOutput: Float32Array,
  originalWidth: number,
  originalHeight: number,
  originalImageData: ImageData,
): ImageData {
  const maskSize = 320;
  const mask = maskOutput; // Should be [1, 1, 320, 320] flattened to 102400 values

  // Validate mask size
  if (mask.length !== maskSize * maskSize) {
    throw new Error(`Invalid mask size: expected ${maskSize * maskSize}, got ${mask.length}`);
  }

  // Resize mask back to original dimensions
  const maskCanvas = new OffscreenCanvas(maskSize, maskSize);
  const maskCtx = maskCanvas.getContext("2d");
  if (!maskCtx) {
    throw new Error("Failed to create mask canvas context");
  }

  // Create mask image data
  const maskImageData = maskCtx.createImageData(maskSize, maskSize);
  for (let i = 0; i < maskSize * maskSize; i++) {
    // Extract mask value and normalize (model outputs 0-1 range)
    // Clamp to [0, 1] then scale to [0, 255]
    const normalizedValue = Math.max(0, Math.min(1, mask[i]));
    const maskValue = Math.round(normalizedValue * 255);

    maskImageData.data[i * 4] = maskValue; // R
    maskImageData.data[i * 4 + 1] = maskValue; // G
    maskImageData.data[i * 4 + 2] = maskValue; // B
    maskImageData.data[i * 4 + 3] = 255; // A
  }
  maskCtx.putImageData(maskImageData, 0, 0);

  // Resize mask to original size
  const resizedMaskCanvas = new OffscreenCanvas(originalWidth, originalHeight);
  const resizedMaskCtx = resizedMaskCanvas.getContext("2d");
  if (!resizedMaskCtx) {
    throw new Error("Failed to create resized mask canvas context");
  }
  resizedMaskCtx.drawImage(maskCanvas, 0, 0, originalWidth, originalHeight);
  const resizedMaskData = resizedMaskCtx.getImageData(0, 0, originalWidth, originalHeight);

  // Apply mask to original image (white background)
  const result = new ImageData(originalWidth, originalHeight);
  const originalData = originalImageData.data;
  const maskData = resizedMaskData.data;

  for (let i = 0; i < originalWidth * originalHeight; i++) {
    const maskAlpha = maskData[i * 4] / 255.0;
    const r = Math.round(originalData[i * 4] * maskAlpha + 255 * (1 - maskAlpha));
    const g = Math.round(originalData[i * 4 + 1] * maskAlpha + 255 * (1 - maskAlpha));
    const b = Math.round(originalData[i * 4 + 2] * maskAlpha + 255 * (1 - maskAlpha));

    result.data[i * 4] = r;
    result.data[i * 4 + 1] = g;
    result.data[i * 4 + 2] = b;
    result.data[i * 4 + 3] = 255;
  }

  return result;
}

export const runRembgLite: TransformHandler = async (task, ctx) => {
  try {
    console.log("[rembg] Starting background removal...");

    // Load ONNX session
    const session = await loadOnnxSession();
    console.log("[rembg] ONNX session loaded successfully");

    // Decode image
    const originalImageData = await decodeImage(task, ctx);
    const { width, height } = originalImageData;
    console.log(`[rembg] Image decoded: ${width}x${height}`);

    // Preprocess image for model input
    const inputTensor = preprocessImage(originalImageData);
    console.log(`[rembg] Image preprocessed: tensor shape [1, 3, 320, 320]`);

    // Create ONNX tensor
    const tensor = new ort.Tensor("float32", inputTensor, [1, 3, 320, 320]);

    // Run inference
    const feeds: Record<string, ort.Tensor> = {};
    const inputName = session.inputNames[0];
    feeds[inputName] = tensor;

    console.log(`[rembg] Running inference with input: ${inputName}`);
    const results = await session.run(feeds);
    const outputName = session.outputNames[0];
    const outputTensor = results[outputName];
    console.log(`[rembg] Inference complete. Output: ${outputName}, shape: [${outputTensor.dims.join(", ")}]`);

    // Extract mask from output
    // u2net outputs shape [1, 1, 320, 320] - tensor data is flattened automatically
    const outputShape = outputTensor.dims;
    const rawMaskData = outputTensor.data as Float32Array;
    const expectedSize = outputShape.reduce((a, b) => a * b, 1);

    console.log(`[rembg] Mask data extracted: ${rawMaskData.length} values, shape: [${outputShape.join(", ")}], expected size: ${expectedSize}`);

    // Validate data size matches expected shape
    if (rawMaskData.length !== expectedSize) {
      throw new Error(`Mask data size mismatch: expected ${expectedSize} (shape [${outputShape.join(", ")}]), got ${rawMaskData.length}`);
    }

    // Extract the actual mask data (skip batch and channel dimensions if present)
    // For shape [1, 1, 320, 320], we need the last two dimensions (320x320)
    let maskData: Float32Array;
    if (outputShape.length === 4 && outputShape[0] === 1 && outputShape[1] === 1) {
      // Shape is [1, 1, 320, 320], data is already flattened, use as-is
      maskData = rawMaskData;
    } else if (outputShape.length === 4) {
      // Need to extract [1, 1, H, W] portion
      const h = outputShape[2];
      const w = outputShape[3];
      maskData = rawMaskData.slice(0, h * w);
    } else {
      // Unexpected shape, try to use as-is
      maskData = rawMaskData;
    }

    console.log(`[rembg] Processed mask data: ${maskData.length} values (expected ${320 * 320})`);

    // Postprocess mask and apply to original image
    const resultImageData = postprocessMask(maskData, width, height, originalImageData);

    // Draw result to canvas
    ctx.canvas.width = width;
    ctx.canvas.height = height;
    ctx.ctx.putImageData(resultImageData, 0, 0);

    console.log("[rembg] Background removal completed successfully");
    return ctx.canvas.transferToImageBitmap();
  } catch (error) {
    // Log detailed error information
    console.error("[rembg] ONNX inference failed:", error);
    console.error("[rembg] Error details:", {
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
    });

    // Fallback to autocontrast if ONNX inference fails
    console.warn("[rembg] Falling back to autocontrast");
    return runAutoContrast(task, ctx);
  }
};
