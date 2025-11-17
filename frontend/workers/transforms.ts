import type { TransformContext, TransformHandler, WorkerTask } from "./types";

function decodeImage(task: WorkerTask, ctx: TransformContext): ImageData {
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
  }) as unknown as ImageData;
}

export const runAutoContrast: TransformHandler = async (task, ctx) => {
  const imageData = decodeImage(task, ctx);
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
  const imageData = decodeImage(task, ctx);
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

export const runResize: TransformHandler = async (task, ctx) => {
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

export const runRembgLite: TransformHandler = async (task, ctx) => {
  // Placeholder: actual ONNX runtime wiring lands when the bundle is available.
  const bitmap = await runAutoContrast(task, ctx);
  return bitmap;
};


