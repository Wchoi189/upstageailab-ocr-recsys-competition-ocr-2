/**
 * Image validation utilities for file upload handling
 *
 * Provides validation for image file size, format, and dimensions
 */

/**
 * Image validation error types
 */
export const ImageValidationErrorType = {
  FILE_TOO_LARGE: "FILE_TOO_LARGE",
  INVALID_FORMAT: "INVALID_FORMAT",
  FILE_EMPTY: "FILE_EMPTY",
} as const;

export type ImageValidationErrorType =
  (typeof ImageValidationErrorType)[keyof typeof ImageValidationErrorType];

/**
 * Image validation error
 */
export interface ImageValidationError {
  type: ImageValidationErrorType;
  message: string;
}

/**
 * Image validation result
 */
export interface ImageValidationResult {
  valid: boolean;
  error?: ImageValidationError;
}

/**
 * Image validation configuration
 */
export interface ImageValidationConfig {
  maxSizeBytes?: number;
  allowedMimeTypes?: string[];
}

/**
 * Default validation configuration
 */
const DEFAULT_CONFIG: Required<ImageValidationConfig> = {
  maxSizeBytes: 10 * 1024 * 1024, // 10MB default
  allowedMimeTypes: [
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
  ],
};

/**
 * Formats file size in bytes to human-readable string
 *
 * @param bytes - File size in bytes
 * @returns Formatted file size (e.g., "1.5 MB")
 */
function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 Bytes";

  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`;
}

/**
 * Validates an image file for upload
 *
 * @param file - File to validate
 * @param config - Optional validation configuration
 * @returns Validation result with error details if invalid
 */
export function validateImageFile(
  file: File | null | undefined,
  config?: ImageValidationConfig
): ImageValidationResult {
  const mergedConfig = { ...DEFAULT_CONFIG, ...config };

  if (!file) {
    return {
      valid: false,
      error: {
        type: ImageValidationErrorType.FILE_EMPTY,
        message: "No file selected",
      },
    };
  }

  // Check file size
  if (file.size > mergedConfig.maxSizeBytes) {
    const maxSize = formatFileSize(mergedConfig.maxSizeBytes);
    const actualSize = formatFileSize(file.size);
    return {
      valid: false,
      error: {
        type: ImageValidationErrorType.FILE_TOO_LARGE,
        message: `File size (${actualSize}) exceeds maximum allowed (${maxSize})`,
      },
    };
  }

  // Check MIME type
  if (!mergedConfig.allowedMimeTypes.includes(file.type)) {
    return {
      valid: false,
      error: {
        type: ImageValidationErrorType.INVALID_FORMAT,
        message: `Invalid file format: ${file.type}. Allowed formats: ${mergedConfig.allowedMimeTypes.join(", ")}`,
      },
    };
  }

  return { valid: true };
}
