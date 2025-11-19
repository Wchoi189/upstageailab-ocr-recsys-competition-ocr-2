import { useState } from "react";
import type React from "react";
import { CheckpointPicker } from "../components/inference/CheckpointPicker";
import { InferencePreviewCanvas } from "../components/inference/InferencePreviewCanvas";
import { InferenceControls } from "../components/inference/InferenceControls";
import type { CheckpointWithMetadata } from "../api/inference";
import type { InferenceParams } from "../components/inference/InferencePreviewCanvas";
import { validateImageFile } from "../utils/imageValidation";
import { useToast } from "../hooks/useToast";

/**
 * Inference Studio page
 *
 * Run inference with trained checkpoints and visualize results
 */
export function Inference(): React.JSX.Element {
  const { showToast, ToastContainer } = useToast();
  const [selectedCheckpoint, setSelectedCheckpoint] =
    useState<CheckpointWithMetadata | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [params, setParams] = useState<InferenceParams>({
    confidenceThreshold: 0.5,
    nmsThreshold: 0.4,
  });

  const handleFileChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ): void => {
    const file = event.target.files?.[0];
    setUploadError(null);

    const validationResult = validateImageFile(file);
    if (!validationResult.valid) {
      const errorMessage = validationResult.error?.message || "Invalid file";
      setUploadError(errorMessage);
      showToast(errorMessage, "error");
      setImageFile(null);
      return;
    }

    if (file) {
      setImageFile(file);
      showToast("Image uploaded successfully", "success");
    }
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h1 style={{ color: "#213547" }}>Inference Studio</h1>
      <p style={{ color: "#666", marginBottom: "2rem" }}>
        Run inference with trained checkpoints and visualize OCR results
      </p>

      {/* File Upload */}
      <div style={{ marginBottom: "2rem" }}>
        <label
          htmlFor="inference-image-upload"
          style={{
            display: "inline-block",
            padding: "0.5rem 1rem",
            backgroundColor: "#007bff",
            color: "white",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          Upload Image
        </label>
        <input
          id="inference-image-upload"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />
        {imageFile && (
          <span style={{ marginLeft: "1rem", color: "#666" }}>
            {imageFile.name}
          </span>
        )}
        {uploadError && (
          <div
            style={{
              marginTop: "0.5rem",
              padding: "0.75rem",
              backgroundColor: "#fee",
              border: "1px solid #fcc",
              borderRadius: "4px",
              color: "#c33",
            }}
          >
            {uploadError}
          </div>
        )}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "400px 1fr",
          gap: "2rem",
          alignItems: "start",
        }}
      >
        {/* Left: Checkpoint Picker */}
        <div>
          <CheckpointPicker
            selectedCheckpoint={selectedCheckpoint}
            onSelect={setSelectedCheckpoint}
          />

          {/* Inference Controls */}
          {selectedCheckpoint && imageFile && (
            <div style={{ marginTop: "1rem" }}>
              <InferenceControls params={params} onChange={setParams} />
            </div>
          )}
        </div>

        {/* Right: Inference Preview Canvas */}
        <InferencePreviewCanvas
          imageFile={imageFile}
          checkpoint={selectedCheckpoint}
          params={params}
          onError={(message) => showToast(message, "error")}
          onSuccess={(message) => showToast(message, "success")}
        />
      </div>
      <ToastContainer />
    </div>
  );
}
