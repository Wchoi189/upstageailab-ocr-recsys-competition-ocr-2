import { useState } from "react";
import type React from "react";
import { PreprocessingCanvas } from "../components/preprocessing/PreprocessingCanvas";
import {
  ParameterControls,
  type PreprocessingParams,
} from "../components/preprocessing/ParameterControls";

/**
 * Preprocessing Studio page
 *
 * Interactive image preprocessing with real-time preview
 */
export function Preprocessing(): React.JSX.Element {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [params, setParams] = useState<PreprocessingParams>({
    autocontrast: false,
    blur: false,
    blurKernelSize: 5,
    resize: false,
    resizeWidth: 1024,
    resizeHeight: 1024,
    rembg: false,
  });

  const handleFileChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ): void => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setImageFile(file);
    }
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h1 style={{ color: "#213547" }}>Preprocessing Studio</h1>
      <p style={{ color: "#666", marginBottom: "2rem" }}>
        Interactive image preprocessing with real-time preview powered by
        Web Workers
      </p>

      {/* File Upload */}
      <div style={{ marginBottom: "2rem" }}>
        <label
          htmlFor="image-upload"
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
          id="image-upload"
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
      </div>

      {/* Main Content */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "auto 1fr",
          gap: "2rem",
          alignItems: "start",
        }}
      >
        {/* Parameter Controls */}
        <ParameterControls params={params} onChange={setParams} />

        {/* Canvas Viewer */}
        <PreprocessingCanvas imageFile={imageFile} params={params} />
      </div>
    </div>
  );
}
