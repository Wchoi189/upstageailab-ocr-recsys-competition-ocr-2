import type { InferenceParams } from "./InferencePreviewCanvas";

/**
 * Props for InferenceControls component
 */
interface InferenceControlsProps {
  params: InferenceParams;
  onChange: (params: InferenceParams) => void;
}

/**
 * Hyperparameter controls for inference
 *
 * Provides sliders for confidence threshold, NMS threshold, etc.
 */
export function InferenceControls({
  params,
  onChange,
}: InferenceControlsProps): JSX.Element {
  const handleSliderChange = (
    key: keyof InferenceParams,
    value: number,
  ): void => {
    onChange({ ...params, [key]: value });
  };

  return (
    <div
      style={{
        padding: "1rem",
        borderRadius: "8px",
        backgroundColor: "#f5f5f5",
      }}
    >
      <h3 style={{ marginTop: 0, marginBottom: "1rem" }}>
        Inference Parameters
      </h3>

      {/* Confidence Threshold */}
      <div style={{ marginBottom: "1rem" }}>
        <label
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "0.25rem",
          }}
        >
          <span style={{ fontSize: "0.875rem", fontWeight: "bold" }}>
            Confidence Threshold: {params.confidenceThreshold.toFixed(2)}
          </span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={params.confidenceThreshold}
            onChange={(e) =>
              handleSliderChange("confidenceThreshold", Number(e.target.value))
            }
            style={{ width: "100%" }}
          />
          <span style={{ fontSize: "0.75rem", color: "#666" }}>
            Minimum confidence score for detections
          </span>
        </label>
      </div>

      {/* NMS Threshold */}
      <div style={{ marginBottom: "1rem" }}>
        <label
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "0.25rem",
          }}
        >
          <span style={{ fontSize: "0.875rem", fontWeight: "bold" }}>
            NMS Threshold: {params.nmsThreshold.toFixed(2)}
          </span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={params.nmsThreshold}
            onChange={(e) =>
              handleSliderChange("nmsThreshold", Number(e.target.value))
            }
            style={{ width: "100%" }}
          />
          <span style={{ fontSize: "0.75rem", color: "#666" }}>
            Non-maximum suppression overlap threshold
          </span>
        </label>
      </div>

      <div
        style={{
          marginTop: "1.5rem",
          padding: "0.75rem",
          backgroundColor: "#fff3cd",
          borderRadius: "4px",
          fontSize: "0.875rem",
          color: "#856404",
        }}
      >
        <strong>💡 Tip:</strong> Higher confidence threshold = fewer,
        more confident detections. Lower NMS threshold = fewer overlapping boxes.
      </div>
    </div>
  );
}
