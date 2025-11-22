import type React from "react";

/**
 * Preprocessing parameters
 */
export interface PreprocessingParams {
  autocontrast: boolean;
  blur: boolean;
  blurKernelSize: number;
  resize: boolean;
  resizeWidth: number;
  resizeHeight: number;
  rembg: boolean;
}

/**
 * Props for ParameterControls component
 */
interface ParameterControlsProps {
  params: PreprocessingParams;
  onChange: (params: PreprocessingParams) => void;
}

/**
 * Parameter control tray with sliders and toggles
 *
 * Provides UI controls for image preprocessing parameters
 */
export function ParameterControls({
  params,
  onChange,
}: ParameterControlsProps): React.JSX.Element {
  const handleToggle = (key: keyof PreprocessingParams): void => {
    onChange({ ...params, [key]: !params[key] });
  };

  const handleSliderChange = (
    key: keyof PreprocessingParams,
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
        maxWidth: "400px",
      }}
    >
      <h3 style={{ marginTop: 0, marginBottom: "1rem", color: "#213547" }}>
        Preprocessing Parameters
      </h3>

      {/* Auto Contrast Toggle */}
      <div style={{ marginBottom: "1rem" }}>
        <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <input
            type="checkbox"
            checked={params.autocontrast}
            onChange={() => handleToggle("autocontrast")}
          />
          <span style={{ color: "#213547" }}>Auto Contrast</span>
        </label>
      </div>

      {/* Gaussian Blur Controls */}
      <div style={{ marginBottom: "1rem" }}>
        <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <input
            type="checkbox"
            checked={params.blur}
            onChange={() => handleToggle("blur")}
          />
          <span style={{ color: "#213547" }}>Gaussian Blur</span>
        </label>
        {params.blur && (
          <div style={{ marginTop: "0.5rem", marginLeft: "1.5rem" }}>
            <label
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "0.25rem",
              }}
            >
              <span style={{ fontSize: "0.875rem", color: "#666" }}>
                Kernel Size: {params.blurKernelSize}
              </span>
              <input
                type="range"
                min="3"
                max="15"
                step="2"
                value={params.blurKernelSize}
                onChange={(e) =>
                  handleSliderChange("blurKernelSize", Number(e.target.value))
                }
                style={{ width: "100%" }}
              />
            </label>
          </div>
        )}
      </div>

      {/* Resize Controls */}
      <div style={{ marginBottom: "1rem" }}>
        <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <input
            type="checkbox"
            checked={params.resize}
            onChange={() => handleToggle("resize")}
          />
          <span style={{ color: "#213547" }}>Resize</span>
        </label>
        {params.resize && (
          <div
            style={{
              marginTop: "0.5rem",
              marginLeft: "1.5rem",
              display: "flex",
              flexDirection: "column",
              gap: "0.5rem",
            }}
          >
            <label style={{ display: "flex", flexDirection: "column" }}>
              <span style={{ fontSize: "0.875rem", color: "#666" }}>
                Width: {params.resizeWidth}px
              </span>
              <input
                type="range"
                min="128"
                max="2048"
                step="64"
                value={params.resizeWidth}
                onChange={(e) =>
                  handleSliderChange("resizeWidth", Number(e.target.value))
                }
                style={{ width: "100%" }}
              />
            </label>
            <label style={{ display: "flex", flexDirection: "column" }}>
              <span style={{ fontSize: "0.875rem", color: "#666" }}>
                Height: {params.resizeHeight}px
              </span>
              <input
                type="range"
                min="128"
                max="2048"
                step="64"
                value={params.resizeHeight}
                onChange={(e) =>
                  handleSliderChange("resizeHeight", Number(e.target.value))
                }
                style={{ width: "100%" }}
              />
            </label>
          </div>
        )}
      </div>

      {/* Background Removal (rembg) */}
      <div style={{ marginBottom: "1rem" }}>
        <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <input
            type="checkbox"
            checked={params.rembg}
            onChange={() => handleToggle("rembg")}
          />
          <span style={{ color: "#213547" }}>Background Removal (rembg)</span>
        </label>
        {params.rembg && (
          <div
            style={{
              marginTop: "0.5rem",
              marginLeft: "1.5rem",
              fontSize: "0.875rem",
              color: "#666",
            }}
          >
            <p style={{ margin: 0 }}>
              Automatically routes to backend for large images (&gt;2048px)
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
