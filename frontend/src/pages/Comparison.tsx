import { useState, useEffect } from "react";
import type {
  ComparisonPreset,
  ComparisonRequest,
  ComparisonResponse,
} from "../api/evaluation";
import {
  listComparisonPresets,
  queueComparison,
} from "../api/evaluation";

/**
 * Comparison Studio page
 *
 * Configure and run model comparison with parameter sweeps
 */
export function Comparison(): JSX.Element {
  const [presets, setPresets] = useState<ComparisonPreset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<ComparisonPreset | null>(
    null,
  );
  const [formData, setFormData] = useState<ComparisonRequest>({
    preset_id: "",
    model_a_path: "",
    model_b_path: "",
    ground_truth_path: "",
    image_dir: "",
    extra_params: {},
  });
  const [result, setResult] = useState<ComparisonResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load presets on mount
  useEffect(() => {
    const loadPresets = async (): Promise<void> => {
      try {
        const presetList = await listComparisonPresets();
        setPresets(presetList);
        if (presetList.length > 0) {
          setSelectedPreset(presetList[0]);
          setFormData((prev) => ({ ...prev, preset_id: presetList[0].id }));
        }
      } catch (err) {
        console.error("Failed to load presets:", err);
      }
    };

    loadPresets();
  }, []);

  const handlePresetChange = (presetId: string): void => {
    const preset = presets.find((p) => p.id === presetId);
    if (preset) {
      setSelectedPreset(preset);
      setFormData({
        preset_id: preset.id,
        model_a_path: "",
        model_b_path: "",
        ground_truth_path: "",
        image_dir: "",
        extra_params: {},
      });
      setResult(null);
      setError(null);
    }
  };

  const handleInputChange = (
    field: keyof ComparisonRequest,
    value: string,
  ): void => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (): Promise<void> => {
    if (!selectedPreset) return;

    setLoading(true);
    setError(null);

    try {
      const response = await queueComparison(formData);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Comparison failed");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const isFieldRequired = (field: string): boolean => {
    return selectedPreset?.required_inputs.includes(field) || false;
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h1>Comparison Studio</h1>
      <p style={{ color: "#666", marginBottom: "2rem" }}>
        Configure parameter sweeps and compare model performance
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "400px 1fr",
          gap: "2rem",
          alignItems: "start",
        }}
      >
        {/* Left: Configuration */}
        <div>
          {/* Preset Selection */}
          <div
            style={{
              padding: "1rem",
              backgroundColor: "#f5f5f5",
              borderRadius: "8px",
              marginBottom: "1rem",
            }}
          >
            <h3 style={{ marginTop: 0, marginBottom: "1rem" }}>
              Select Preset
            </h3>
            <select
              value={selectedPreset?.id || ""}
              onChange={(e) => handlePresetChange(e.target.value)}
              style={{
                width: "100%",
                padding: "0.5rem",
                fontSize: "0.875rem",
                border: "1px solid #ddd",
                borderRadius: "4px",
                backgroundColor: "white",
              }}
            >
              {presets.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.label}
                </option>
              ))}
            </select>
            {selectedPreset?.description && (
              <p
                style={{
                  marginTop: "0.5rem",
                  fontSize: "0.875rem",
                  color: "#666",
                }}
              >
                {selectedPreset.description}
              </p>
            )}
          </div>

          {/* Parameter Configuration */}
          {selectedPreset && (
            <div
              style={{
                padding: "1rem",
                backgroundColor: "#f5f5f5",
                borderRadius: "8px",
              }}
            >
              <h3 style={{ marginTop: 0, marginBottom: "1rem" }}>
                Parameters
              </h3>

              {/* Model A Path */}
              <div style={{ marginBottom: "1rem" }}>
                <label
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.25rem",
                  }}
                >
                  <span style={{ fontSize: "0.875rem", fontWeight: "bold" }}>
                    Model A Path
                    {isFieldRequired("model_a_path") && (
                      <span style={{ color: "red" }}> *</span>
                    )}
                  </span>
                  <input
                    type="text"
                    value={formData.model_a_path || ""}
                    onChange={(e) =>
                      handleInputChange("model_a_path", e.target.value)
                    }
                    placeholder="outputs/exp_1/predictions.csv"
                    style={{
                      padding: "0.5rem",
                      fontSize: "0.875rem",
                      border: "1px solid #ddd",
                      borderRadius: "4px",
                    }}
                  />
                </label>
              </div>

              {/* Model B Path */}
              {isFieldRequired("model_b_path") && (
                <div style={{ marginBottom: "1rem" }}>
                  <label
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "0.25rem",
                    }}
                  >
                    <span style={{ fontSize: "0.875rem", fontWeight: "bold" }}>
                      Model B Path <span style={{ color: "red" }}>*</span>
                    </span>
                    <input
                      type="text"
                      value={formData.model_b_path || ""}
                      onChange={(e) =>
                        handleInputChange("model_b_path", e.target.value)
                      }
                      placeholder="outputs/exp_2/predictions.csv"
                      style={{
                        padding: "0.5rem",
                        fontSize: "0.875rem",
                        border: "1px solid #ddd",
                        borderRadius: "4px",
                      }}
                    />
                  </label>
                </div>
              )}

              {/* Ground Truth Path */}
              <div style={{ marginBottom: "1rem" }}>
                <label
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.25rem",
                  }}
                >
                  <span style={{ fontSize: "0.875rem", fontWeight: "bold" }}>
                    Ground Truth Path (optional)
                  </span>
                  <input
                    type="text"
                    value={formData.ground_truth_path || ""}
                    onChange={(e) =>
                      handleInputChange("ground_truth_path", e.target.value)
                    }
                    placeholder="data/labels/ground_truth.csv"
                    style={{
                      padding: "0.5rem",
                      fontSize: "0.875rem",
                      border: "1px solid #ddd",
                      borderRadius: "4px",
                    }}
                  />
                </label>
              </div>

              {/* Image Directory */}
              {isFieldRequired("image_dir") && (
                <div style={{ marginBottom: "1rem" }}>
                  <label
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "0.25rem",
                    }}
                  >
                    <span style={{ fontSize: "0.875rem", fontWeight: "bold" }}>
                      Image Directory <span style={{ color: "red" }}>*</span>
                    </span>
                    <input
                      type="text"
                      value={formData.image_dir || ""}
                      onChange={(e) =>
                        handleInputChange("image_dir", e.target.value)
                      }
                      placeholder="data/datasets/images/val"
                      style={{
                        padding: "0.5rem",
                        fontSize: "0.875rem",
                        border: "1px solid #ddd",
                        borderRadius: "4px",
                      }}
                    />
                  </label>
                </div>
              )}

              {/* Submit Button */}
              <button
                onClick={handleSubmit}
                disabled={loading}
                style={{
                  width: "100%",
                  padding: "0.75rem",
                  fontSize: "0.875rem",
                  fontWeight: "bold",
                  backgroundColor: loading ? "#ccc" : "#007bff",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: loading ? "not-allowed" : "pointer",
                }}
              >
                {loading ? "Running..." : "Run Comparison"}
              </button>
            </div>
          )}
        </div>

        {/* Right: Results Display */}
        <div
          style={{
            padding: "2rem",
            border: "1px solid #ddd",
            borderRadius: "8px",
            minHeight: "400px",
          }}
        >
          <h3 style={{ marginTop: 0, marginBottom: "1rem" }}>Results</h3>

          {error && (
            <div
              style={{
                padding: "1rem",
                backgroundColor: "#f8d7da",
                color: "#721c24",
                borderRadius: "4px",
                marginBottom: "1rem",
              }}
            >
              {error}
            </div>
          )}

          {result ? (
            <div>
              <div
                style={{
                  padding: "1rem",
                  backgroundColor: "#d4edda",
                  color: "#155724",
                  borderRadius: "4px",
                  marginBottom: "1rem",
                }}
              >
                <strong>{result.status.toUpperCase()}:</strong> {result.message}
              </div>

              {result.next_steps.length > 0 && (
                <div>
                  <h4>Next Steps:</h4>
                  <ul>
                    {result.next_steps.map((step, idx) => (
                      <li key={idx} style={{ marginBottom: "0.5rem" }}>
                        {step}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div
                style={{
                  marginTop: "2rem",
                  padding: "1.5rem",
                  backgroundColor: "#f5f5f5",
                  borderRadius: "4px",
                  textAlign: "center",
                  color: "#666",
                }}
              >
                <p style={{ marginBottom: "0.5rem" }}>
                  📊 Metrics visualization coming soon
                </p>
                <p style={{ fontSize: "0.875rem", margin: 0 }}>
                  Side-by-side comparison charts, tables, and diff metrics
                  will be displayed here once the backend pipeline is wired.
                </p>
              </div>
            </div>
          ) : (
            <div
              style={{
                padding: "2rem",
                textAlign: "center",
                color: "#666",
              }}
            >
              <p>Configure parameters and run comparison to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
