import { useState, useEffect } from "react";
import type React from "react";
import type { CheckpointWithMetadata } from "../../api/inference";
import {
  listCheckpoints,
  filterCheckpoints,
  filterByArchitecture,
  filterByBackbone,
  getUniqueArchitectures,
  getUniqueBackbones,
} from "../../api/inference";

/**
 * Props for CheckpointPicker component
 */
interface CheckpointPickerProps {
  selectedCheckpoint: CheckpointWithMetadata | null;
  onSelect: (checkpoint: CheckpointWithMetadata | null) => void;
}

/**
 * Checkpoint catalog with search and filter
 *
 * Displays available checkpoints with metadata and filtering options
 */
export function CheckpointPicker({
  selectedCheckpoint,
  onSelect,
}: CheckpointPickerProps): React.JSX.Element {
  const [allCheckpoints, setAllCheckpoints] = useState<
    CheckpointWithMetadata[]
  >([]);
  const [filteredCheckpoints, setFilteredCheckpoints] = useState<
    CheckpointWithMetadata[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [searchTerm, setSearchTerm] = useState("");
  const [architectureFilter, setArchitectureFilter] = useState<string | null>(
    null,
  );
  const [backboneFilter, setBackboneFilter] = useState<string | null>(null);

  // Load checkpoints on mount
  useEffect(() => {
    const loadCheckpoints = async (): Promise<void> => {
      try {
        setLoading(true);
        const checkpoints = await listCheckpoints(100);
        setAllCheckpoints(checkpoints);
        setFilteredCheckpoints(checkpoints);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load");
        setAllCheckpoints([]);
        setFilteredCheckpoints([]);
      } finally {
        setLoading(false);
      }
    };

    loadCheckpoints();
  }, []);

  // Apply filters when search/filter changes
  useEffect(() => {
    let result = allCheckpoints;
    result = filterCheckpoints(result, searchTerm);
    result = filterByArchitecture(result, architectureFilter);
    result = filterByBackbone(result, backboneFilter);
    setFilteredCheckpoints(result);
  }, [allCheckpoints, searchTerm, architectureFilter, backboneFilter]);

  const uniqueArchitectures = getUniqueArchitectures(allCheckpoints);
  const uniqueBackbones = getUniqueBackbones(allCheckpoints);

  if (loading) {
    return (
      <div style={{ padding: "1rem", textAlign: "center", color: "#666" }}>
        Loading checkpoints...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: "1rem", color: "red" }}>Error: {error}</div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      {/* Search and Filters */}
      <div
        style={{
          padding: "1rem",
          backgroundColor: "#f5f5f5",
          borderRadius: "8px",
        }}
      >
        <h3 style={{ marginTop: 0, marginBottom: "1rem" }}>
          Checkpoint Selection
        </h3>

        {/* Search */}
        <div style={{ marginBottom: "0.75rem" }}>
          <input
            type="text"
            placeholder="Search by name, experiment, architecture..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{
              width: "100%",
              padding: "0.5rem",
              fontSize: "0.875rem",
              border: "1px solid #ddd",
              borderRadius: "4px",
            }}
          />
        </div>

        {/* Filter Row */}
        <div
          style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}
        >
          {/* Architecture Filter */}
          {uniqueArchitectures.length > 0 && (
            <select
              value={architectureFilter || ""}
              onChange={(e) =>
                setArchitectureFilter(e.target.value || null)
              }
              style={{
                color: "#213547",
                backgroundColor: "white",
                padding: "0.5rem",
                fontSize: "0.875rem",
                border: "1px solid #ddd",
                borderRadius: "4px",
              }}
            >
              <option value="" style={{ color: "#213547", backgroundColor: "white" }}>
                All Architectures
              </option>
              {uniqueArchitectures.map((arch) => (
                <option
                  key={arch}
                  value={arch}
                  style={{ color: "#213547", backgroundColor: "white" }}
                >
                  {arch}
                </option>
              ))}
            </select>
          )}

          {/* Backbone Filter */}
          {uniqueBackbones.length > 0 && (
            <select
              value={backboneFilter || ""}
              onChange={(e) => setBackboneFilter(e.target.value || null)}
              style={{
                color: "#213547",
                backgroundColor: "white",
                padding: "0.5rem",
                fontSize: "0.875rem",
                border: "1px solid #ddd",
                borderRadius: "4px",
              }}
            >
              <option value="" style={{ color: "#213547", backgroundColor: "white" }}>
                All Backbones
              </option>
              {uniqueBackbones.map((backbone) => (
                <option
                  key={backbone}
                  value={backbone}
                  style={{ color: "#213547", backgroundColor: "white" }}
                >
                  {backbone}
                </option>
              ))}
            </select>
          )}

          {/* Clear Filters */}
          {(searchTerm || architectureFilter || backboneFilter) && (
            <button
              onClick={() => {
                setSearchTerm("");
                setArchitectureFilter(null);
                setBackboneFilter(null);
              }}
              style={{
                padding: "0.5rem 1rem",
                fontSize: "0.875rem",
                backgroundColor: "#6c757d",
                color: "white",
                border: "none",
                borderRadius: "4px",
                cursor: "pointer",
              }}
            >
              Clear Filters
            </button>
          )}
        </div>

        <div style={{ marginTop: "0.5rem", fontSize: "0.875rem", color: "#666" }}>
          Showing {filteredCheckpoints.length} of {allCheckpoints.length}{" "}
          checkpoints
        </div>
      </div>

      {/* Checkpoint List */}
      <div
        style={{
          maxHeight: "400px",
          overflowY: "auto",
          border: "1px solid #ddd",
          borderRadius: "8px",
        }}
      >
        {filteredCheckpoints.length === 0 ? (
          <div
            style={{
              padding: "2rem",
              textAlign: "center",
              color: "#666",
            }}
          >
            No checkpoints found
          </div>
        ) : (
          <div>
            {filteredCheckpoints.map((ckpt) => (
              <div
                key={ckpt.checkpoint_path}
                onClick={() => onSelect(ckpt)}
                style={{
                  padding: "1rem",
                  borderBottom: "1px solid #eee",
                  cursor: "pointer",
                  backgroundColor:
                    selectedCheckpoint?.checkpoint_path ===
                    ckpt.checkpoint_path
                      ? "#e3f2fd"
                      : "white",
                  transition: "background-color 0.2s",
                }}
                onMouseEnter={(e) => {
                  if (
                    selectedCheckpoint?.checkpoint_path !==
                    ckpt.checkpoint_path
                  ) {
                    e.currentTarget.style.backgroundColor = "#f5f5f5";
                  }
                }}
                onMouseLeave={(e) => {
                  if (
                    selectedCheckpoint?.checkpoint_path !==
                    ckpt.checkpoint_path
                  ) {
                    e.currentTarget.style.backgroundColor = "white";
                  }
                }}
              >
                <div
                  style={{
                    fontWeight: "bold",
                    marginBottom: "0.25rem",
                  }}
                >
                  {ckpt.display_name}
                </div>
                <div
                  style={{
                    fontSize: "0.875rem",
                    color: "#666",
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "1rem",
                  }}
                >
                  {ckpt.exp_name && <span>Exp: {ckpt.exp_name}</span>}
                  {ckpt.architecture && <span>Arch: {ckpt.architecture}</span>}
                  {ckpt.backbone && <span>Backbone: {ckpt.backbone}</span>}
                  <span>{ckpt.size_mb} MB</span>
                  <span>
                    {ckpt.modifiedDate.toLocaleDateString()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
