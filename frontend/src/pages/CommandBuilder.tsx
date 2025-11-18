/**
 * Command Builder page with training/test/predict tabs.
 * Schema-driven form generator for CLI command building.
 */

import { useState, useEffect } from "react";
import type {
  SchemaId,
  CommandSchema,
  FormValues,
  BuildCommandResponse,
  Recommendation,
} from "@/types/schema";
import {
  getSchemaDetails,
  buildCommand,
  getCommandRecommendations,
} from "@/api/commands";
import { SchemaForm } from "@/components/forms/SchemaForm";
import {
  CommandDisplay,
  CommandDiffViewer,
} from "@/components/commands/CommandDisplay";
import { RecommendationsGrid } from "@/components/recommendations/RecommendationCard";

export function CommandBuilder(): JSX.Element {
  const [activeTab, setActiveTab] = useState<SchemaId>("train");
  const [schema, setSchema] = useState<CommandSchema | null>(null);
  const [values, setValues] = useState<FormValues>({});
  const [commandResult, setCommandResult] = useState<BuildCommandResponse | null>(
    null
  );
  const [previousCommand, setPreviousCommand] = useState<string>("");
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [selectedRecommendationId, setSelectedRecommendationId] = useState<
    string | undefined
  >();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showRecommendations, setShowRecommendations] = useState(true);

  // Load schema when tab changes
  useEffect(() => {
    loadSchema(activeTab);
    loadRecommendations();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  // Reload recommendations when architecture changes
  useEffect(() => {
    const architecture = values.architecture as string | undefined;
    if (architecture) {
      loadRecommendations(architecture);
    }
  }, [values.architecture]);

  // Build command when values change
  useEffect(() => {
    if (Object.keys(values).length > 0) {
      buildCommandFromValues();
    }
  }, [values]);

  const loadSchema = async (schemaId: SchemaId): Promise<void> => {
    try {
      setLoading(true);
      setError(null);
      const schemaData = await getSchemaDetails(schemaId);
      setSchema(schemaData);
      setValues({});
      setCommandResult(null);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "Failed to load schema. Make sure the API server is running at http://127.0.0.1:8000";
      setError(errorMessage);
      console.error("Error loading schema:", err);
      setSchema(null);
    } finally {
      setLoading(false);
    }
  };

  const loadRecommendations = async (architecture?: string): Promise<void> => {
    try {
      const recs = await getCommandRecommendations(architecture);
      setRecommendations(recs);
    } catch (err) {
      console.error("Error loading recommendations:", err);
    }
  };

  const handleRecommendationSelect = (rec: Recommendation): void => {
    setSelectedRecommendationId(rec.id);
    // Merge recommendation parameters into form values
    setValues((prev) => ({ ...prev, ...rec.parameters }));
  };

  const buildCommandFromValues = async (): Promise<void> => {
    try {
      // Save previous command before building new one
      if (commandResult?.command) {
        setPreviousCommand(commandResult.command);
      }

      const result = await buildCommand({
        schema_id: activeTab,
        values,
        append_model_suffix: true,
      });
      setCommandResult(result);
    } catch (err) {
      console.error("Error building command:", err);
    }
  };

  return (
    <div style={{ padding: "2rem", maxWidth: "1200px", margin: "0 auto" }}>
      <h1 style={{ marginBottom: "0.5rem" }}>Command Builder</h1>
      <p style={{ color: "#6b7280", marginBottom: "2rem" }}>
        Build metadata-aware training, testing, and prediction commands
      </p>

      {/* Tab Navigation */}
      <div
        style={{
          borderBottom: "2px solid #e5e7eb",
          marginBottom: "2rem",
          display: "flex",
          gap: "1rem",
        }}
      >
        <TabButton
          label="Training"
          active={activeTab === "train"}
          onClick={() => setActiveTab("train")}
        />
        <TabButton
          label="Testing"
          active={activeTab === "test"}
          onClick={() => setActiveTab("test")}
        />
        <TabButton
          label="Prediction"
          active={activeTab === "predict"}
          onClick={() => setActiveTab("predict")}
        />
      </div>

      {/* Content */}
      {loading && (
        <div style={{ padding: "2rem", textAlign: "center" }}>
          <p>Loading schema...</p>
          <p style={{ fontSize: "0.875rem", color: "#6b7280", marginTop: "0.5rem" }}>
            If this takes too long, make sure the API server is running at http://127.0.0.1:8000
          </p>
        </div>
      )}

      {error && (
        <div
          style={{
            padding: "1.5rem",
            backgroundColor: "#fef2f2",
            border: "2px solid #fecaca",
            borderRadius: "0.5rem",
            color: "#991b1b",
            marginTop: "1rem",
          }}
        >
          <h3 style={{ marginTop: 0, marginBottom: "0.5rem" }}>⚠️ Error Loading Schema</h3>
          <p style={{ margin: 0 }}>{error}</p>
          <p style={{ marginTop: "1rem", fontSize: "0.875rem", color: "#7f1d1d" }}>
            <strong>To fix this:</strong>
            <br />
            1. Make sure the API server is running: <code>uv run python run_spa.py --api-only</code>
            <br />
            2. Check that the server is accessible at <code>http://127.0.0.1:8000</code>
            <br />
            3. Check the browser console (F12) for more details
          </p>
        </div>
      )}

      {schema && !loading && (
        <div>
          <h2 style={{ fontSize: "1.25rem", marginBottom: "1.5rem" }}>
            {schema.title}
          </h2>

          {/* Recommendations Panel */}
          {activeTab === "train" && recommendations.length > 0 && (
            <div style={{ marginBottom: "2rem" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: "1rem",
                }}
              >
                <h3
                  style={{
                    margin: 0,
                    fontSize: "1rem",
                    fontWeight: 600,
                    color: "#374151",
                  }}
                >
                  Recommended Configurations
                </h3>
                <button
                  onClick={() => setShowRecommendations(!showRecommendations)}
                  style={{
                    padding: "0.25rem 0.75rem",
                    fontSize: "0.75rem",
                    border: "1px solid #d1d5db",
                    borderRadius: "0.375rem",
                    backgroundColor: "white",
                    cursor: "pointer",
                    color: "#6b7280",
                  }}
                >
                  {showRecommendations ? "Hide" : "Show"}
                </button>
              </div>

              {showRecommendations && (
                <RecommendationsGrid
                  recommendations={recommendations}
                  onSelect={handleRecommendationSelect}
                  selectedId={selectedRecommendationId}
                />
              )}
            </div>
          )}

          <SchemaForm schema={schema} values={values} onChange={setValues} />

          {commandResult && (
            <>
              <CommandDisplay
                command={commandResult.command}
                validationError={commandResult.validation_error}
              />

              {previousCommand && previousCommand !== commandResult.command && (
                <CommandDiffViewer
                  before={previousCommand}
                  after={commandResult.command}
                />
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

interface TabButtonProps {
  label: string;
  active: boolean;
  onClick: () => void;
}

function TabButton(props: TabButtonProps): JSX.Element {
  const { label, active, onClick } = props;

  return (
    <button
      onClick={onClick}
      style={{
        padding: "0.75rem 1.5rem",
        backgroundColor: "transparent",
        border: "none",
        borderBottom: active ? "2px solid #3b82f6" : "2px solid transparent",
        color: active ? "#3b82f6" : "#6b7280",
        fontWeight: active ? 600 : 400,
        fontSize: "0.875rem",
        cursor: "pointer",
        marginBottom: "-2px",
      }}
    >
      {label}
    </button>
  );
}
