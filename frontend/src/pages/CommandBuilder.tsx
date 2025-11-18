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
} from "@/types/schema";
import { getSchemaDetails, buildCommand } from "@/api/commands";
import { SchemaForm } from "@/components/forms/SchemaForm";
import {
  CommandDisplay,
  CommandDiffViewer,
} from "@/components/commands/CommandDisplay";

export function CommandBuilder(): JSX.Element {
  const [activeTab, setActiveTab] = useState<SchemaId>("train");
  const [schema, setSchema] = useState<CommandSchema | null>(null);
  const [values, setValues] = useState<FormValues>({});
  const [commandResult, setCommandResult] = useState<BuildCommandResponse | null>(
    null
  );
  const [previousCommand, setPreviousCommand] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load schema when tab changes
  useEffect(() => {
    loadSchema(activeTab);
  }, [activeTab]);

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
      setError(err instanceof Error ? err.message : "Failed to load schema");
      console.error("Error loading schema:", err);
    } finally {
      setLoading(false);
    }
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
      {loading && <p>Loading schema...</p>}

      {error && (
        <div
          style={{
            padding: "1rem",
            backgroundColor: "#fef2f2",
            border: "1px solid #fecaca",
            borderRadius: "0.5rem",
            color: "#991b1b",
          }}
        >
          {error}
        </div>
      )}

      {schema && !loading && (
        <div>
          <h2 style={{ fontSize: "1.25rem", marginBottom: "1.5rem" }}>
            {schema.title}
          </h2>

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
