/**
 * Command display component with copy functionality.
 * Shows generated CLI command in a code block.
 */

import { useState } from "react";

export interface CommandDisplayProps {
  command: string;
  validationError?: string;
}

export function CommandDisplay(props: CommandDisplayProps): JSX.Element {
  const { command, validationError } = props;
  const [copied, setCopied] = useState(false);

  const handleCopy = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy command:", err);
    }
  };

  return (
    <div style={{ marginTop: "2rem" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "0.5rem",
        }}
      >
        <h3 style={{ margin: 0, fontSize: "1rem", fontWeight: 600 }}>
          Generated Command
        </h3>
        <button
          onClick={handleCopy}
          style={{
            padding: "0.5rem 1rem",
            backgroundColor: copied ? "#10b981" : "#3b82f6",
            color: "white",
            border: "none",
            borderRadius: "0.375rem",
            fontSize: "0.875rem",
            cursor: "pointer",
          }}
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>

      <pre
        style={{
          backgroundColor: "#1f2937",
          color: "#f9fafb",
          padding: "1rem",
          borderRadius: "0.5rem",
          overflow: "auto",
          fontSize: "0.875rem",
          lineHeight: "1.5",
          margin: 0,
        }}
      >
        <code>{command || "# Configure options above to generate command"}</code>
      </pre>

      {validationError && (
        <div
          style={{
            marginTop: "0.5rem",
            padding: "0.75rem",
            backgroundColor: "#fef2f2",
            border: "1px solid #fecaca",
            borderRadius: "0.375rem",
            color: "#991b1b",
            fontSize: "0.875rem",
          }}
        >
          <strong>Validation Error:</strong> {validationError}
        </div>
      )}
    </div>
  );
}

export interface CommandDiffViewerProps {
  before: string;
  after: string;
}

export function CommandDiffViewer(props: CommandDiffViewerProps): JSX.Element {
  const { before, after } = props;

  return (
    <div style={{ marginTop: "2rem" }}>
      <h3 style={{ marginBottom: "1rem", fontSize: "1rem", fontWeight: 600 }}>
        Command Diff
      </h3>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
        <div>
          <h4 style={{ fontSize: "0.875rem", fontWeight: 500, marginBottom: "0.5rem" }}>
            Before
          </h4>
          <pre
            style={{
              backgroundColor: "#fef2f2",
              color: "#991b1b",
              padding: "1rem",
              borderRadius: "0.5rem",
              overflow: "auto",
              fontSize: "0.75rem",
              lineHeight: "1.5",
              margin: 0,
            }}
          >
            <code>{before || "N/A"}</code>
          </pre>
        </div>

        <div>
          <h4 style={{ fontSize: "0.875rem", fontWeight: 500, marginBottom: "0.5rem" }}>
            After
          </h4>
          <pre
            style={{
              backgroundColor: "#f0fdf4",
              color: "#166534",
              padding: "1rem",
              borderRadius: "0.5rem",
              overflow: "auto",
              fontSize: "0.75rem",
              lineHeight: "1.5",
              margin: 0,
            }}
          >
            <code>{after || "N/A"}</code>
          </pre>
        </div>
      </div>
    </div>
  );
}
