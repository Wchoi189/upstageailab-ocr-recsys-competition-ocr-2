/**
 * Command display component with copy functionality.
 * Shows generated CLI command in a code block.
 */

import { useMemo, useState } from "react";
import type React from "react";

export interface CommandDisplayProps {
  command: string;
  validationError?: string;
  overrides?: string[];
  constantOverrides?: string[];
}

export function CommandDisplay(props: CommandDisplayProps): React.JSX.Element {
  const { command, validationError, overrides, constantOverrides } = props;
  const [copied, setCopied] = useState(false);
  const [downloaded, setDownloaded] = useState(false);

  const formattedCommand = useMemo(() => {
    return (
      formatCommandForLinux(command, constantOverrides, overrides) ||
      command ||
      ""
    );
  }, [command, constantOverrides, overrides]);

  const commandText = formattedCommand || command || "";

  const handleCopy = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(commandText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy command:", err);
    }
  };

  const handleDownload = (): void => {
    try {
      const blob = new Blob([commandText], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "command.sh";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setDownloaded(true);
      setTimeout(() => setDownloaded(false), 2000);
    } catch (err) {
      console.error("Failed to download command:", err);
    }
  };

  const isValid = !validationError && command;

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
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <h3 style={{ margin: 0, fontSize: "1rem", fontWeight: 600 }}>
            Generated Command
          </h3>
          {command && (
            <span
              style={{
                padding: "0.25rem 0.75rem",
                borderRadius: "9999px",
                fontSize: "0.75rem",
                fontWeight: 600,
                backgroundColor: isValid ? "#d1fae5" : "#fee2e2",
                color: isValid ? "#065f46" : "#991b1b",
              }}
            >
              {isValid ? "✓ Valid" : "⚠ Invalid"}
            </span>
          )}
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button
            onClick={handleCopy}
            disabled={!command}
            style={{
              padding: "0.5rem 1rem",
              backgroundColor: copied ? "#10b981" : command ? "#3b82f6" : "#d1d5db",
              color: "white",
              border: "none",
              borderRadius: "0.375rem",
              fontSize: "0.875rem",
              cursor: command ? "pointer" : "not-allowed",
              opacity: command ? 1 : 0.6,
            }}
          >
            {copied ? "Copied!" : "Copy"}
          </button>
          <button
            onClick={handleDownload}
            disabled={!command}
            style={{
              padding: "0.5rem 1rem",
              backgroundColor: downloaded ? "#10b981" : command ? "#6b7280" : "#d1d5db",
              color: "white",
              border: "none",
              borderRadius: "0.375rem",
              fontSize: "0.875rem",
              cursor: command ? "pointer" : "not-allowed",
              opacity: command ? 1 : 0.6,
            }}
          >
            {downloaded ? "Downloaded!" : "Download"}
          </button>
        </div>
      </div>

      <pre
        style={{
          backgroundColor: "#1f2937",
          color: "#f9fafb",
          padding: "1rem",
          borderRadius: "0.5rem",
          overflow: "auto",
          overflowX: "hidden",
          overflowY: "auto",
          fontSize: "0.875rem",
          lineHeight: "1.5",
          margin: 0,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          overflowWrap: "break-word",
          minHeight: "150px",
          maxHeight: "600px",
          border: "1px solid #374151",
          width: "100%",
          boxSizing: "border-box",
          textAlign: "left",
        }}
      >
        <code style={{
          fontFamily: "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          overflowWrap: "break-word",
          display: "block",
        }}>
          {commandText || "# Configure options above to generate command"}
        </code>
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

function splitCommandTokens(command: string): string[] {
  const tokenRegex = /(?:[^\s"']+|"[^"]*"|'[^']*')+/g;
  return command.match(tokenRegex) ?? [];
}

function formatCommandForLinux(
  command: string,
  constantOverrides?: string[],
  overrides?: string[]
): string {
  const rawCommand = command?.trim();
  if (!rawCommand) {
    return "";
  }

  const tokens = splitCommandTokens(rawCommand);
  if (tokens.length === 0) {
    return rawCommand;
  }

  let continuationSegments = [
    ...(constantOverrides ?? []),
    ...(overrides ?? []),
  ]
    .map((segment) => segment?.trim())
    .filter((segment): segment is string => Boolean(segment));

  let baseTokens: string[] | null = null;

  if (continuationSegments.length === 0) {
    const inferred = inferCommandSegments(tokens);
    if (!inferred) {
      return rawCommand;
    }
    baseTokens = inferred.baseTokens;
    continuationSegments = inferred.continuationSegments;
  } else {
    if (continuationSegments.length >= tokens.length) {
      return rawCommand;
    }
    baseTokens = tokens.slice(0, tokens.length - continuationSegments.length);
  }

  if (!baseTokens || baseTokens.length === 0 || continuationSegments.length === 0) {
    return rawCommand;
  }

  const lines: string[] = [];
  lines.push(`${baseTokens.join(" ")} \\`);

  continuationSegments.forEach((segment, index) => {
    const isLast = index === continuationSegments.length - 1;
    lines.push(`  ${segment}${isLast ? "" : " \\"}`);
  });

  return lines.join("\n");
}

function inferCommandSegments(tokens: string[]): {
  baseTokens: string[];
  continuationSegments: string[];
} | null {
  if (tokens.length < 5) {
    return null;
  }

  const splitIndex = tokens.findIndex((token, index) => {
    if (index < 4) {
      return false;
    }
    return token.includes("=") || token.startsWith("--");
  });

  if (splitIndex === -1) {
    return null;
  }

  const baseTokens = tokens.slice(0, splitIndex);
  const continuationSegments = tokens.slice(splitIndex);

  if (baseTokens.length === 0 || continuationSegments.length === 0) {
    return null;
  }

  return { baseTokens, continuationSegments };
}

export interface CommandDiffViewerProps {
  before: string;
  after: string;
}

export function CommandDiffViewer(props: CommandDiffViewerProps): React.JSX.Element {
  const { before, after } = props;

  const formattedBefore = formatCommandForLinux(before);
  const formattedAfter = formatCommandForLinux(after);

  return (
    <div style={{ marginTop: "2rem" }}>
      <h3 style={{ marginBottom: "1rem", fontSize: "1rem", fontWeight: 600 }}>
        Command Diff
      </h3>

      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
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
              overflowX: "hidden",
              overflowY: "auto",
              fontSize: "0.875rem",
              lineHeight: "1.5",
              margin: 0,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              overflowWrap: "break-word",
              minHeight: "80px",
              maxHeight: "200px",
              border: "1px solid #fecaca",
              width: "100%",
              boxSizing: "border-box",
            }}
          >
            <code style={{
              fontFamily: "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              overflowWrap: "break-word",
              display: "block",
            }}>
              {formattedBefore || before || "N/A"}
            </code>
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
              overflowX: "hidden",
              overflowY: "auto",
              fontSize: "0.875rem",
              lineHeight: "1.5",
              margin: 0,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              overflowWrap: "break-word",
              minHeight: "80px",
              maxHeight: "200px",
              border: "1px solid #bbf7d0",
              width: "100%",
              boxSizing: "border-box",
            }}
          >
            <code style={{
              fontFamily: "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              overflowWrap: "break-word",
              display: "block",
            }}>
              {formattedAfter || after || "N/A"}
            </code>
          </pre>
        </div>
      </div>
    </div>
  );
}
