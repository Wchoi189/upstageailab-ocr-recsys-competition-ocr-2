/**
 * Execution log console component (placeholder).
 * Shows streaming output when commands are executed.
 */

import { useState } from "react";
import type React from "react";

export interface LogEntry {
  timestamp: Date;
  level: "info" | "error" | "warning" | "success";
  message: string;
}

export interface ExecutionLogProps {
  logs: LogEntry[];
  isRunning?: boolean;
}

export function ExecutionLog(props: ExecutionLogProps): React.JSX.Element {
  const { logs, isRunning = false } = props;
  const [expanded, setExpanded] = useState(true);

  const getLevelColor = (level: LogEntry["level"]): string => {
    switch (level) {
      case "error":
        return "#ef4444";
      case "warning":
        return "#f59e0b";
      case "success":
        return "#10b981";
      default:
        return "#d1d5db";
    }
  };

  const formatTime = (timestamp: Date): string => {
    return timestamp.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
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
          Execution Log {isRunning && <span style={{ color: "#3b82f6" }}>●</span>}
        </h3>
        <button
          onClick={() => setExpanded(!expanded)}
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
          {expanded ? "Collapse" : "Expand"}
        </button>
      </div>

      {expanded && (
        <div
          style={{
            backgroundColor: "#1f2937",
            borderRadius: "0.5rem",
            padding: "1rem",
            maxHeight: "300px",
            overflowY: "auto",
            fontFamily: "monospace",
            fontSize: "0.875rem",
          }}
        >
          {logs.length === 0 ? (
            <div style={{ color: "#9ca3af", fontStyle: "italic" }}>
              No execution logs yet. Run a command to see output here.
            </div>
          ) : (
            logs.map((log, index) => (
              <div
                key={index}
                style={{
                  display: "flex",
                  gap: "0.75rem",
                  marginBottom: "0.5rem",
                  color: "#f9fafb",
                }}
              >
                <span style={{ color: "#9ca3af", flexShrink: 0 }}>
                  [{formatTime(log.timestamp)}]
                </span>
                <span
                  style={{
                    color: getLevelColor(log.level),
                    flexShrink: 0,
                    fontWeight: 600,
                    textTransform: "uppercase",
                    fontSize: "0.75rem",
                  }}
                >
                  {log.level}
                </span>
                <span style={{ wordBreak: "break-word" }}>{log.message}</span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
