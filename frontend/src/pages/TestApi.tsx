import { useState } from "react";
import type React from "react";
import { getCommandSchemas } from "../api/commands";

export function TestApi(): React.JSX.Element {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<unknown | null>(null);

  async function handleTest(): Promise<void> {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const schemas = await getCommandSchemas();
      setData(schemas);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error occurred");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: "2rem" }}>
      <h1>Test API Connection</h1>
      <p>Test connectivity to FastAPI backend at http://127.0.0.1:8000</p>

      <button
        onClick={handleTest}
        disabled={loading}
        style={{
          padding: "0.75rem 1.25rem",
          marginTop: "1rem",
          cursor: loading ? "wait" : "pointer",
          border: "none",
          borderRadius: "999px",
          fontSize: "1rem",
          fontWeight: 600,
          color: "#ffffff",
          backgroundColor: loading ? "#475569" : "#2563eb",
          boxShadow: loading ? "none" : "0 10px 22px rgba(37, 99, 235, 0.35)",
          opacity: loading ? 0.8 : 1,
          transition: "background-color 0.2s ease, transform 0.2s ease",
        }}
      >
        {loading ? "Testing..." : "Test /api/commands/schemas"}
      </button>

      {error && (
        <div
          style={{
            marginTop: "1rem",
            padding: "1rem",
            backgroundColor: "#fee",
            border: "1px solid #c00",
            borderRadius: "4px",
          }}
        >
          <strong>Error:</strong> {error}
        </div>
      )}

      {data && (
        <div
          style={{
            marginTop: "1rem",
            padding: "1rem",
            backgroundColor: "#efe",
            border: "1px solid #0c0",
            borderRadius: "4px",
          }}
        >
          <strong>Success! Received data:</strong>
          <pre style={{ marginTop: "0.5rem", overflow: "auto" }}>
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
