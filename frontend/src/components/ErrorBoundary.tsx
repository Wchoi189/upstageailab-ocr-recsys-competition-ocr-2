/**
 * Error Boundary component to catch React errors and display them
 */

import { Component, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: unknown): void {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div
          style={{
            padding: "2rem",
            maxWidth: "800px",
            margin: "0 auto",
            backgroundColor: "#1a1a1a",
            color: "#fff",
            borderRadius: "8px",
            marginTop: "2rem",
          }}
        >
          <h2 style={{ color: "#f87171", marginTop: 0 }}>
            ⚠️ Something went wrong
          </h2>
          <p style={{ color: "#d1d5db" }}>
            {this.state.error?.message || "An unexpected error occurred"}
          </p>
          <details style={{ marginTop: "1rem" }}>
            <summary style={{ cursor: "pointer", color: "#9ca3af" }}>
              Error Details
            </summary>
            <pre
              style={{
                backgroundColor: "#0f0f0f",
                padding: "1rem",
                borderRadius: "4px",
                overflow: "auto",
                marginTop: "0.5rem",
                fontSize: "0.875rem",
              }}
            >
              {this.state.error?.stack || "No stack trace available"}
            </pre>
          </details>
          <button
            onClick={() => {
              this.setState({ hasError: false, error: null });
              window.location.reload();
            }}
            style={{
              marginTop: "1rem",
              padding: "0.5rem 1rem",
              backgroundColor: "#3b82f6",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            Reload Page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

