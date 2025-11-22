import type React from "react";

/**
 * Spinner component props
 */
interface SpinnerProps {
  size?: "small" | "medium" | "large";
  color?: string;
}

/**
 * Simple CSS-based spinner component
 *
 * Provides a rotating spinner for loading states
 */
export function Spinner({
  size = "medium",
  color = "#007bff",
}: SpinnerProps): React.JSX.Element {
  const sizeMap = {
    small: "16px",
    medium: "32px",
    large: "48px",
  };

  const spinnerSize = sizeMap[size];

  return (
    <div
      style={{
        width: spinnerSize,
        height: spinnerSize,
        border: `3px solid ${color}33`,
        borderTop: `3px solid ${color}`,
        borderRadius: "50%",
        animation: "spin 0.8s linear infinite",
        display: "inline-block",
      }}
    >
      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
}
