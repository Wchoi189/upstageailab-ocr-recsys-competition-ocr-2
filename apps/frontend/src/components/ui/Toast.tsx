import { useEffect } from "react";
import type React from "react";

/**
 * Toast notification types
 */
export type ToastType = "success" | "error" | "info" | "warning";

/**
 * Toast notification props
 */
interface ToastProps {
  message: string;
  type?: ToastType;
  duration?: number;
  onClose: () => void;
}

/**
 * Simple toast notification component
 *
 * Displays temporary notifications with auto-dismiss
 */
export function Toast({
  message,
  type = "info",
  duration = 4000,
  onClose,
}: ToastProps): React.JSX.Element {
  useEffect(() => {
    const timer = setTimeout(onClose, duration);
    return () => clearTimeout(timer);
  }, [duration, onClose]);

  const colors = {
    success: { bg: "#d4edda", border: "#c3e6cb", text: "#155724" },
    error: { bg: "#f8d7da", border: "#f5c6cb", text: "#721c24" },
    warning: { bg: "#fff3cd", border: "#ffeeba", text: "#856404" },
    info: { bg: "#d1ecf1", border: "#bee5eb", text: "#0c5460" },
  };

  const color = colors[type];

  return (
    <div
      style={{
        position: "fixed",
        top: "1rem",
        right: "1rem",
        backgroundColor: color.bg,
        border: `1px solid ${color.border}`,
        color: color.text,
        padding: "1rem 1.5rem",
        borderRadius: "4px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
        maxWidth: "400px",
        zIndex: 9999,
        display: "flex",
        alignItems: "center",
        gap: "1rem",
        animation: "slideIn 0.3s ease-out",
      }}
    >
      <span style={{ flex: 1 }}>{message}</span>
      <button
        onClick={onClose}
        style={{
          background: "none",
          border: "none",
          color: color.text,
          cursor: "pointer",
          fontSize: "1.25rem",
          lineHeight: 1,
          padding: 0,
        }}
        aria-label="Close"
      >
        ×
      </button>
      <style>
        {`
          @keyframes slideIn {
            from {
              transform: translateX(100%);
              opacity: 0;
            }
            to {
              transform: translateX(0);
              opacity: 1;
            }
          }
        `}
      </style>
    </div>
  );
}
