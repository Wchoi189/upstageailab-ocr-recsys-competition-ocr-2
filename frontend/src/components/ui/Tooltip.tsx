/**
 * Tooltip component for displaying additional information.
 * Used for architecture-aware metadata tooltips.
 */

import { useState, type ReactNode } from "react";

export interface TooltipProps {
  content: string | ReactNode;
  children: ReactNode;
  position?: "top" | "bottom" | "left" | "right";
}

export function Tooltip(props: TooltipProps): JSX.Element {
  const { content, children, position = "top" } = props;
  const [visible, setVisible] = useState(false);

  const positionStyles: Record<string, React.CSSProperties> = {
    top: {
      bottom: "100%",
      left: "50%",
      transform: "translateX(-50%)",
      marginBottom: "0.5rem",
    },
    bottom: {
      top: "100%",
      left: "50%",
      transform: "translateX(-50%)",
      marginTop: "0.5rem",
    },
    left: {
      right: "100%",
      top: "50%",
      transform: "translateY(-50%)",
      marginRight: "0.5rem",
    },
    right: {
      left: "100%",
      top: "50%",
      transform: "translateY(-50%)",
      marginLeft: "0.5rem",
    },
  };

  return (
    <div
      style={{ position: "relative", display: "inline-block" }}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      {visible && (
        <div
          style={{
            position: "absolute",
            zIndex: 50,
            ...positionStyles[position],
          }}
        >
          <div
            style={{
              backgroundColor: "#1f2937",
              color: "white",
              padding: "0.5rem 0.75rem",
              borderRadius: "0.375rem",
              fontSize: "0.875rem",
              lineHeight: "1.5",
              maxWidth: "300px",
              boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
              whiteSpace: "pre-wrap",
            }}
          >
            {content}
          </div>
        </div>
      )}
    </div>
  );
}

export interface InfoIconProps {
  tooltip: string | ReactNode;
  position?: "top" | "bottom" | "left" | "right";
}

export function InfoIcon(props: InfoIconProps): JSX.Element {
  const { tooltip, position = "top" } = props;

  return (
    <Tooltip content={tooltip} position={position}>
      <svg
        width="16"
        height="16"
        viewBox="0 0 16 16"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{
          display: "inline-block",
          verticalAlign: "middle",
          cursor: "help",
          color: "#9ca3af",
        }}
      >
        <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
        <path
          d="M8 11.5V8M8 5.5H8.005"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </Tooltip>
  );
}
