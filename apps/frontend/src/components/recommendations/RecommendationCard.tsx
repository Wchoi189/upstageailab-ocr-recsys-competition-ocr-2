/**
 * Recommendation card component for displaying use case recommendations.
 * Shows architecture, title, description, and parameter preview.
 */

import type React from "react";
import type { Recommendation } from "@/types/schema";

export interface RecommendationCardProps {
  recommendation: Recommendation;
  onSelect: (recommendation: Recommendation) => void;
  selected?: boolean;
}

export function RecommendationCard(props: RecommendationCardProps): React.JSX.Element {
  const { recommendation, onSelect, selected = false } = props;

  const handleClick = (): void => {
    onSelect(recommendation);
  };

  const parameterCount = Object.keys(recommendation.parameters || {}).length;

  return (
    <div
      onClick={handleClick}
      style={{
        border: selected ? "2px solid #3b82f6" : "1px solid #d1d5db",
        borderRadius: "0.5rem",
        padding: "1rem",
        cursor: "pointer",
        backgroundColor: selected ? "#eff6ff" : "white",
        transition: "all 0.2s",
        boxShadow: selected
          ? "0 4px 6px -1px rgba(59, 130, 246, 0.1)"
          : "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
      }}
      onMouseEnter={(e) => {
        if (!selected) {
          e.currentTarget.style.borderColor = "#9ca3af";
          e.currentTarget.style.boxShadow = "0 4px 6px -1px rgba(0, 0, 0, 0.1)";
        }
      }}
      onMouseLeave={(e) => {
        if (!selected) {
          e.currentTarget.style.borderColor = "#d1d5db";
          e.currentTarget.style.boxShadow = "0 1px 2px 0 rgba(0, 0, 0, 0.05)";
        }
      }}
    >
      {/* Architecture Badge */}
      {recommendation.architecture && (
        <div style={{ marginBottom: "0.5rem" }}>
          <span
            style={{
              display: "inline-block",
              padding: "0.25rem 0.75rem",
              backgroundColor: "#e0e7ff",
              color: "#4338ca",
              borderRadius: "9999px",
              fontSize: "0.75rem",
              fontWeight: 600,
              textTransform: "uppercase",
            }}
          >
            {recommendation.architecture}
          </span>
        </div>
      )}

      {/* Title */}
      <h3
        style={{
          margin: "0 0 0.5rem 0",
          fontSize: "1rem",
          fontWeight: 600,
          color: selected ? "#1e40af" : "#111827",
        }}
      >
        {recommendation.title}
      </h3>

      {/* Description */}
      <p
        style={{
          margin: "0 0 0.75rem 0",
          fontSize: "0.875rem",
          color: "#6b7280",
          lineHeight: "1.5",
        }}
      >
        {recommendation.description}
      </p>

      {/* Parameter Count */}
      {parameterCount > 0 && (
        <div
          style={{
            fontSize: "0.75rem",
            color: "#9ca3af",
            display: "flex",
            alignItems: "center",
            gap: "0.25rem",
          }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M8 4V8L10.5 9.5"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <circle
              cx="8"
              cy="8"
              r="6"
              stroke="currentColor"
              strokeWidth="1.5"
            />
          </svg>
          <span>
            {parameterCount} parameter{parameterCount === 1 ? "" : "s"} preset
          </span>
        </div>
      )}
    </div>
  );
}

export interface RecommendationsGridProps {
  recommendations: Recommendation[];
  onSelect: (recommendation: Recommendation) => void;
  selectedId?: string;
}

export function RecommendationsGrid(
  props: RecommendationsGridProps
): React.JSX.Element {
  const { recommendations, onSelect, selectedId } = props;

  if (recommendations.length === 0) {
    return (
      <div
        style={{
          padding: "2rem",
          textAlign: "center",
          color: "#6b7280",
          fontSize: "0.875rem",
        }}
      >
        No recommendations available for this configuration.
      </div>
    );
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
        gap: "1rem",
        marginTop: "1rem",
      }}
    >
      {recommendations.map((rec) => (
        <RecommendationCard
          key={rec.id}
          recommendation={rec}
          onSelect={onSelect}
          selected={rec.id === selectedId}
        />
      ))}
    </div>
  );
}
