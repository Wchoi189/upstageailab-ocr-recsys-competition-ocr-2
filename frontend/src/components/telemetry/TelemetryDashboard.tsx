import { useState, useEffect } from "react";
import { getMetricsSummary, type MetricsSummary } from "../../api/metrics";

/**
 * Telemetry Dashboard Component
 *
 * Displays real-time metrics for worker performance, cache hit rates,
 * and fallback routing decisions.
 */

interface TelemetryDashboardProps {
  timeRangeHours?: number;
  refreshIntervalMs?: number;
}

export function TelemetryDashboard({
  timeRangeHours = 1,
  refreshIntervalMs = 5000,
}: TelemetryDashboardProps): JSX.Element {
  const [summary, setSummary] = useState<MetricsSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async (): Promise<void> => {
      try {
        const data = await getMetricsSummary(timeRangeHours);
        setSummary(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch metrics");
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, refreshIntervalMs);

    return () => clearInterval(interval);
  }, [timeRangeHours, refreshIntervalMs]);

  if (loading && !summary) {
    return (
      <div style={{ padding: "2rem", textAlign: "center" }}>
        <p>Loading metrics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div
        style={{
          padding: "2rem",
          textAlign: "center",
          color: "#d32f2f",
          backgroundColor: "#ffebee",
          borderRadius: "8px",
        }}
      >
        <p>Error: {error}</p>
      </div>
    );
  }

  if (!summary) {
    return (
      <div style={{ padding: "2rem", textAlign: "center" }}>
        <p>No metrics available</p>
      </div>
    );
  }

  return (
    <div style={{ padding: "1.5rem" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "1.5rem",
        }}
      >
        <h2 style={{ margin: 0 }}>Telemetry Dashboard</h2>
        <span style={{ fontSize: "0.875rem", color: "#666" }}>
          Last {timeRangeHours}h | Auto-refresh: {refreshIntervalMs / 1000}s
        </span>
      </div>

      {/* Metrics Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "1rem",
          marginBottom: "1.5rem",
        }}
      >
        {/* Total Tasks */}
        <MetricCard
          title="Total Tasks"
          value={summary.total_tasks.toString()}
          description="Tasks queued in last hour"
          color="#1976d2"
        />

        {/* Worker Queue Depth */}
        <MetricCard
          title="Queue Depth"
          value={summary.worker_queue_depth.toString()}
          description="Tasks currently in queue"
          color={summary.worker_queue_depth > 5 ? "#d32f2f" : "#388e3c"}
          warning={
            summary.worker_queue_depth > 5
              ? "Queue depth exceeds target (<5)"
              : undefined
          }
        />

        {/* Cache Hit Rate */}
        <MetricCard
          title="Cache Hit Rate"
          value={`${(summary.cache_hit_rate * 100).toFixed(1)}%`}
          description="Percentage of cache hits"
          color="#7b1fa2"
        />
      </div>

      {/* Performance by Task Type */}
      <div style={{ marginBottom: "1.5rem" }}>
        <h3 style={{ marginBottom: "0.75rem", fontSize: "1rem" }}>
          Average Duration by Task Type
        </h3>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: "0.75rem",
          }}
        >
          {Object.entries(summary.avg_duration_ms).map(([taskType, avgMs]) => {
            const isSlowTask = taskType === "rembg" || taskType === "inference";
            const threshold = isSlowTask ? 400 : 100;
            const isWarning = avgMs > threshold;

            return (
              <div
                key={taskType}
                style={{
                  padding: "0.75rem",
                  border: `1px solid ${isWarning ? "#ff9800" : "#e0e0e0"}`,
                  borderRadius: "8px",
                  backgroundColor: isWarning ? "#fff3e0" : "#fafafa",
                }}
              >
                <div
                  style={{
                    fontSize: "0.75rem",
                    textTransform: "uppercase",
                    color: "#666",
                    marginBottom: "0.25rem",
                  }}
                >
                  {taskType.replace(/_/g, " ")}
                </div>
                <div
                  style={{
                    fontSize: "1.25rem",
                    fontWeight: "bold",
                    color: isWarning ? "#e65100" : "#333",
                  }}
                >
                  {avgMs.toFixed(1)}ms
                </div>
                {isWarning && (
                  <div style={{ fontSize: "0.7rem", color: "#e65100" }}>
                    Exceeds {threshold}ms target
                  </div>
                )}
              </div>
            );
          })}
          {Object.keys(summary.avg_duration_ms).length === 0 && (
            <div style={{ gridColumn: "1 / -1", textAlign: "center", color: "#999" }}>
              No performance data yet
            </div>
          )}
        </div>
      </div>

      {/* Task Status Breakdown */}
      <div style={{ marginBottom: "1.5rem" }}>
        <h3 style={{ marginBottom: "0.75rem", fontSize: "1rem" }}>
          Task Status Breakdown
        </h3>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "0.5rem",
          }}
        >
          {Object.entries(summary.tasks_by_status).map(([status, count]) => {
            const statusColors: Record<string, string> = {
              task_queued: "#2196f3",
              task_started: "#ff9800",
              task_completed: "#4caf50",
              task_failed: "#f44336",
              task_cancelled: "#9e9e9e",
            };

            return (
              <div
                key={status}
                style={{
                  padding: "0.5rem 0.75rem",
                  backgroundColor: statusColors[status] || "#757575",
                  color: "white",
                  borderRadius: "16px",
                  fontSize: "0.875rem",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                }}
              >
                <span>{status.replace("task_", "")}</span>
                <span
                  style={{
                    backgroundColor: "rgba(255, 255, 255, 0.3)",
                    padding: "0.125rem 0.5rem",
                    borderRadius: "12px",
                    fontWeight: "bold",
                  }}
                >
                  {count}
                </span>
              </div>
            );
          })}
          {Object.keys(summary.tasks_by_status).length === 0 && (
            <div style={{ color: "#999" }}>No task events yet</div>
          )}
        </div>
      </div>

      {/* Fallback Routing */}
      <div>
        <h3 style={{ marginBottom: "0.75rem", fontSize: "1rem" }}>
          Backend Fallback Rate
        </h3>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: "0.75rem",
          }}
        >
          {Object.entries(summary.fallback_rate).map(([operation, rate]) => (
            <div
              key={operation}
              style={{
                padding: "0.75rem",
                border: "1px solid #e0e0e0",
                borderRadius: "8px",
                backgroundColor: "#fafafa",
              }}
            >
              <div
                style={{
                  fontSize: "0.75rem",
                  textTransform: "uppercase",
                  color: "#666",
                  marginBottom: "0.25rem",
                }}
              >
                {operation}
              </div>
              <div style={{ fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
                {(rate * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: "0.7rem", color: "#666" }}>
                routed to backend
              </div>
            </div>
          ))}
          {Object.keys(summary.fallback_rate).length === 0 && (
            <div style={{ gridColumn: "1 / -1", textAlign: "center", color: "#999" }}>
              No fallback data yet
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface MetricCardProps {
  title: string;
  value: string;
  description?: string;
  color?: string;
  warning?: string;
}

function MetricCard({
  title,
  value,
  description,
  color = "#1976d2",
  warning,
}: MetricCardProps): JSX.Element {
  return (
    <div
      style={{
        padding: "1.25rem",
        border: `2px solid ${color}`,
        borderRadius: "12px",
        backgroundColor: "white",
      }}
    >
      <div
        style={{
          fontSize: "0.875rem",
          fontWeight: "500",
          color,
          marginBottom: "0.5rem",
          textTransform: "uppercase",
          letterSpacing: "0.5px",
        }}
      >
        {title}
      </div>
      <div
        style={{
          fontSize: "2.5rem",
          fontWeight: "bold",
          color: "#333",
          marginBottom: "0.25rem",
        }}
      >
        {value}
      </div>
      {description && (
        <div style={{ fontSize: "0.8rem", color: "#666" }}>{description}</div>
      )}
      {warning && (
        <div
          style={{
            marginTop: "0.5rem",
            fontSize: "0.75rem",
            color: "#d32f2f",
            fontWeight: "500",
          }}
        >
          ⚠ {warning}
        </div>
      )}
    </div>
  );
}
