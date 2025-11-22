import type React from "react";
import { TelemetryDashboard } from "../components/telemetry/TelemetryDashboard";

/**
 * Telemetry Page
 *
 * Displays real-time performance metrics and worker statistics
 */

export function TelemetryPage(): React.JSX.Element {
  return (
    <div style={{ maxWidth: "1400px", margin: "0 auto" }}>
      <TelemetryDashboard timeRangeHours={1} refreshIntervalMs={5000} />
    </div>
  );
}
