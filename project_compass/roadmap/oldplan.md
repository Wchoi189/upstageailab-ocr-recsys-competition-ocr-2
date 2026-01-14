# Implementation Plan - Fix Refresh & Context Bundles

The goal is to fix the "Refresh" button in the MCP Visibility Dashboard and demonstrate the "Context Bundles" feature.

## User Review Required
> [!NOTE]
> The "Context Bundles" feature relies on the directory `AgentQMS/context_bundles`, which currently does not exist. I will create this directory and a sample bundle to demonstrate functionality.

## Proposed Changes

### Extension Logic (`src/`)

#### [MODIFY] [telemetryWatcher.ts](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/mcp-visibility-extension/src/telemetryWatcher.ts)
- Add `public async refresh()` method.
- This method will call [loadExisting()](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/mcp-visibility-extension/src/telemetryWatcher.ts#55-78) to re-read the telemetry file from disk, ensuring the dashboard reflects the latest state even if file watchers missed an event.

#### [MODIFY] [bundleProvider.ts](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/mcp-visibility-extension/src/bundleProvider.ts)
- Update `bundlesPath` to point to `AgentQMS/.agentqms/plugins/context_bundles`.
- Add `public refresh()` method that triggers `emitUpdate()`.

#### [MODIFY] [dashboardPanel.ts](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/mcp-visibility-extension/src/dashboardPanel.ts)
- Update the `refresh` message handler to call `telemetryWatcher.refresh()` and `bundleProvider.refresh()`.

## Verification Plan

### Automated
- Run `npm run compile` to ensure no type errors.

### Manual Verification
1. **Refresh Logic**:
   - Run the extension.
   - Click "Refresh" and ensure no errors occur.

2. **Context Bundles**:
   - Verify that real bundles (e.g., `ocr-debugging`, `pipeline-development`) appear in the "Context Bundles" section.
