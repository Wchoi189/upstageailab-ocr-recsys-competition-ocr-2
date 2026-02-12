# Project Specification

## Scope
The scope includes the `project_compass` toolchain, specifically:
- `src/core.py` for pulse management logic.
- `src/pulse_exporter.py` for export and snapshot logic.
- `src/state_schema.py` for manifest updates.
- Spec-Kit tools integration for auto-registration.
- Context bundle system (likely in `src/rule_injector.py` or separate module).

## Requirements
1. **Auto-Reconciliation**: The system must automatically identify and remove artifact references from the pulse manifest if the corresponding files are deleted from the filesystem, preventing "orphaned artifact" errors during export.
2. **Snapshot System**: A new `pulse-snapshot` workflow is required to allow users to save a checkpoint of their current work (state + artifacts) without closing the active pulse. This enables multi-session work and safe experimentation.
3. **Auto-Registration**: Artifacts created via Spec-Kit tools (constitution, specify, plan, tasks) must be automatically registered with the active pulse, eliminating the need for manual `pulse-sync` steps.
4. **Enhanced Error Handling**: Error messages, especially regarding active pulses or export failures, must provide actionable options (e.g., "Run X to fix") rather than just stating the problem.
5. **Context Bundle Improvements**: Implement hierarchical bundles (bundles within bundles) and lazy loading to improve performance and relevance of context suggestions.
6. **Artifact Lifecycle Clarity**: Introduce commands to view (`pulse-ls`) and manage (`pulse-unstage`) artifact states explicitly.

## Status
- Created: 2026-02-02T20:07:02.759335
- Tool: Project Compass v2
- Status: Draft
