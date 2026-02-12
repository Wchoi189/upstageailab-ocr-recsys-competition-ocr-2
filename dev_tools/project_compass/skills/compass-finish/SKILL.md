---
name: compass-finish
description: Export pulse with pre-checks and audit. Use when work is complete and ready to archive.
disable-model-invocation: true
---

# Compass Finish - Pulse Export with Audit

## Pre-Export Audit

Before exporting, check:

### 1. List All Staging Files
```bash
ls -la /workspaces/dev_tools/project_compass/pulse_staging/artifacts/
```

### 2. Get Registered Artifacts
```bash
uv run compass pulse-status
```

### 3. Identify Unregistered Files
Compare lists and identify files NOT in manifest.

## Handle Unregistered Files

If unregistered files exist, prompt user:
```
⚠️  Unregistered files found:
- [file-1]
- [file-2]

Actions:
1. Register these files (type /compass-register [filename])
2. Delete temporary files (which ones?)
3. Cancel export

Choose action:
```

## Export Execution

Only proceed if NO unregistered files exist or user confirms deletion.

### Execute Export
```bash
uv run compass pulse-export
```

### Success Output
```
✅ Pulse exported successfully
📦 Location: /workspaces/dev_tools/project_compass/history/[milestone]/[timestamp]_[pulse-id]/
📊 Artifacts archived: [count]
🔄 Ready for next pulse (use /compass-start)
```

## Error Handling

If export blocked:
1. Show exact error from CLI command
2. List specific files causing block
3. Provide corrective actions
4. DO NOT manually move files or edit JSON

## Post-Export Cleanup

Remind user:
```
📌 Staging cleared. Directives reset.
To start new work: /compass-start [pulse-id] "[objective]" [milestone]
```
