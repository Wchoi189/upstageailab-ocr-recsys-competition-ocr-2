# Abandoned: Unified OCR App

**Status**: ⛔ Abandoned (2024-Q4)
**Reason**: Failed migration attempt, replaced by dual Next.js applications

---

## Background

This directory contains an abandoned attempt to consolidate all OCR functionality (inference, command building, preprocessing, comparison) into a single Streamlit application.

The project was started as a migration path from the legacy Streamlit apps but was **abandoned in favor of a dual-app strategy**:

1. **OCR Inference Console** ([`apps/ocr-inference-console/`](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/)) - Vite+React, inference-focused
2. **Playground Console** ([`apps/playground-console/`](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/playground-console/)) - Next.js, full playground features

---

## ⚠️ DO NOT USE THIS CODE

This implementation is:
- ❌ Incomplete (~40% implemented)
- ❌ Unmaintained (no development since Q4 2024)
- ❌ Superseded by modern Next.js applications
- ❌ Not referenced in current documentation

If you need similar functionality, use:
- **For inference**: `apps/ocr-inference-console`
- **For full playground**: `apps/playground-console`
- **For legacy (temporary)**: `ui/apps/inference`, `ui/apps/command_builder`

---

## Archive Date

Moved to archive: 2025-12-12

## Related Documentation

- [System Overview](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/architecture/00_system_overview.md)
- [Architecture Audit](file:///home/vscode/.gemini/antigravity/brain/e233fabb-0950-4377-903d-e30dbc71cd13/ARCHITECTURE_AND_DOCS_AUDIT.md)
