# Implementation Plans

Active implementation plans and development roadmaps.

**Last Updated**: 2026-01-08 03:59:43
**Total Artifacts**: 22

## Active (16)

- [Implementation Plan: AgentQMS Architecture Consolidation](2025-11-29_1800_implementation_plan_architecture-consolidation.md) (📅 2025-11-29 18:00 (KST), 📄 implementation_plan) - > **Context**: Converging `.agentqms/`, `.ai-instructions/`, and `AgentQMS/` into a single `AgentQMS
- [Phase 1: Core Recognition Implementation](2025-12-25_0410_implementation_plan_phase1-core-recognition.md) (📅 2025-12-25 04:10 (KST), 📄 implementation_plan) - Replace the current `StubRecognizer` with a production-ready `PaddleOCRRecognizer` backend using Pad
- [Overnight Code Quality Resolution (Ruff & Mypy)](2025-12-25_1639_implementation_plan_code-quality-overnight.md) (📅 2025-12-25 16:39 (KST), 📄 implementation_plan) - This plan outlines the systematic resolution of linting and type errors across the project, optimize
- [Overnight Autonomous Operations Master Plan](2025-12-25_1651_implementation_plan_overnight-autonomous-ops.md) (📅 2025-12-25 16:51 (KST), 📄 implementation_plan) - This plan outlines a multi-role autonomous strategy to accelerate the OCR project overnight. It is d
- [Implementation Plan - Survey Follow-up & Cleanup](2026-01-05_1750_implementation_plan_data-merge.md) (📅 2026-01-05 17:50 (KST), 📄 implementation_plan) - **Goal:** Address "Immediate" and "Short-term" findings from the AI Workflow Efficiency Survey to pr
- [Master Architecture Refactoring Plan "Project Polaris](2026-01-05_1750_implementation_plan_master-refactoring.md) (📅 2026-01-05 17:50 (KST), 📄 implementation_plan) - **Date:** 2026-01-05 17:50 (KST)
- [Implementation Plan - Github Sync Extension](2026-01-05_1854_implementation_plan_github-sync-extension.md) (📅 2026-01-05 18:54 (KST), 📄 implementation_plan) - This plan outlines the steps to extend the `sync_github_projects.py` script to support comprehensive
- [MCP Sharing Implementation Plan](2026-01-05_2049_implementation_plan_mcp-sharing-plan.md) (📅 2026-01-05 20:49 (KST), 📄 implementation_plan) - The goal is to centralize MCP (Model Context Protocol) server definitions into a shared directory wi
- [Refactor Phase 1 & 2: Core Foundation & Data Pivot](2026-01-07_0054_implementation_plan_refactor-phase1-core-data.md) (📅 2026-01-07 00:54 (KST), 📄 implementation_plan) - Establish the 'Feature-First' architecture foundation by creating `ocr/core`, `configs/_foundation` 
- [Refactor Phase 3: Recognition Feature Migration](2026-01-07_0054_implementation_plan_refactor-phase3-recognition.md) (📅 2026-01-07 00:54 (KST), 📄 implementation_plan) - Fully isolate the **Recognition** domain by moving PARSeq models, encoders, and decoders into `ocr/r
- [Refactor Phase 4: Detection & KIE Feature Migration](2026-01-07_0054_implementation_plan_refactor-phase4-detection-kie.md) (📅 2026-01-07 00:54 (KST), 📄 implementation_plan) - Isolate the **Detection** and **KIE** domains into their own feature packages.
- [Implementation Plan for Codebase Analysis and Debugging Tools](2026-01-07_0423_implementation_plan_codebase-analysis-tools-implementation.md) (📅 2026-01-07 04:23 (KST), 📄 implementation_plan) - Implement a suite of tools for systematic analysis and debugging of a large, complex codebase. Focus
- [AST Analyzer Phase 1: Low Risk Features](2026-01-07_0448_implementation_plan_ast-analyzer-phase1-low-risk.md) (📅 2026-01-07 04:48 (KST), 📄 implementation_plan) - Extend `agent-debug-toolkit` with foundational analysis capabilities for AI agent codebase navigatio
- [AST Analyzer Phase 2: Medium Risk Features](2026-01-07_0448_implementation_plan_ast-analyzer-phase2-medium-risk.md) (📅 2026-01-07 04:48 (KST), 📄 implementation_plan) - Implement advanced analyzers requiring deeper AST traversal, scope tracking, and heuristic matching.
- [AST Analyzer Phase 2.5: Hydra Config Resolver](2026-01-07_1918_implementation_plan_ast-analyzer-phase2-5-config-resolver.md) (📅 2026-01-07 19:18 (KST), 📄 implementation_plan) - Implement the **Hydra Config Resolver** tool ("Phase 2.5") to statically map Hydra configuration `_t
- [Consolidate Session Protocol Documentation](2026-01-08_0302_implementation_plan_consolidate-session-docs.md) (📅 2026-01-08 03:02 (KST), 📄 implementation_plan) - Plan to consolidate session protocol documentation into existing AI-optimized files.

## Completed (1)

- [Model Head Signature Fix - LSP Violation Resolution](2025-12-27_0610_implementation_plan_model-head-signature-fix.md) (📅 2025-12-27 06:10 (KST), 📄 implementation_plan) - **Type:** Liskov Substitution Principle (LSP) Violation - HIGH RISK

## Other (5)

- [Phase 2-3: Pipeline Integration + VLM + Optimization](2025-12-25_0410_implementation_plan_phase2-pipeline-integration.md) (📅 2025-12-25 04:10 (KST), 📄 implementation_plan) - Integrate the recognition, layout, and extraction modules into a unified end-to-end pipeline with hy
- [Implementation Plan - GitHub Sync Entrypoints](2026-01-05_1900_implementation_plan_github-sync-entrypoints.md) (📅 2026-01-05 18:54 (KST), 📄 implementation_plan) - This plan outlines the steps to expose the new GitHub Sync features (`--init`, `--roadmap`) via VS C
- [AST Analyzer Phase 3: Conceptual Plan for Automated Refactoring](2026-01-07_0448_implementation_plan_ast-analyzer-phase3-conceptual.md) (📅 2026-01-07 04:48 (KST), 📄 implementation_plan) - > "I have a very big code base... bloat, inconsistencies, obsolete, deprecated, broken, outdated, su
- [Cleanup & Organization Implementation Plan](2026-01-08_0340_implementation_plan_cleanup-organization-plan.md) (📅 2026-01-08, 📄 implementation_plan) - Implement the "Future Recommendations" (1-3) from the Refactor Audit Survey to reduce bloat, improve
- [2026-01-08_0359_implementation_plan_hydra-config-restructuring](2026-01-08_0359_implementation_plan_hydra-config-restructuring.md) (📅 , 📄 ) - **Objective**: Map all Hydra entry points and config access patterns before making changes.

## Summary

| Status | Count |
|--------|-------|
| Active | 16 |
| Completed | 1 |
| Other | 5 |

---

*This index is automatically generated. Do not edit manually.*
