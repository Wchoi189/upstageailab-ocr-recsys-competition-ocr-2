# ADS Schemas Specification

**Tier**: Schemas
**Scope**: Agentic Document Standard (ADS) Definitions.

## 1. ADS Version 2.0
**Core Requirement**: All artifacts must declare `ads_version: '2.0'`.

### Structure
1.  **Frontmatter**: See `tier1-contracts/validation.spec.md`.
2.  **Body**: Markdown content.
3.  **Metadata**: Embedded constraints (id, tier, priority).

### Legacy Support (V1)
*   **V1 Artifacts**: Missing `ads_version`.
*   **Support**: Deprecated but readable. New artifacts **must** be V2.
