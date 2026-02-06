# [Compass:Lock] Core Directive 02: Artifact Purity

## The Staging Constraint

**THE ONLY valid artifact location**: `pulse_staging/artifacts/`

Files created ANYWHERE else are violations and will be ignored by export.

## Zero-Narrative Policy

**Max 20 words** for any status update or note.

| ❌ Banned                                                    | ✅ Required                   |
| ----------------------------------------------------------- | ---------------------------- |
| "We discussed that the implementation could potentially..." | "Implemented X in Y.py"      |
| Multi-paragraph summaries                                   | Bullet points with file refs |
| Future tense speculation                                    | Past tense facts             |

## Artifact Lifecycle

| Tier             | Status          | Location                   | Persistence                 |
| ---------------- | --------------- | -------------------------- | --------------------------- |
| **Transitional** | Working draft   | `pulse_staging/artifacts/` | Deleted on pulse completion |
| **Archived**     | Historic record | `history/{pulse_id}/`      | Permanent, read-only        |

## Registration Requirement

Before pulse-export, ALL files in staging MUST be registered via `pulse-sync`.

Unregistered files = **BLOCKED export**.

## The Scratchpad Exception

If you need verbose reasoning, use `pulse_staging/artifacts/scratchpad.md`.

This file is **auto-deleted** on export and never archived.
