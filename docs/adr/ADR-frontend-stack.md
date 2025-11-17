---
title: "ADR: Playground Frontend Stack"
status: "accepted"
date: "2025-11-17"
---

# ADR: Playground Frontend Stack

## Context

- Streamlit-based Command Builder, Evaluation Viewer, and Inference apps no longer meet latency and UX targets.
- Unified App (Preprocessing, Inference, Comparison) is incomplete and cannot be easily extended to Albumentations/Upstage-style experiences.
- We must support unlimited web workers for two days, hybrid rembg processing, and future layout/text-recognition modules.
- Backend service stubs now exist in `services/playground_api`, so the new frontend can be stateless.

## Decision

1. **Framework & Tooling**
   - Adopt **Vite + React 19 + TypeScript** for the SPA shell.
   - Use **Zustand** for lightweight global state and **TanStack Query** for API orchestration (playground API + worker responses).
   - Use **Vanilla Extract** (or CSS Modules) with design tokens defined in `docs/ui/design-system.md`.
2. **Routing Model**
   - Leverage **React Router v7** with file-based routes under `apps/playground/src/routes`.
   - Support both “Unified Shell” (tabbed pages) and “Micro-App” deployments through module federation hosts so Command Builder can be deployed standalone when needed.
3. **Build & Dev Experience**
   - Dev server script `run_spa.py` will proxy `/api/**` to FastAPI (`services/playground_api`).
   - Unit tests: Vitest + React Testing Library; e2e: Playwright.
4. **State/Data Contracts**
   - Shared schema package `packages/contracts` generated from Pydantic models exposed by FastAPI.
   - Workers communicate via Comlink RPC with typed message envelopes (`WorkerTask`, `WorkerResult`).

## Alternatives Considered

| Option | Pros | Cons |
| --- | --- | --- |
| Next.js (App Router) | Built-in SSR, ergonomics | Overkill for single-page playground, heavier bundler, makes module-federated micro-apps harder |
| SvelteKit | Smaller bundles, reactivity | Worker ecosystem + existing React component reuse would need rewrites |
| Keep Streamlit | Minimal refactor | Cannot hit <100 ms preview targets; worker orchestration impossible |

## Consequences

- Need to scaffold a new `apps/playground` workspace with Vite config, linking to shared ESLint/Prettier settings.
- Requires adding FastAPI + Pydantic dependencies (already added in `pyproject.toml`).
- We must maintain the contracts package whenever backend models change.
- Browser bundle now must ship worker pool + WASM assets; ensure CDN caching strategy.

## Follow-up Tasks

- Implement base SPA skeleton with routing, Zustand store, and API client pointing at `/api`.
- Generate TypeScript types from FastAPI’s OpenAPI schema during CI.
- Document local dev workflow in `run_spa.py` (Track E).


