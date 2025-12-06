---
title: "Next.js Console Migration And Chakra Adoption Plan"
date: "2025-12-06 18:09 (KST)"
type: "implementation_plan"
category: "planning"
status: "active"
version: "1.0"
tags: ['implementation_plan', 'planning', 'documentation']
---







# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Next.js Console Migration and Chakra Adoption Plan**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Next.js Console Migration and Chakra Adoption Plan

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Phase 4 Complete - Blocked on Chakra UI v3 Migration
- **CURRENT STEP:** Phase 4 Complete (Tasks 4.1 & 4.2) – API Proxy, Session Context, Analytics & GTM Integration
- **LAST COMPLETED TASK:** Phase 4 (Tasks 4.1 & 4.2) – Implemented API proxy routes, session context, and analytics/GTM with consent management
- **NEXT TASK:** Complete Chakra UI v3 form component migration (FormControl → Field component) to enable production build
- **BLOCKER:** Chakra UI v3 compatibility - Phase 3 components use deprecated FormControl/FormLabel/FormHelperText that need migration to Field component

### Implementation Outline (Checklist)

#### **Phase 1: Workspace & Shell Foundations (Week 1)**
1. [x] **Task 1.1: Next.js Workspace Initialization**
   - [x] Bootstrap Next.js 15 App Router project under `/apps/playground-console`
   - [x] Configure npm workspaces + root scripts to manage SPA and console together
   - [x] Ensure TypeScript + ESLint baselines follow repo standards (Next.js strict TS, lint script wired via root package)

2. [x] **Task 1.2: CI & Dev Tooling**
   - [x] Add scripts to root package + Makefile for console dev/build/lint (short commands per coding standards)
   - [x] Update CI/Makefile to run lint+build for both SPA and Next.js
   - [ ] Address existing SPA lint failures surfaced by `npm run lint:spa` (tracked separately in legacy plan)

#### **Phase 2: Theming & Shell (Week 2)**
3. [x] **Task 2.1: Chakra Theme + Providers**
   - [x] Install Chakra UI + Emotion + Framer Motion + React Query dependencies
   - [x] Create shared theme (`src/theme/index.ts`) with brand/semantic tokens aligned to Upstage console look
   - [x] Wrap `app/layout.tsx` in `AppProviders` (Chakra + React Query + ColorModeScript per coding standards)

4. [x] **Task 2.2: Console Shell Components**
   - [x] Implement TopNav, SideNav, Banner, Notifications using Chakra primitives
   - [x] Add responsive layout skeleton with breadcrumb + CTA slots

#### **Phase 3: Feature Migration (Week 3)**
5. [x] **Task 3.1: Command Builder Migration**
   - [x] Scaffold Next.js route + Chakra UI for Command Builder with live API integration (schemas, build, recommendations)
   - [x] Create `packages/console-shared` workspace for shared types + visibility helpers (consumed by SPA + console)
   - [x] Re-implement SchemaForm/CommandDisplay/Diff/Recommendations in Chakra using shared data contracts and debounced builds
   - [x] Add API loading/error states, clipboard actions, and diff viewer parity within console shell

6. [x] **Task 3.2: Extract Playground Pages**
   - [x] Build Universal Extraction route with upload inputs, pipeline settings, and preview/JSON tabs
   - [x] Build Prebuilt Extraction route with business-card thumbnails, Preview/JSON panes, and template metadata callouts
   - [x] Note layout-recognition backlog item in implementation notes for future enhancements

#### **Phase 4: API Proxy, Auth, and Observability (Week 4)**
7. [x] **Task 4.1: API Proxy & Auth**
   - [x] Implement `/app/api` routes/server actions that proxy to FastAPI (GET schemas, GET details, POST build, GET recommendations)
   - [x] Create shared API proxy utility with error handling (`src/lib/api-proxy.ts`)
   - [x] Update client API calls to use `/api` routes instead of direct FastAPI calls
   - [x] Add FASTAPI_BASE_URL environment variable configuration
   - [x] Create SessionContext and useSession hook (placeholder for future auth integration)
   - [x] Integrate SessionProvider into app providers

8. [x] **Task 4.2: Analytics & Compliance**
   - [x] Install @next/third-parties for Next.js GTM integration
   - [x] Create GTMProvider with conditional loading based on consent
   - [x] Implement ConsentBanner using Chakra UI with localStorage persistence
   - [x] Create useConsent hook for managing cookie consent state
   - [x] Add analytics tracking utilities (`src/lib/analytics.ts`) with trackEvent, trackPageView, trackCommandBuild, trackImageUpload, trackInferenceRun
   - [x] Integrate GTMProvider and ConsentBanner into app layout
   - [x] Document instrumentation events with JSDoc comments

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Modular monorepo structure with shared packages
- [ ] API proxy alignment with existing FastAPI schemas
- [ ] Environment-driven configuration for endpoints/keys
- [ ] React Query + Zustand state management composed inside Next.js providers

### **Integration Points**
- [ ] Hook into FastAPI command/extraction endpoints via `/app/api`
- [ ] Reuse existing SchemaForm/CommandDisplay utilities
- [ ] Authentication provider (NextAuth/custom JWT) wired to backend
- [ ] Analytics SDK (PostHog/GTM) integrated

### **Quality Assurance**
- [ ] Component/unit tests for shell + key widgets (>80% coverage target)
- [ ] Integration tests for API proxy routes (Playwright/contract tests)
- [ ] Bundle/perf budget validation (<200kb initial JS for shell)
- [ ] Visual/UX regression tests for playground flows

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] Command Builder fully operational in Next.js console
- [ ] Universal & Prebuilt Extraction pages deliver previews + JSON output
- [ ] API proxy handles auth + inference calls reliably (<1% failure outside backend issues)
- [ ] Console shell mirrors Upstage UX (navigation, banners, notifications)

### **Technical Requirements**
- [ ] Type-safe shared packages with strict TS configs
- [ ] Server actions stay within resource limits (edge-friendly)
- [ ] Compatible with existing FastAPI deployment and auth model
- [ ] Linting, formatting, and CI checks enforced

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: MEDIUM
### **Active Mitigation Strategies**:
1. Incremental migration while SPA remains online
2. Shared component package to avoid duplicate logic
3. Early instrumentation of API proxy to catch latency/errors

### **Fallback Options**:
1. If Next.js blockers arise, embed SSR-like shell into Vite SPA temporarily
2. If Chakra gaps appear, mix in Mantine/Radix components for specific widgets
3. If API proxy is delayed, route client directly to FastAPI with feature flags

---

## 🔄 **Blueprint Update Protocol**

**Update Triggers:**
- Task completion (move to next task)
- Blocker encountered (document and propose solution)
- Technical discovery (update approach if needed)
- Quality gate failure (address issues before proceeding)

**Update Format:**
1. Update Progress Tracker (STATUS, CURRENT STEP, LAST COMPLETED TASK, NEXT TASK)
2. Mark completed items with [x]
3. Add any new discoveries or changes to approach
4. Update risk assessment if needed

---

## 🚀 **Immediate Next Action**

**TASK:** Fix Chakra UI v3 form component compatibility

**OBJECTIVE:** Migrate Phase 3 form components from Chakra v2 API to v3 API to enable production builds.

**BLOCKER DETAILS:**
Phase 3 components use Chakra v2 form APIs that don't exist in v3:
- `FormControl` → needs migration to `Field.Root`
- `FormLabel` → needs migration to `Field.Label`
- `FormHelperText` → needs migration to `Field.HelperText`
- `FormErrorMessage` → needs migration to `Field.ErrorText`

Affected files:
- `apps/playground-console/src/components/command-builder/SchemaForm.tsx` (primary blocker)
- Any other form components created in Phase 3

**APPROACH:**
1. Update SchemaForm.tsx to use Chakra v3 Field component API
2. Test form rendering and validation
3. Ensure TypeScript types are correct
4. Run production build to verify

**SUCCESS CRITERIA:**
- `npm run build` completes successfully
- All form fields render correctly
- Form validation works as expected

---

## ✅ **Phase 4 Completion Summary (2025-11-20)**

**Completed Tasks:**
- ✅ Task 4.1: API Proxy & Session Context
- ✅ Task 4.2: Analytics & GTM Integration
- ✅ Fixed partial Chakra UI v3 compatibility issues (Alert, Separator, theme, providers)

**Key Deliverables:**
1. **API Proxy Layer** - All FastAPI calls now proxied through Next.js `/app/api` routes
2. **Session Infrastructure** - SessionContext and useSession hook ready for auth integration
3. **Analytics Framework** - GTM integration with consent management and event tracking utilities
4. **Environment Configuration** - `.env.local` with FASTAPI_BASE_URL (gitignored)

**Committed Work:**
- Branch: `feature/nextjs-console-migration`
- Commit: `d6f3635` - "feat: Complete Next.js Console Phase 4 - API Proxy, Session Context & Analytics"
- Files changed: 22 files, +1454/-487 lines

**Outstanding Work:**
- Chakra UI v3 form component migration (blocker for production build)
- Push to remote (403 permission error - needs proper branch permissions)

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*

## Summary
Path 2 migrates the current Vite SPA to a Next.js App Router console that mirrors the Upstage playground, adds Chakra UI for shared theming, and establishes an inference-friendly backend-for-frontend. This plan scopes the first migration wave, highlights dependencies, and notes where further assessments are required.

## Goals & Non-Goals
- **Goals:** deliver a shared console shell, port the Command Builder playground, integrate Chakra UI theming, and prepare API proxy endpoints for OCR inference.
- **Non-Goals:** rewriting OCR models, replacing backend infra, or finalizing analytics/billing integrations (tracked separately).

## Current State
- Frontend runs as a React 19 + Vite SPA (`frontend/package.json`) with inline styling and no SSR, limiting SEO and cross-tool navigation.
- UI logic (Command Builder, schema forms, command diffing) is encapsulated in components that can be transplanted into Next.js client components.
- No design system or layout shell yet; instrumentation minimal compared to Upstage console. [1](https://console.upstage.ai/playground/document-parsing)

## Target State
- Next.js 15 App Router project hosting marketing shell, navigation, and playground routes with hybrid SSR/CSR behavior.
- Chakra UI MIT library provides primitives, theming, and layout scaffolding; optional Radix/Mantine components for gaps.
- `/app/api` handlers or server actions proxy OCR jobs to backend services, enabling job lifecycle UI similar to Upstage’s document parsing playground. [1](https://console.upstage.ai/playground/document-parsing)

## Workstreams & Tasks
1. **Foundation & Workspace Setup**
   - Scaffold Next.js (App Router) workspace within repo; configure Turborepo or npm workspaces to share packages with existing SPA during migration.
   - Configure TypeScript path aliases, ESLint, testing harness (Playwright) consistent with current tooling.
   - Add CI job to build/test both apps until SPA retirement.
2. **Chakra UI Theming Layer**
   - Install `@chakra-ui/react`, `@emotion/react`, `@emotion/styled`, `framer-motion`.
   - Create design tokens (colors, typography) that emulate Upstage shell; wrap root layout with `ChakraProvider` + custom theme.
   - Build primitive components (AppShell, SideNav, TopNav, Banner, Notifications) using Chakra layouts.
3. **Playground Feature Migration**
   - Move Command Builder page into `app/(console)/playground/command-builder/page.tsx` as client component.
   - Convert inline styles to Chakra components; ensure SchemaForm, CommandDisplay, CommandDiffViewer render identically.
   - Implement Tab navigation via Chakra `Tabs` or custom Radix `Tabs` if more control is needed.
   - Introduce Extract workspace routes (`/playground/extract/universal`, `/playground/extract/prebuilt`) that share the console shell and reuse Chakra layout primitives.
   - For Prebuilt Extraction, implement a “business cards” gallery strip with thumbnail previews above the main preview panels, and dual synchronized panes (“Preview” + “JSON”) underneath. Note: advanced layout recognition is deferred but should be captured as a follow-up enhancement.
4. **API Proxy & State Management**
   - Stand up `/app/api/commands/[action]/route.ts` endpoints or server actions that call existing FastAPI backend; secure via session tokens.
   - Reuse TanStack Query + Zustand for client state; configure QueryClient provider in layout.
   - Add SSE or polling support for long-running OCR jobs.
5. **Observability & Compliance**
   - Integrate analytics (e.g., PostHog or GTM per compliance) plus cookie consent banner implemented via Chakra.
   - Instrument key events (uploads, command builds, inference runs) for future dashboards.

## Timeline / Milestones
- **Week 1:** Workspace + Next.js scaffold, Chakra theme, base shell components.
- **Week 2:** Command Builder migration + Chakra refactor, API proxy skeleton.
- **Week 3:** Additional playground modules (document viewer, recommendations), auth wiring, instrumentation baseline.
- **Week 4:** Hardening, cross-browser testing, SPA deprecation plan.

## Dependencies & Risks
- Backend endpoints must support CORS/auth tokens suitable for Next.js server actions.
- Chakra adoption may expose gaps (e.g., PDF viewer) requiring Radix or Mantine components.
- SEO/SSR will need marketing copy and CMS integration later; currently stubbed.

## Progress Tracker (Blueprint Lite)
- [ ] Foundation workspace ready
- [ ] Chakra theme + shell implemented
- [ ] Command Builder migrated to Next.js
- [ ] API proxy routes operational
- [ ] Instrumentation + consent banner live

## Required Follow-up Assessments
1. **Backend Proxy Assessment:** inventory FastAPI endpoints, auth model, and evaluate whether to build a BFF or reuse existing routes.
2. **Document Viewer Research:** identify PDF/image viewer libraries compatible with Chakra + Next.js (PDF.js, React-PDF, PSPDFKit, etc.).
3. **Analytics/Compliance Plan:** confirm telemetry stack (GTM vs PostHog) and cookie consent requirements.
4. **Performance Budgets:** measure current bundle sizes and set budgets for `_next/static` outputs; requires profiling once prototype exists.

## Open Questions & Resolutions
- **Auth strategy:** choose the most straightforward solution (e.g., NextAuth with credentials/JWT sessions) to keep onboarding simple; document final pick during backend proxy assessment.
- **Branding scope:** single Upstage-branded console only—no multi-tenant theming required in phase 1.
- **Phase 1 playground scope:** include Extract → Universal Extraction and Prebuilt Extraction pages alongside Command Builder. Prebuilt Extraction must surface the “business cards” thumbnail gallery and twin Preview/JSON panels; layout-aware extraction is a future enhancement but should be noted in backlog.
