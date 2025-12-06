---
type: assessment
title: "Upstage Document Parsing Playground Tech Stack"
date: "2025-11-19 19:39 (KST)"
category: architecture
status: draft
version: "1.0"
tags:
  - architecture
  - tech-stack
  - document-parsing
author: ai-agent
branch: main
---

## Progress Tracker
*(Required for iterative assessments, debugging sessions, or incremental work)*

- **STATUS:** Not Started / In Progress / Completed
- **CURRENT STEP:** [Current phase or task being worked on]
- **LAST COMPLETED TASK:** [Description of last completed task]
- **NEXT TASK:** [Description of the immediate next task]

### Assessment Checklist
- [ ] Initial assessment complete
- [ ] Analysis phase complete
- [ ] Recommendations documented
- [ ] Review and validation complete

---

## 1. Summary

## 2. Assessment

## 3. Recommendations

## Summary
- Document Parsing playground runs on React 18 + Next.js (App Router) with route-scoped bundles under `_next/static/chunks/...`, implying SSR/ISR support plus client hydration for interactive widgets [1](https://console.upstage.ai/playground/document-parsing)
- Styling is precompiled into hashed CSS assets, indicating a utility-first or CSS Modules design system rather than runtime CSS-in-JS [1](https://console.upstage.ai/playground/document-parsing)
- Console shell exposes multi-product navigation (Generate, Digitize, Extract) with authenticated links to Docs/API/Dashboard, highlighting the need for shared layout, routing, and auth context [1](https://console.upstage.ai/playground/document-parsing)
- Third-party instrumentation includes Microsoft Clarity, Google Tag Manager/Analytics, LinkedIn Insight, and Elfsight, showing a strong focus on behavioral analytics and marketing integrations [1](https://console.upstage.ai/playground/document-parsing)
- API surface is hinted via `/api/document-digitization/document-parsing`; playground likely proxies authenticated REST/GQL calls for inference jobs, requiring a typed client and job lifecycle management [1](https://console.upstage.ai/playground/document-parsing)

## Progress Tracker
- [x] Inspect playground bundles, markup, and external resources (2025-11-19)
- [x] Capture architectural/UX patterns relevant to OCR app
- [x] Document recommendations for replication

## Frontend & Runtime Stack
- **Framework:** Next.js w/ App Router (evidenced by `_next/static/chunks/app/(frame)/...` assets) enables hybrid SSR/CSR, route nesting, and server actions for inference requests [1](https://console.upstage.ai/playground/document-parsing)
- **Language/tooling:** Likely TypeScript + React 18; chunk naming plus modern bundle splitting imply SWC/Vite-powered build pipeline shipped via Vercel-style edge infra [1](https://console.upstage.ai/playground/document-parsing)
- **Styling:** Two hashed CSS bundles indicate build-time extraction (Tailwind JIT, Vanilla Extract, or CSS Modules). No runtime style tags detected, reducing layout shift risk [1](https://console.upstage.ai/playground/document-parsing)
- **Asset delivery:** CDN-hosted `_next` assets and polyfills bundle align with Next.js default asset pipeline; good pattern for global OCR users [1](https://console.upstage.ai/playground/document-parsing)

## Layout, UX, and IA Observations
- **Console shell:** Top bar (Docs, API Reference, Playground, Dashboard) and side nav (Generate, Digitize, Extract) suggest a shared shell with route groups; implement via Next.js layout nesting to keep context/state [1](https://console.upstage.ai/playground/document-parsing)
- **Hero + CTA:** Marketing banner (“Solar Pro 2 is now live”) and CTA linking to chat playground show SSR content for SEO plus dynamic sections; plan for CMS-fed hero modules [1](https://console.upstage.ai/playground/document-parsing)
- **Auth gating:** Prominent “Log in” button and dashboard link highlight need for session-aware components, probably NextAuth or custom JWT cookies [1](https://console.upstage.ai/playground/document-parsing)
- **Cookie consent + notifications:** Global trays/banners point to client-side state store (Zustand/Redux) for global modals; replicate for compliance messaging [1](https://console.upstage.ai/playground/document-parsing)

## Instrumentation & Integrations
- **Analytics:** Microsoft Clarity (`clarity.js`), Google Tag Manager/Analytics (`gtag.js`, `gtm.js`), LinkedIn Insight, and Google Ads conversion pixels are all present, ensuring funnel and UX telemetry [1](https://console.upstage.ai/playground/document-parsing)
- **Widgets:** Elfsight platform script suggests embeddable widgets (e.g., testimonials, forms). For your OCR console, budget for similar marketing/CS features or replace with internal components [1](https://console.upstage.ai/playground/document-parsing)
- **Operational insight:** Maintain tagging plan upfront (events for uploads, inference runs, errors) using a privacy-compliant stack like PostHog if you prefer self-hosting.

## Backend/API Implications
- **Inference proxy:** Route structure implies the UI calls Upstage Console APIs rather than hitting the foundation model directly. Adopt a BFF (backend-for-frontend) that authenticates users, signs OCR jobs, and streams progress updates [1](https://console.upstage.ai/playground/document-parsing)
- **Job lifecycle:** Expect long-running document parsing; design endpoints that return job IDs, support polling/webhooks/SSE, and store intermediate artifacts (thumbnails, text layers).
- **Sample assets:** Provide canned PDFs/images in storage (e.g., S3) to let visitors test without uploading, mirroring the “Try now” experience.

## Recommendations for Your OCR Playground
1. **Framework parity:** Use Next.js App Router with React 18, TypeScript, and a design system (Tailwind + Radix) to match the responsiveness and layout flexibility observed. [1](https://console.upstage.ai/playground/document-parsing)
2. **Shared console shell:** Build a persistent layout that houses navigation, search, auth, and notifications so future OCR tools (document OCR, extraction, evaluation) feel cohesive. [1](https://console.upstage.ai/playground/document-parsing)
3. **Document viewer:** Integrate PDF.js or a Canvas-based viewer synchronized with recognized text/metadata, plus drag-and-drop uploads and preset samples.
4. **API client + job manager:** Create an SDK wrapping your OCR inference endpoints with retries, progress streaming (SSE/WebSocket), and error normalization.
5. **Observability:** Wire analytics and session replay early (PostHog, OpenTelemetry) to mirror Console’s data-driven approach while honoring privacy requirements. [1](https://console.upstage.ai/playground/document-parsing)
6. **Compliance overlays:** Include cookie consent, status banners, and contextual alerts managed through a global store; this keeps regulatory messaging consistent. [1](https://console.upstage.ai/playground/document-parsing)
