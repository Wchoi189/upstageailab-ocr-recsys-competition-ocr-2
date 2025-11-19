---
title: "Autonomous Overnight Tasks"
author: "ai-agent"
timestamp: "2025-11-20 01:30 KST"
date: "2025-11-20"
type: "autonomous_work_plan"
category: "development"
status: "ready"
tags: ["autonomous", "testing", "code-quality", "documentation"]
---

# Autonomous Overnight Work Prompts

This document contains concrete, executable prompts for autonomous agents to work on unsupervised overnight. Tasks are prioritized by **autonomy** (clear success criteria, no human decisions), **risk** (won't break existing functionality), and **value** (testing, quality, documentation).

---

## 🎯 **PRIORITY 1: Testing & Quality Assurance**

### **Task A1: Run E2E Test Suite & Fix Failures**

**Prompt:**
```
You are working on the Frontend Functionality Completion plan (artifacts/implementation_plans/2025-11-19_1514_frontend-functionality-completion.md).

Your task is Task 4.1: E2E Test Coverage.

STEPS:
1. Navigate to frontend directory: `cd frontend`
2. Run E2E test suite: `npm run test:e2e`
3. If tests fail:
   - Identify failing tests and root causes
   - Fix failing tests incrementally (one at a time)
   - Verify each fix by re-running the specific test
   - Document any tests that require manual intervention or are flaky
4. If all tests pass:
   - Add new E2E tests for:
     - Image upload functionality (Preprocessing and Inference pages)
     - Worker pipeline (rembg, autocontrast, blur transforms)
     - Inference flow (checkpoint selection, preview generation)
     - Error scenarios (invalid files, network errors, API failures)
   - Each new test should verify:
     - UI elements are visible/accessible
     - User interactions work (file upload, button clicks)
     - API calls are made correctly
     - Error states are handled properly

SUCCESS CRITERIA:
- All existing E2E tests pass
- New tests added for at least 3 major user flows
- Test failures are documented with root causes if not fixable
- Update the implementation plan to mark Task 4.1 as complete

FILES TO CHECK:
- `tests/e2e/` - Existing test files
- `frontend/playwright.config.ts` - Playwright configuration
- `frontend/src/pages/Preprocessing.tsx` - Preprocessing page
- `frontend/src/pages/Inference.tsx` - Inference page

APPROACH:
- Start with running existing tests
- Fix failures before adding new tests
- Use Playwright best practices (page objects, wait strategies)
- Keep tests independent and idempotent
```

---

### **Task A2: Component Unit Testing**

**Prompt:**
```
You are working on the Frontend Functionality Completion plan (artifacts/implementation_plans/2025-11-19_1514_frontend-functionality-completion.md).

Your task is Task 4.2: Component Testing.

STEPS:
1. Set up testing framework if not already configured:
   - Install Vitest or React Testing Library if needed
   - Configure test setup in `frontend/vitest.config.ts` or similar
   - Add test scripts to `package.json`

2. Create unit tests for form components:
   - Test `frontend/src/components/forms/FormPrimitives.tsx`:
     - TextInput: value changes, validation, error states
     - Checkbox: checked/unchecked states
     - SelectBox: option selection, disabled states
     - NumberInput: number validation, min/max constraints
     - Slider: value changes, min/max bounds
   - Test `frontend/src/components/forms/SchemaForm.tsx`:
     - Form rendering from schema
     - Value updates propagate correctly
     - Validation rules are applied
     - Conditional visibility works

3. Create unit tests for API client:
   - Test `frontend/src/api/client.ts`:
     - Request retry logic (exponential backoff)
     - Error handling (ApiError, network errors, timeouts)
     - Request/response interceptors
   - Test API functions:
     - `frontend/src/api/commands.ts` - buildCommand, getSchemaDetails
     - `frontend/src/api/inference.ts` - runInferencePreview
     - `frontend/src/api/pipelines.ts` - queuePreview, queueFallback
     - Mock fetch API for all tests

4. Create integration tests for worker pipeline:
   - Test `frontend/workers/transforms.ts`:
     - runAutoContrast transforms images correctly
     - runGaussianBlur applies blur with correct kernel size
     - runRembgLite (test with mocked ONNX session to avoid loading model)
     - Error handling and fallbacks work

SUCCESS CRITERIA:
- At least 80% code coverage for form components
- All API client functions have tests
- Worker transforms have integration tests (mocked dependencies)
- All tests pass
- Update implementation plan with progress

FILES TO CREATE:
- `frontend/src/components/forms/__tests__/FormPrimitives.test.tsx`
- `frontend/src/components/forms/__tests__/SchemaForm.test.tsx`
- `frontend/src/api/__tests__/client.test.ts`
- `frontend/src/api/__tests__/commands.test.ts`
- `frontend/workers/__tests__/transforms.test.ts`

APPROACH:
- Use React Testing Library for component tests
- Use Vitest for unit tests
- Mock external dependencies (fetch, ONNX, ImageBitmap)
- Test both success and error paths
```

---

## 🎯 **PRIORITY 2: Code Quality & Organization**

### **Task A3: Fix TypeScript Types & Add JSDoc**

**Prompt:**
```
You are working on the Frontend Functionality Completion plan (artifacts/implementation_plans/2025-11-19_1514_frontend-functionality-completion.md).

Your task is Task 6.4: Code Quality & Organization (subset - types and documentation).

STEPS:
1. Find all `any` types in frontend codebase:
   ```bash
   cd frontend
   grep -r ": any" src/ workers/ --include="*.ts" --include="*.tsx"
   ```
   - Replace each `any` with proper TypeScript types
   - Use `unknown` if type is truly unknown, then narrow it
   - Create proper interfaces/types if needed

2. Add missing type definitions:
   - Check `frontend/src/types/` for missing interfaces
   - Ensure all API responses have proper types
   - Add types for worker messages and results
   - Add types for form values and schemas

3. Add JSDoc comments to complex functions:
   - Document all public functions in:
     - `frontend/src/api/*.ts` - API client functions
     - `frontend/workers/transforms.ts` - Transform functions
     - `frontend/src/hooks/*.ts` - Custom hooks
     - `frontend/src/components/preprocessing/PreprocessingCanvas.tsx`
     - `frontend/src/components/inference/InferencePreviewCanvas.tsx`
   - JSDoc should include:
     - Function description
     - @param descriptions for all parameters
     - @returns description
     - @throws if applicable
     - @example if helpful

4. Run TypeScript type check:
   ```bash
   cd frontend
   npm run type-check
   ```
   - Fix all type errors
   - Ensure strict mode passes

SUCCESS CRITERIA:
- Zero `any` types in frontend codebase (except where truly necessary)
- All public API functions have JSDoc comments
- All complex worker functions have JSDoc comments
- TypeScript strict mode passes without errors
- Update implementation plan with progress

FILES TO CHECK:
- `frontend/src/**/*.ts` and `frontend/src/**/*.tsx`
- `frontend/workers/**/*.ts`
- `frontend/tsconfig.json` - Verify strict mode is enabled

APPROACH:
- Work file by file, starting with API client and workers
- Use TypeScript's type inference where possible
- Create shared types in `frontend/src/types/`
- Document as you go, don't skip
```

---

### **Task A4: Extract Reusable Components**

**Prompt:**
```
You are working on the Frontend Functionality Completion plan (artifacts/implementation_plans/2025-11-19_1514_frontend-functionality-completion.md).

Your task is Task 6.4: Code Quality & Organization (subset - component extraction).

STEPS:
1. Analyze code duplication:
   - Check `frontend/src/pages/Preprocessing.tsx` and `frontend/src/pages/Inference.tsx`
   - Identify duplicate image upload logic
   - Identify duplicate error handling patterns
   - Identify duplicate loading state patterns

2. Create reusable image upload component:
   - Create `frontend/src/components/shared/ImageUpload.tsx`
   - Extract file upload logic from Preprocessing and Inference pages
   - Include:
     - File input with label
     - File validation (size, format)
     - Error display
     - File name display
     - Clear/reset functionality
   - Props: `onFileChange`, `accept`, `maxSize`, `error`, `disabled`
   - Update Preprocessing and Inference pages to use new component

3. Create shared image display utilities:
   - Create `frontend/src/utils/imageDisplay.ts`
   - Extract common image utilities:
     - `getImageMetadata(file: File): Promise<{width, height, size}>`
     - `createImageBitmapFromFile(file: File): Promise<ImageBitmap>`
     - `downloadImage(blob: Blob, filename: string): void`
     - `formatFileSize(bytes: number): string`

4. Consolidate duplicate error handling:
   - Check if toast notification logic is duplicated
   - Ensure consistent error message formatting
   - Create shared error display component if needed

SUCCESS CRITERIA:
- ImageUpload component is reusable and tested
- Preprocessing and Inference pages use ImageUpload
- Image display utilities are extracted and tested
- No duplicate code for image upload across pages
- Code is cleaner and more maintainable
- Update implementation plan with progress

FILES TO CREATE:
- `frontend/src/components/shared/ImageUpload.tsx`
- `frontend/src/utils/imageDisplay.ts`

FILES TO UPDATE:
- `frontend/src/pages/Preprocessing.tsx`
- `frontend/src/pages/Inference.tsx`

APPROACH:
- Start by identifying exact duplicate code
- Extract to component/utility first
- Update all usages
- Test that functionality still works
- Remove old duplicate code
```

---

## 🎯 **PRIORITY 3: Documentation**

### **Task A5: Add Inline Documentation & Tooltips**

**Prompt:**
```
You are working on the Frontend Functionality Completion plan (artifacts/implementation_plans/2025-11-19_1514_frontend-functionality-completion.md).

Your task is Task 6.5: Documentation (subset - inline help and tooltips).

STEPS:
1. Add tooltips to complex form fields:
   - Review `frontend/src/components/forms/FormPrimitives.tsx`
   - Add help text/tooltips for:
     - Slider inputs (explain min/max, step)
     - NumberInput (explain validation rules)
     - SelectBox (explain option sources)
   - Use `help` prop from schema to display tooltips
   - If no help prop, add sensible defaults

2. Add inline help for each page:
   - Add help text section at top of:
     - `frontend/src/pages/Preprocessing.tsx` - Explain preprocessing options
     - `frontend/src/pages/Inference.tsx` - Explain inference workflow
     - `frontend/src/pages/Comparison.tsx` - Explain comparison features
     - `frontend/src/pages/CommandBuilder.tsx` - Explain command building
   - Help text should be:
     - Brief and actionable
     - Use collapsible sections (details/summary HTML)
     - Include links to full docs if available

3. Document API contracts:
   - Create `frontend/src/api/README.md`
   - Document:
     - All API endpoints used
     - Request/response formats
     - Error handling
     - Retry logic
     - Authentication (if applicable)
   - Include examples for each endpoint

4. Document worker pipeline flow:
   - Create `frontend/workers/README.md`
   - Document:
     - Worker architecture
     - Transform pipeline flow
     - How to add new transforms
     - ONNX.js integration details
     - Error handling and fallbacks

SUCCESS CRITERIA:
- All complex form fields have tooltips
- All pages have inline help sections
- API contracts are documented with examples
- Worker pipeline is documented
- Documentation is clear and actionable
- Update implementation plan with progress

FILES TO CREATE:
- `frontend/src/api/README.md`
- `frontend/workers/README.md`

FILES TO UPDATE:
- `frontend/src/components/forms/FormPrimitives.tsx`
- `frontend/src/pages/Preprocessing.tsx`
- `frontend/src/pages/Inference.tsx`
- `frontend/src/pages/Comparison.tsx`
- `frontend/src/pages/CommandBuilder.tsx`

APPROACH:
- Keep help text concise (users don't read long paragraphs)
- Use progressive disclosure (show basics, hide details)
- Include code examples where helpful
- Test that tooltips don't break layout
```

---

## 🎯 **PRIORITY 4: Feature Enhancements (Lower Risk)**

### **Task A6: Image Download Functionality**

**Prompt:**
```
You are working on the Frontend Functionality Completion plan (artifacts/implementation_plans/2025-11-19_1514_frontend-functionality-completion.md).

Your task is Task 6.1: Image Display Enhancements (subset - download functionality).

STEPS:
1. Add download button to PreprocessingCanvas:
   - Add "Download" button next to processed image
   - Button should download the processed image as PNG
   - Use filename pattern: `preprocessed-{timestamp}.png`
   - Implement in `frontend/src/components/preprocessing/PreprocessingCanvas.tsx`

2. Add download button to InferencePreviewCanvas:
   - Add "Download" button next to inference result
   - Button should download the image with overlays as PNG
   - Use filename pattern: `inference-{timestamp}.png`
   - Implement in `frontend/src/components/inference/InferencePreviewCanvas.tsx`

3. Create shared download utility:
   - Create `frontend/src/utils/imageDownload.ts`
   - Function: `downloadCanvasAsImage(canvas: HTMLCanvasElement, filename: string): void`
   - Function: `downloadImageDataAsImage(imageData: ImageData, filename: string): void`
   - Reuse in both components

4. Add keyboard shortcut (optional):
   - Add Ctrl+S (Cmd+S on Mac) to trigger download
   - Use `useEffect` with keyboard event listener
   - Show toast notification when download completes

SUCCESS CRITERIA:
- Download buttons work on both Preprocessing and Inference pages
- Downloaded images match displayed images
- Filenames are descriptive and unique
- Download utility is reusable
- Update implementation plan with progress

FILES TO CREATE:
- `frontend/src/utils/imageDownload.ts`

FILES TO UPDATE:
- `frontend/src/components/preprocessing/PreprocessingCanvas.tsx`
- `frontend/src/components/inference/InferencePreviewCanvas.tsx`

APPROACH:
- Test with different image sizes
- Ensure downloads work in all browsers
- Handle edge cases (no image available, processing errors)
- Add loading state if download is slow
```

---

### **Task A7: State Persistence with localStorage**

**Prompt:**
```
You are working on the Frontend Functionality Completion plan (artifacts/implementation_plans/2025-11-19_1514_frontend-functionality-completion.md).

Your task is Task 6.2: State Persistence.

STEPS:
1. Create localStorage utility:
   - Create `frontend/src/utils/persistence.ts`
   - Functions:
     - `saveToLocalStorage<T>(key: string, value: T): void`
     - `loadFromLocalStorage<T>(key: string, defaultValue: T): T`
     - Handle JSON serialization/deserialization
     - Handle errors gracefully (quota exceeded, etc.)

2. Persist form values in CommandBuilder:
   - Save form values to localStorage on change (debounced)
   - Load saved values on page load
   - Use key: `command-builder-{schemaId}-values`
   - Implement in `frontend/src/pages/CommandBuilder.tsx`

3. Persist preprocessing parameters:
   - Save preprocessing params to localStorage
   - Load saved params on page load
   - Use key: `preprocessing-params`
   - Implement in `frontend/src/pages/Preprocessing.tsx`

4. Cache checkpoint list:
   - Save checkpoint list to localStorage with timestamp
   - Reload if cache is older than 5 minutes
   - Use key: `checkpoints-cache`
   - Implement in checkpoint loading logic

5. Remember last used image (optional):
   - Store image file metadata (not the file itself)
   - Store last used parameters with image
   - Use key: `last-image-params`

SUCCESS CRITERIA:
- Form values persist across page reloads
- Preprocessing params persist across sessions
- Checkpoint list is cached appropriately
- localStorage errors are handled gracefully
- No performance degradation from persistence
- Update implementation plan with progress

FILES TO CREATE:
- `frontend/src/utils/persistence.ts`

FILES TO UPDATE:
- `frontend/src/pages/CommandBuilder.tsx`
- `frontend/src/pages/Preprocessing.tsx`
- Checkpoint loading logic (if exists)

APPROACH:
- Use debouncing for frequent saves
- Validate loaded data before using
- Clear invalid/corrupted data
- Test with full localStorage (quota exceeded)
- Don't store large objects (images, bitmaps)
```

---

## 📋 **Execution Guidelines for Autonomous Agents**

### **General Rules:**
1. **Always update the implementation plan** after completing a task or encountering blockers
2. **Commit incrementally** - Small, focused commits with clear messages
3. **Test before committing** - Run relevant tests/linters before marking complete
4. **Document blockers** - If something can't be completed, document why and what's needed
5. **Follow existing patterns** - Match code style, naming conventions, and architecture

### **If You Encounter Blockers:**
1. Document the blocker in the implementation plan
2. Move to the next task if possible
3. Don't force-fix things that require human decisions
4. Leave clear notes for follow-up

### **Success Indicators:**
- ✅ All tests pass
- ✅ TypeScript compiles without errors
- ✅ No new linting errors introduced
- ✅ Code follows project conventions
- ✅ Implementation plan is updated

---

## 🎯 **Recommended Execution Order**

For overnight autonomous work, execute tasks in this order:

1. **Task A3** (Fix Types) - Quick wins, improves code quality
2. **Task A1** (E2E Tests) - Critical for quality assurance
3. **Task A5** (Documentation) - Low risk, high value
4. **Task A7** (State Persistence) - Simple, independent feature
5. **Task A2** (Component Tests) - More time-intensive
6. **Task A6** (Image Download) - Feature enhancement
7. **Task A4** (Extract Components) - Refactoring (do last if time permits)

---

## 🎯 **PRIORITY 5: Next.js Console Migration (Phase 4)**

### **Task A8: API Proxy Routes Implementation**

**Prompt:**
```
You are working on the Next.js Console Migration plan (artifacts/implementation_plans/2025-11-19_1957_next.js-console-migration-and-chakra-adoption-plan.md).

Your task is Task 4.1 (Phase 4): API Proxy & Auth (subset - API proxy routes).

STEPS:
1. Create API route structure:
   - Create `apps/playground-console/src/app/api/commands/` directory
   - Create route handlers for:
     - `apps/playground-console/src/app/api/commands/schemas/route.ts` - GET `/api/commands/schemas`
     - `apps/playground-console/src/app/api/commands/details/route.ts` - GET `/api/commands/details?schema_id={id}`
     - `apps/playground-console/src/app/api/commands/build/route.ts` - POST `/api/commands/build`
     - `apps/playground-console/src/app/api/commands/recommendations/route.ts` - GET `/api/commands/recommendations`

2. Implement proxy logic:
   - Each route handler should:
     - Read FastAPI URL from environment: `process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000"`
     - Forward request to FastAPI endpoint
     - Forward request body/query params
     - Handle errors and return appropriate status codes
     - Add CORS headers if needed (though server-side shouldn't need them)
   - Example structure:
     ```typescript
     // app/api/commands/build/route.ts
     import { NextRequest, NextResponse } from "next/server";

     export async function POST(request: NextRequest) {
       const body = await request.json();
       const apiUrl = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";

       try {
         const response = await fetch(`${apiUrl}/api/commands/build`, {
           method: "POST",
           headers: { "Content-Type": "application/json" },
           body: JSON.stringify(body),
         });

         if (!response.ok) {
           return NextResponse.json(
             { error: await response.text() },
             { status: response.status }
           );
         }

         return NextResponse.json(await response.json());
       } catch (error) {
         return NextResponse.json(
           { error: error instanceof Error ? error.message : "Unknown error" },
           { status: 500 }
         );
       }
     }
     ```

3. Create shared proxy utility (optional):
   - Create `apps/playground-console/src/lib/api-proxy.ts`
   - Extract common proxy logic:
     - `proxyRequest(path: string, options: RequestInit): Promise<Response>`
     - Error handling and logging

4. Update client API calls:
   - Update `apps/playground-console/src/api/client.ts`:
     - Change `API_BASE_URL` to use relative paths: `const API_BASE_URL = "/api"`
     - All client calls now go through Next.js API routes
   - Verify all API calls work through proxy

5. Add environment variables:
   - Create/update `.env.local`:
     ```
     FASTAPI_BASE_URL=http://127.0.0.1:8000
     ```
   - Document in README

6. Test proxy routes:
   - Start Next.js dev server: `cd apps/playground-console && npm run dev`
   - Test Command Builder page - verify API calls go to `/api/commands/*`
   - Verify in Network tab that client calls `/app/api/...` and server proxies to FastAPI
   - Check server logs to confirm FastAPI requests

SUCCESS CRITERIA:
- All API calls from client go through Next.js proxy routes (`/app/api/...`)
- Proxy routes correctly forward to FastAPI backend
- Error handling works (network errors, API errors, timeouts)
- Command Builder still works end-to-end
- Network tab shows calls to `/api/commands/*` not directly to FastAPI
- Update Next.js plan to mark API proxy part of Task 4.1 as complete

FILES TO CREATE:
- `apps/playground-console/src/app/api/commands/schemas/route.ts`
- `apps/playground-console/src/app/api/commands/details/route.ts`
- `apps/playground-console/src/app/api/commands/build/route.ts`
- `apps/playground-console/src/app/api/commands/recommendations/route.ts`
- `apps/playground-console/src/lib/api-proxy.ts` (optional)

FILES TO UPDATE:
- `apps/playground-console/src/api/client.ts` - Change to use `/api` instead of FastAPI URL
- `apps/playground-console/.env.local` - Add FASTAPI_BASE_URL

APPROACH:
- Start with one route (build), test it, then replicate pattern
- Use Next.js 16 App Router conventions (route.ts files)
- Handle errors gracefully with appropriate status codes
- Keep proxy logic simple - just forward requests/responses
- Don't implement auth yet (that's separate step)
```

---

### **Task A9: Session & Auth Context (Simplified)**

**Prompt:**
```
You are working on the Next.js Console Migration plan (artifacts/implementation_plans/2025-11-19_1957_next.js-console-migration-and-chakra-adoption-plan.md).

Your task is Task 4.1 (Phase 4): API Proxy & Auth (subset - session hooks).

STEPS:
1. Create session context provider:
   - Create `apps/playground-console/src/contexts/SessionContext.tsx`
   - Create a simple session context (no auth yet, just session state):
     ```typescript
     interface Session {
       user: { id?: string; email?: string } | null;
       isAuthenticated: boolean;
     }

     const SessionContext = createContext<Session>({
       user: null,
       isAuthenticated: false,
     });
     ```
   - For now, session is always unauthenticated (placeholder for future auth)

2. Create useSession hook:
   - Create `apps/playground-console/src/hooks/useSession.ts`
   - Hook returns current session state
   - Hook can be extended later for auth logic

3. Integrate session provider:
   - Update `apps/playground-console/src/providers/AppProviders.tsx`:
     - Wrap app in `<SessionProvider>`
     - Session provider provides context to all children

4. Add session hooks to client pages:
   - Update `apps/playground-console/src/components/command-builder/CommandBuilderClient.tsx`:
     - Import and use `useSession` hook
     - Conditionally show/hide elements based on session (placeholder for now)
   - Update TopNav if it needs session info

5. Add session types:
   - Ensure types are exported from context
   - Types match what will be needed for future auth

SUCCESS CRITERIA:
- SessionContext is available throughout app
- useSession hook works and returns session state
- AppProviders wraps app with SessionProvider
- No runtime errors related to session
- Ready for future auth integration
- Update Next.js plan to mark session hooks part of Task 4.1 as complete

FILES TO CREATE:
- `apps/playground-console/src/contexts/SessionContext.tsx`
- `apps/playground-console/src/hooks/useSession.ts`

FILES TO UPDATE:
- `apps/playground-console/src/providers/AppProviders.tsx` - Add SessionProvider
- `apps/playground-console/src/components/layout/TopNav.tsx` - Use session if needed

APPROACH:
- Keep it simple - just the structure for now
- No actual auth implementation (that requires backend decisions)
- Make it easy to extend later with NextAuth or custom JWT
- Session is always unauthenticated for now (placeholder)
```

---

### **Task A10: Analytics & GTM Integration**

**Prompt:**
```
You are working on the Next.js Console Migration plan (artifacts/implementation_plans/2025-11-19_1957_next.js-console-migration-and-chakra-adoption-plan.md).

Your task is Task 4.2: Analytics & Compliance.

STEPS:
1. Install GTM dependencies:
   - Add `@next/third-parties` package for Next.js GTM integration:
     ```bash
     cd apps/playground-console
     npm install @next/third-parties
     ```

2. Create GTM provider component:
   - Create `apps/playground-console/src/components/analytics/GTMProvider.tsx`
   - Use `@next/third-parties/google` to load GTM:
     ```typescript
     import { GoogleTagManager } from "@next/third-parties/google";

     export function GTMProvider({ children }: { children: React.ReactNode }) {
       const gtmId = process.env.NEXT_PUBLIC_GTM_ID;

       if (!gtmId) {
         return <>{children}</>;
       }

       return (
         <>
           <GoogleTagManager gtmId={gtmId} />
           {children}
         </>
       );
     }
     ```

3. Add consent banner:
   - Create `apps/playground-console/src/components/analytics/ConsentBanner.tsx`
   - Use Chakra UI components:
     - Use `Alert` or `Banner` component from Chakra
     - Show cookie consent message
     - Accept/Decline buttons
     - Store consent in localStorage or cookie
     - Only load GTM if consent is given

4. Integrate consent management:
   - Create `apps/playground-console/src/hooks/useConsent.ts`:
     - Hook to get/set consent state
     - Check localStorage for existing consent
     - Show banner if no consent given
   - Update GTMProvider to only load if consent is given

5. Add consent banner to layout:
   - Update `apps/playground-console/src/app/layout.tsx`:
     - Add ConsentBanner component
     - Add GTMProvider (conditionally based on consent)
   - Banner should appear at top or bottom of page

6. Document instrumentation events:
   - Create `apps/playground-console/src/lib/analytics.ts`:
     - Function: `trackEvent(name: string, properties?: Record<string, any>): void`
     - Pushes to `window.dataLayer` (GTM's data layer)
     - Only tracks if consent is given
   - Document key events to track:
     - Command build (`command_built`)
     - Image upload (`image_uploaded`)
     - Inference run (`inference_run`)
     - Page view (`page_view`)

7. Add event tracking to key actions:
   - Update Command Builder to track command builds
   - Update Extract pages to track uploads and inference
   - Use `trackEvent` function consistently

8. Add environment variable:
   - Update `.env.local`:
     ```
     NEXT_PUBLIC_GTM_ID=GTM-XXXXXXX
     ```
   - Document that GTM_ID is optional (app works without it)

9. Verify consent flow:
   - Test that banner appears on first visit
   - Test that accepting hides banner
   - Test that declining prevents GTM from loading
   - Test that consent persists across page reloads
   - Verify events are pushed to dataLayer only with consent

SUCCESS CRITERIA:
- GTM integration works (if GTM_ID is provided)
- Consent banner appears and works correctly
- Consent state persists across sessions
- Analytics events are tracked (with consent)
- Events are documented in code
- App works without GTM_ID (graceful degradation)
- Update Next.js plan to mark Task 4.2 as complete

FILES TO CREATE:
- `apps/playground-console/src/components/analytics/GTMProvider.tsx`
- `apps/playground-console/src/components/analytics/ConsentBanner.tsx`
- `apps/playground-console/src/hooks/useConsent.ts`
- `apps/playground-console/src/lib/analytics.ts`

FILES TO UPDATE:
- `apps/playground-console/package.json` - Add @next/third-parties
- `apps/playground-console/src/app/layout.tsx` - Add ConsentBanner and GTMProvider
- `apps/playground-console/src/components/command-builder/CommandBuilderClient.tsx` - Add event tracking
- `apps/playground-console/.env.local` - Add NEXT_PUBLIC_GTM_ID

APPROACH:
- Use Next.js official GTM package (@next/third-parties)
- Keep consent banner simple but compliant
- Make analytics optional (don't break app without GTM_ID)
- Document all tracked events
- Test consent flow thoroughly
- Respect user's consent choice
```

---

## 📋 **Updated Execution Order for Next.js Tasks**

For Next.js console migration, execute in this order:

1. **Task A8** (API Proxy Routes) - Foundation for all API calls
2. **Task A9** (Session Context) - Simple structure, no auth yet
3. **Task A10** (Analytics & GTM) - Independent feature

These can be done alongside the frontend functionality tasks (A1-A7) as they're in a different codebase (`apps/playground-console/` vs `frontend/`).

---

*These prompts are designed for autonomous execution. Each task has clear success criteria, specific files to modify, and actionable steps. Agents can work on these unsupervised overnight.*

