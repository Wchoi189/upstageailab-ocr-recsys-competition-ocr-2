# Remaining Tasks to Make App More Functional

**Date:** 2025-11-19
**Status:** Active Development

---

## 🚨 **Critical Functionality Gaps**

### 1. **Image Upload & Processing**

#### Preprocessing Studio (`/preprocessing`)
- ✅ **File upload UI** - Already wired (file input handler exists)
- ⚠️ **Image validation** - Add file size limits, format validation
- ⚠️ **Error handling** - Better error messages for invalid files
- ⚠️ **Drag & drop** - Consider adding drag-and-drop support
- ✅ **Worker pipeline** - Already wired, but see Worker Implementation below

#### Inference Studio (`/inference`)
- ✅ **File upload UI** - Already wired (file input handler exists)
- ⚠️ **Image validation** - Add file size limits, format validation
- ⚠️ **Error handling** - Better error messages for invalid files
- ⚠️ **Base64 conversion** - Currently works but could be optimized
- ⚠️ **Multiple image support** - Currently only handles single image

---

## 🔧 **Backend API Integration**

### 2. **Inference API - Real Implementation**

**File:** `services/playground_api/routers/inference.py`

**Current Status:** Stub implementation returns mock data

**Tasks:**
- [ ] Wire `/api/inference/preview` to actual model inference
- [ ] Integrate with existing inference service/checkpoint loading
- [ ] Handle image preprocessing (if needed before inference)
- [ ] Return real text regions with actual confidence scores
- [ ] Add proper error handling for model loading failures
- [ ] Support batch inference (if needed)

**Reference:** Line 162-186 in `inference.py` shows stub implementation

---

### 3. **Pipeline API - Real Implementation**

**File:** `services/playground_api/routers/pipeline.py`

**Current Status:** Likely stub implementations

**Tasks:**
- [ ] Implement `/api/pipelines/preview` endpoint
- [ ] Wire to actual preprocessing pipeline
- [ ] Support hybrid routing (client-side vs server-side)
- [ ] Implement `/api/pipelines/fallback` endpoint
- [ ] Add proper job queuing/status tracking

---

### 4. **Worker Implementation - ONNX.js Integration**

**File:** `frontend/workers/transforms.ts`

**Current Status:** `runRembgLite` is a stub that calls `runAutoContrast`

**Tasks:**
- [ ] Integrate actual ONNX.js runtime
- [ ] Load rembg ONNX model
- [ ] Implement background removal transform
- [ ] Handle model loading errors gracefully
- [ ] Optimize model bundle size
- [ ] Add fallback to server-side rembg if client-side fails

**Reference:** Line 89-93 in `transforms.ts` shows stub

---

## 🎨 **UI/UX Improvements**

### 5. **Loading States**

**Files:** Multiple components

**Tasks:**
- [ ] Add loading spinners to PreprocessingCanvas
- [ ] Add loading states to InferencePreviewCanvas
- [ ] Show progress indicators for long-running operations
- [ ] Add skeleton loaders for checkpoint list
- [ ] Disable buttons during async operations

---

### 6. **Error Handling & User Feedback**

**Files:** All pages and components

**Tasks:**
- [ ] Better error messages for API failures
- [ ] Toast notifications for success/error states
- [ ] Retry mechanisms for failed requests
- [ ] Network error detection and handling
- [ ] Validation error display in forms
- [ ] File upload error messages (size, format)

---

### 7. **Image Display Enhancements**

**Files:** `PreprocessingCanvas.tsx`, `InferencePreviewCanvas.tsx`

**Tasks:**
- [ ] Zoom/pan controls for large images
- [ ] Image download functionality
- [ ] Before/after comparison slider
- [ ] Image metadata display (dimensions, file size)
- [ ] Keyboard shortcuts for navigation

---

## 🔌 **API Client Enhancements**

### 8. **API Error Handling**

**File:** `frontend/src/api/client.ts`

**Tasks:**
- [ ] Centralized error handling
- [ ] Retry logic for transient failures
- [ ] Request timeout handling
- [ ] Better error type discrimination
- [ ] Logging for debugging

---

### 9. **Missing API Endpoints**

**Check if these endpoints exist and wire them:**

- [ ] Image upload endpoint (if needed for server-side processing)
- [ ] Checkpoint metadata endpoint (if not already covered)
- [ ] Pipeline status/job tracking endpoints
- [ ] Gallery/image management endpoints

---

## 🧪 **Testing & Quality**

### 10. **E2E Test Coverage**

**Files:** `tests/e2e/`

**Tasks:**
- [ ] Run full E2E test suite: `cd frontend && npm run test:e2e`
- [ ] Fix any failing tests
- [ ] Add tests for image upload functionality
- [ ] Add tests for worker pipeline
- [ ] Add tests for inference flow
- [ ] Add tests for error scenarios

---

### 11. **Component Testing**

**Tasks:**
- [ ] Unit tests for form components
- [ ] Unit tests for API client functions
- [ ] Integration tests for worker pipeline
- [ ] Visual regression tests (optional)

---

## 📊 **Data & State Management**

### 12. **State Persistence**

**Tasks:**
- [ ] Persist form values to localStorage
- [ ] Save user preferences (theme, default settings)
- [ ] Cache checkpoint list
- [ ] Remember last used parameters

---

### 13. **Performance Optimization**

**Tasks:**
- [ ] Image compression before upload
- [ ] Lazy loading for checkpoint list
- [ ] Debounce/throttle expensive operations
- [ ] Memoize expensive computations
- [ ] Code splitting for large components
- [ ] Optimize worker bundle size

---

## 🎯 **Feature Completeness**

### 14. **Comparison Studio**

**File:** `frontend/src/pages/Comparison.tsx`

**Tasks:**
- [ ] Verify all comparison presets work
- [ ] Wire up actual comparison execution
- [ ] Display comparison results
- [ ] Add result visualization
- [ ] Export comparison reports

---

### 15. **Command Builder Enhancements**

**File:** `frontend/src/pages/CommandBuilder.tsx`

**Tasks:**
- [ ] Command execution (if not already implemented)
- [ ] Command history
- [ ] Save/load command templates
- [ ] Export commands to file
- [ ] Command validation feedback

---

## 🔍 **Code Quality**

### 16. **TypeScript Improvements**

**Tasks:**
- [ ] Add missing type definitions
- [ ] Fix any `any` types
- [ ] Add JSDoc comments for complex functions
- [ ] Ensure all API responses are properly typed

---

### 17. **Code Organization**

**Tasks:**
- [ ] Extract reusable image upload component
- [ ] Create shared image display utilities
- [ ] Consolidate duplicate code
- [ ] Improve component composition

---

## 📝 **Documentation**

### 18. **User Documentation**

**Tasks:**
- [ ] Add tooltips/help text for complex features
- [ ] Create user guide
- [ ] Add inline help for each page
- [ ] Document keyboard shortcuts

---

### 19. **Developer Documentation**

**Tasks:**
- [ ] Document API contracts
- [ ] Update architecture diagrams
- [ ] Document worker pipeline flow
- [ ] Add code examples for common tasks

---

## 🚀 **Priority Order**

### **High Priority (Core Functionality)**
1. ✅ Image upload UI (already done)
2. ⚠️ Real inference API implementation (#2)
3. ⚠️ ONNX.js rembg integration (#4)
4. ⚠️ Loading states (#5)
5. ⚠️ Error handling (#6)

### **Medium Priority (User Experience)**
6. Image validation (#1)
7. Better error messages (#6)
8. Image display enhancements (#7)
9. State persistence (#12)
10. E2E test fixes (#10)

### **Low Priority (Polish)**
11. Drag & drop upload
12. Zoom/pan controls
13. Performance optimizations (#13)
14. Documentation (#18, #19)

---

## 📋 **Quick Reference: Files to Modify**

### **Backend (Python)**
- `services/playground_api/routers/inference.py` - Real inference
- `services/playground_api/routers/pipeline.py` - Pipeline endpoints
- `services/playground_api/routers/evaluation.py` - Comparison execution

### **Frontend (TypeScript/React)**
- `frontend/workers/transforms.ts` - ONNX.js integration
- `frontend/src/components/preprocessing/PreprocessingCanvas.tsx` - Loading states
- `frontend/src/components/inference/InferencePreviewCanvas.tsx` - Error handling
- `frontend/src/api/client.ts` - Error handling
- `frontend/src/pages/Preprocessing.tsx` - Image validation
- `frontend/src/pages/Inference.tsx` - Image validation

---

## 🔗 **Related Documentation**

- **Session Handover:** `docs/sessions/2025-11-19_session_handover_frontend.md`
- **Implementation Plan:** `artifacts/implementation_plans/2025-11-18_0241_high-performance-playground-migration.md`
- **API Documentation:** `http://127.0.0.1:8000/docs` (when server is running)

---

**Last Updated:** 2025-11-19






