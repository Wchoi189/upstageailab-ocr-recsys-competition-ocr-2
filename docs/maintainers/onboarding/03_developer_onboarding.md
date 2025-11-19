# Developer Onboarding Guide

Welcome to the OCR High-Performance Playground development team! This guide will help you get set up and productive quickly.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Architecture Overview](#architecture-overview)
4. [Development Workflow](#development-workflow)
5. [Code Standards](#code-standards)
6. [Testing](#testing)
7. [Debugging Tips](#debugging-tips)
8. [Common Tasks](#common-tasks)
9. [Resources](#resources)

---

## Prerequisites

### Required Software

- **Git**: Version control
- **Python 3.10+**: Backend runtime
- **Node.js 18+**: Frontend runtime
- **UV**: Python package manager
- **npm**: Node.js package manager

### Recommended Tools

- **VS Code**: IDE with extensions:
  - Python (ms-python.python)
  - Pylance (ms-python.vscode-pylance)
  - ESLint (dbaeumer.vscode-eslint)
  - Prettier (esbenp.prettier-vscode)
  - Playwright Test for VSCode (ms-playwright.playwright)
- **Git**: For version control
- **Postman** or **curl**: For API testing
- **React DevTools**: Browser extension for debugging

### Optional Tools

- **PyCharm**: Alternative IDE
- **Docker**: For containerized development (future)
- **Redis**: For persistent cache (future)

---

## Environment Setup

### Step 1: Clone Repository

```bash
git clone <repo-url>
cd upstageailab-ocr-recsys-competition-ocr-2
```

### Step 2: Install Backend Dependencies

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync Python dependencies
uv sync
```

### Step 3: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 4: Install Playwright (for E2E testing)

```bash
cd frontend
npx playwright install
cd ..
```

### Step 5: Create Test Fixtures

```bash
# Create fixtures directory
mkdir -p tests/e2e/fixtures

# Copy a sample image (or use any test image)
# Example: Copy from project samples if available
cp data/samples/receipt_001.jpg tests/e2e/fixtures/sample-image.jpg
# Or create a dummy image for testing
```

### Step 6: Verify Installation

```bash
# Run backend tests
uv run pytest tests/ -v

# Run frontend E2E tests
cd frontend
npx playwright test
cd ..
```

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │         React SPA (Vite)                            │  │
│  │  ┌──────────────┐  ┌──────────────┐               │  │
│  │  │   Pages      │  │  Components  │               │  │
│  │  ├──────────────┤  ├──────────────┤               │  │
│  │  │ Preprocessing│  │  Canvas      │               │  │
│  │  │ Inference    │  │  Controls    │               │  │
│  │  │ Comparison   │  │  Gallery     │               │  │
│  │  │ Command      │  │  Telemetry   │               │  │
│  │  └──────────────┘  └──────────────┘               │  │
│  │                                                     │  │
│  │  ┌──────────────────────────────────────────┐    │  │
│  │  │      Web Workers (pipelineWorker.ts)     │    │  │
│  │  │  - Auto Contrast                         │    │  │
│  │  │  - Gaussian Blur                         │    │  │
│  │  │  - Resize                                │    │  │
│  │  │  - Background Removal (rembg)            │    │  │
│  │  └──────────────────────────────────────────┘    │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                    HTTP/WebSocket
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  API Routers                         │  │
│  │  /api/commands    - Command Builder                 │  │
│  │  /api/inference   - Inference Service               │  │
│  │  /api/pipelines   - Preprocessing Pipeline          │  │
│  │  /api/evaluation  - Model Evaluation                │  │
│  │  /api/metrics     - Telemetry & Monitoring          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Lazy-Loaded Services                    │  │
│  │  - ConfigParser (Streamlit-era utilities)           │  │
│  │  - CommandBuilder                                   │  │
│  │  - Recommendation Service                           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
project-root/
├── frontend/                   # React SPA
│   ├── src/
│   │   ├── pages/              # Top-level page components
│   │   │   ├── Preprocessing.tsx
│   │   │   ├── Inference.tsx
│   │   │   ├── Comparison.tsx
│   │   │   └── Telemetry.tsx
│   │   ├── components/         # Reusable UI components
│   │   │   ├── preprocessing/  # Preprocessing-specific components
│   │   │   ├── inference/      # Inference-specific components
│   │   │   ├── gallery/        # Image gallery
│   │   │   └── telemetry/      # Telemetry dashboard
│   │   ├── workers/            # Web Worker implementation
│   │   │   ├── workerHost.ts   # Worker pool manager
│   │   │   ├── workerTelemetry.ts  # Telemetry integration
│   │   │   └── pipelineWorker.ts   # Worker logic
│   │   ├── api/                # API client modules
│   │   │   ├── commands.ts     # Command Builder API
│   │   │   ├── inference.ts    # Inference API
│   │   │   ├── pipelines.ts    # Preprocessing API
│   │   │   └── metrics.ts      # Telemetry API
│   │   ├── hooks/              # Custom React hooks
│   │   │   └── useWorkerTask.ts    # Worker task hook
│   │   └── utils/              # Utility functions
│   │       └── rembgRouting.ts     # Hybrid routing logic
│   ├── public/                 # Static assets
│   └── playwright.config.ts    # E2E test config
├── services/
│   └── playground_api/         # FastAPI backend
│       ├── app.py              # Application entry point
│       ├── routers/            # API route handlers
│       │   ├── command_builder.py
│       │   ├── inference.py
│       │   ├── pipeline.py
│       │   ├── evaluation.py
│       │   └── metrics.py
│       └── utils/
│           └── paths.py        # Path utilities
├── tests/
│   ├── e2e/                    # Playwright E2E tests
│   │   ├── preprocessing.spec.ts
│   │   ├── command-builder.spec.ts
│   │   ├── inference.spec.ts
│   │   └── comparison.spec.ts
│   └── perf/                   # Performance tests
│       └── test_api_startup.py
├── docs/                       # Documentation
│   ├── HIGH_PERFORMANCE_PLAYGROUND.md   # User guide
│   └── DEVELOPER_ONBOARDING.md          # This file
└── run_spa.py                  # Unified launcher
```

---

## Development Workflow

### Daily Workflow

1. **Pull latest changes**
   ```bash
   git pull origin main
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Start development servers**
   ```bash
   # Option 1: Unified launcher
   python run_spa.py

   # Option 2: Manual (for debugging)
   # Terminal 1:
   uvicorn services.playground_api.app:app --reload --port 8000
   # Terminal 2:
   cd frontend && npm run dev
   ```

4. **Make changes and test**
   ```bash
   # Run tests frequently
   npx playwright test  # E2E tests
   uv run pytest        # Backend tests
   ```

5. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: Add your feature description"
   git push origin feature/your-feature-name
   ```

6. **Create pull request**
   - Open PR on GitHub
   - Request review from team
   - Address feedback

### Git Workflow

**Branch Naming**:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/fixes

**Commit Messages**:
Follow conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `refactor:` - Code refactoring
- `test:` - Test changes
- `perf:` - Performance improvement

Example:
```bash
git commit -m "feat: Add confidence threshold slider to inference studio"
git commit -m "fix: Resolve worker queue depth calculation bug"
git commit -m "docs: Update telemetry API documentation"
```

---

## Code Standards

### TypeScript (Frontend)

**Line Length**: 100 characters

**Naming Conventions**:
- **Components**: PascalCase (`PreprocessingCanvas.tsx`)
- **Hooks**: camelCase with `use` prefix (`useWorkerTask.ts`)
- **Utilities**: camelCase (`rembgRouting.ts`)
- **Constants**: SCREAMING_SNAKE_CASE (`MAX_IMAGE_SIZE`)

**Type Safety**:
```typescript
// ✅ Good: Explicit types
interface WorkerTask {
  operation: string;
  imageBuffer: ArrayBuffer;
  params: Record<string, unknown>;
}

function submitTask(task: WorkerTask): Promise<WorkerResult> {
  // ...
}

// ❌ Bad: Implicit any
function submitTask(task): Promise<any> {
  // ...
}
```

**React Patterns**:
```typescript
// ✅ Good: Functional components with explicit props
interface PreprocessingCanvasProps {
  imageData: ImageData | null;
  operations: PreprocessingOperation[];
}

export function PreprocessingCanvas({
  imageData,
  operations,
}: PreprocessingCanvasProps): JSX.Element {
  // ...
}

// ❌ Bad: No prop types
export function PreprocessingCanvas(props) {
  // ...
}
```

### Python (Backend)

**Line Length**: 140 characters

**Naming Conventions**:
- **Functions**: snake_case (`get_metrics_summary`)
- **Classes**: PascalCase (`WorkerEvent`)
- **Constants**: SCREAMING_SNAKE_CASE (`MAX_METRICS_PER_TYPE`)

**Type Hints**:
```python
# ✅ Good: Type hints
def get_metrics_summary(hours: int = 1) -> MetricsSummary:
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    # ...
    return MetricsSummary(...)

# ❌ Bad: No type hints
def get_metrics_summary(hours=1):
    # ...
```

**Pydantic Models**:
```python
# ✅ Good: Descriptive field documentation
class PerformanceMetric(BaseModel):
    """Performance measurement for a task."""

    task_type: Literal["auto_contrast", "gaussian_blur", ...]
    duration_ms: float = Field(description="Task duration in milliseconds")
    success: bool = Field(default=True)

# ❌ Bad: No documentation
class PerformanceMetric(BaseModel):
    task_type: str
    duration_ms: float
    success: bool
```

---

## Testing

### E2E Tests (Playwright)

**Location**: `tests/e2e/`

**Running Tests**:
```bash
cd frontend

# Run all tests
npx playwright test

# Run specific test file
npx playwright test preprocessing.spec.ts

# Run in headed mode (see browser)
npx playwright test --headed

# Debug mode
npx playwright test --debug

# View report
npx playwright show-report
```

**Writing Tests**:
```typescript
import { test, expect } from '@playwright/test';

test.describe('Preprocessing Studio', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/preprocessing');
  });

  test('should process image with auto contrast', async ({ page }) => {
    // Upload image
    await page.locator('#image-upload').setInputFiles('tests/e2e/fixtures/sample-image.jpg');

    // Enable auto contrast
    await page.locator('text=Auto Contrast').click();

    // Wait for processing
    await page.waitForSelector('text=Processing time:');

    // Verify latency < 100ms
    const processingTime = await page.locator('text=Processing time:').textContent();
    const match = processingTime?.match(/(\d+\.\d+)ms/);
    if (match) {
      const latency = parseFloat(match[1]);
      expect(latency).toBeLessThan(100);
    }
  });
});
```

### Backend Tests (pytest)

**Location**: `tests/`

**Running Tests**:
```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/perf/test_api_startup.py -v

# Run with coverage
uv run pytest tests/ --cov=services --cov-report=html
```

**Writing Tests**:
```python
import pytest
from services.playground_api.routers.metrics import MetricsSummary

def test_metrics_summary_calculation():
    """Test that metrics summary aggregates correctly."""
    # Setup test data
    # ...

    # Call function
    summary = get_metrics_summary(hours=1)

    # Assertions
    assert summary.total_tasks >= 0
    assert 0.0 <= summary.cache_hit_rate <= 1.0
    assert summary.worker_queue_depth >= 0
```

---

## Debugging Tips

### Frontend Debugging

**React DevTools**:
1. Install React DevTools browser extension
2. Open DevTools → Components tab
3. Inspect component props and state

**Worker Debugging**:
```typescript
// Add console.logs in workerHost.ts
console.log('[WorkerPool] Task queued:', task.operation);
console.log('[WorkerPool] Queue depth:', this.taskQueue.length);
console.log('[WorkerPool] Available workers:', this.availableWorkers.length);
```

**Telemetry Inspection**:
```bash
# Check worker events
curl http://localhost:8000/api/metrics/events/recent?event_type=worker&limit=20

# Check metrics summary
curl http://localhost:8000/api/metrics/summary?hours=1 | jq
```

**Network Inspection**:
1. Open DevTools → Network tab
2. Filter by XHR
3. Inspect API request/response payloads

### Backend Debugging

**FastAPI Interactive Docs**:
```
http://localhost:8000/docs
```

**Uvicorn Debug Mode**:
```bash
uvicorn services.playground_api.app:app --reload --log-level debug
```

**Print Debugging**:
```python
# Add logging statements
import logging
logger = logging.getLogger(__name__)

@router.get("/schemas")
def list_schemas():
    logger.debug("Fetching schemas...")
    # ...
```

**API Testing with curl**:
```bash
# Test command schemas endpoint
curl http://localhost:8000/api/commands/schemas | jq

# Test inference preview
curl -X POST http://localhost:8000/api/inference/preview \
  -H "Content-Type: application/json" \
  -d '{"checkpoint_path":"checkpoints/model.pth","image_base64":"..."}' | jq
```

---

## Common Tasks

### Adding a New API Endpoint

1. **Create router function** in `services/playground_api/routers/<module>.py`:
   ```python
   @router.get("/new-endpoint")
   def new_endpoint() -> ResponseModel:
       # Implementation
       return ResponseModel(...)
   ```

2. **Register router** in `services/playground_api/app.py` (if new module):
   ```python
   from .routers import new_module
   app.include_router(new_module.router, prefix="/api/new", tags=["new"])
   ```

3. **Create API client** in `frontend/src/api/<module>.ts`:
   ```typescript
   export async function callNewEndpoint(): Promise<ResponseModel> {
     const response = await fetch(`${API_BASE}/api/new-endpoint`);
     return response.json();
   }
   ```

4. **Write tests**:
   - Backend: `tests/test_new_endpoint.py`
   - E2E: `tests/e2e/new-feature.spec.ts`

### Adding a New Worker Operation

1. **Add operation to worker** in `workers/pipelineWorker.ts`:
   ```typescript
   case 'new_operation':
     result = await performNewOperation(task.imageBuffer, task.params);
     break;
   ```

2. **Update API types** in `frontend/src/api/pipelines.ts`:
   ```typescript
   export type PipelineOperation = "auto_contrast" | "gaussian_blur" | ... | "new_operation";
   ```

3. **Add UI controls** in `frontend/src/components/preprocessing/ParameterControls.tsx`:
   ```typescript
   <div>
     <label>
       <input type="checkbox" checked={params.newOperation} ... />
       New Operation
     </label>
   </div>
   ```

4. **Write E2E test** in `tests/e2e/preprocessing.spec.ts`:
   ```typescript
   test('should process image with new operation', async ({ page }) => {
     // ...
   });
   ```

### Adding a New Page/Route

1. **Create page component** in `frontend/src/pages/NewPage.tsx`:
   ```typescript
   export function NewPage(): JSX.Element {
     return <div>New Page Content</div>;
   }
   ```

2. **Add route** in `frontend/src/App.tsx` or router config:
   ```typescript
   <Route path="/new-page" element={<NewPage />} />
   ```

3. **Add navigation link** in nav bar/menu

4. **Write E2E tests** in `tests/e2e/new-page.spec.ts`

---

## Resources

### Documentation

- [High-Performance Playground Guide](HIGH_PERFORMANCE_PLAYGROUND.md)
- [E2E Testing Guide](../tests/e2e/README.md)
- [FastAPI Startup Optimization](performance/fastapi-startup-optimization.md)

### External Resources

- [React Documentation](https://react.dev)
- [Vite Documentation](https://vitejs.dev)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Playwright Documentation](https://playwright.dev)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)

### Code Examples

- **Worker Pool Pattern**: `frontend/src/workers/workerHost.ts`
- **Lazy Imports (Python)**: `services/playground_api/routers/command_builder.py`
- **Hybrid Routing**: `frontend/src/utils/rembgRouting.ts`
- **Telemetry Integration**: `frontend/src/workers/workerTelemetry.ts`

---

## Getting Help

### Internal Resources

- Check existing documentation in `docs/`
- Review E2E tests for usage examples
- Ask team members in Slack/Discord

### External Resources

- Stack Overflow
- React/FastAPI communities
- GitHub issues (for library-specific problems)

---

## Next Steps

Now that you're set up, try these tasks to get familiar with the codebase:

1. **Run the playground** locally and explore all pages
2. **Run E2E tests** and watch them execute in headed mode
3. **Add a console.log** in a worker task and observe it in DevTools
4. **Make a small UI change** (e.g., button text) and see hot-reload in action
5. **Review a recent PR** to understand the review process

Welcome to the team! 🎉

---

**Last Updated**: 2025-11-18
**Version**: 1.0
**Maintained By**: Development Team
