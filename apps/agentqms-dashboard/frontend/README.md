# AgentQMS Dashboard - Frontend

**React + TypeScript dashboard for AgentQMS framework management**

## Overview

Modern web interface for AgentQMS artifact management, compliance checking, and quality monitoring. Built with React 19.2, TypeScript, and Vite.

## Features

### 🎯 Artifact Generator
AI-powered creation of AgentQMS artifacts (implementation plans, assessments, audits, bug reports) with automatic frontmatter generation and naming convention enforcement.

### 🔍 Framework Auditor
Dual-mode validation system:
- **AI Analysis**: Gemini-powered document quality assessment
- **Tool Runner**: Direct execution of Python validation tools (validate, compliance, boundary checks)

### 📊 Strategy Dashboard
Framework health visualization with:
- Compliance metrics and progress tracking
- AI architectural recommendations
- Recharts-based data visualization

### 🔗 Integration Hub
Real-time system monitoring:
- Tracking database status
- System health checks
- Backend connectivity verification

### 🌐 Context Explorer
Artifact relationship visualization (planned - data structure ready)

### 📚 Librarian
Document discovery and management interface

### 🔗 Reference Manager
Link migration and resolution tools for maintaining documentation integrity

## Quick Start

```bash
# Install dependencies
npm install

# Set environment variable
# Create .env.local with:
GEMINI_API_KEY=your_api_key_here

# Start development server
npm run dev

# Access at http://localhost:3000
```

## Development

### Available Scripts

```bash
npm run dev      # Start Vite dev server (port 3000)
npm run build    # Build production bundle
npm run preview  # Preview production build
npm test         # Run tests
npm run lint     # Lint code
```

### Using Make Commands

The root Makefile provides convenient commands:

```bash
make dev-frontend        # Start frontend server
make install-frontend    # Install dependencies
make test-frontend       # Run tests
make lint-frontend       # Lint code
make build               # Build production bundle
```

## Project Structure

```
frontend/
├── components/          # React components
│   ├── ArtifactGenerator.tsx
│   ├── FrameworkAuditor.tsx
│   ├── StrategyDashboard.tsx
│   ├── IntegrationHub.tsx
│   ├── ContextExplorer.tsx
│   ├── Librarian.tsx
│   ├── ReferenceManager.tsx
│   ├── Settings.tsx
│   └── ...
├── services/            # API integration
│   ├── aiService.ts     # Gemini API client
│   └── bridgeService.ts # Backend API client
├── config/              # Configuration
│   └── constants.ts     # App constants
├── docs/                # Documentation
│   ├── architecture/    # Architecture docs
│   ├── api/             # API specifications
│   ├── plans/           # Implementation plans
│   └── ...
├── App.tsx              # Main app component
├── index.tsx            # Entry point
└── vite.config.ts       # Vite configuration
```

## Configuration

### Vite Proxy Setup

The Vite dev server proxies `/api` requests to the backend:

```typescript
// vite.config.ts
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      secure: false,
    }
  }
}
```

### API Integration

Two service layers:
- **aiService**: Gemini API for AI-powered features
- **bridgeService**: Backend API for file system, tools, and tracking

```typescript
// Usage example
import { bridgeService } from './services/bridgeService';

const result = await bridgeService.executeTool('validate', {});
const status = await bridgeService.getTrackingStatus('all');
```

## Components

### Layout & Navigation
- **Layout.tsx**: Main app shell with sidebar navigation
- **DashboardHome.tsx**: Landing page with quick actions

### Core Features
- **ArtifactGenerator.tsx**: AI-powered artifact creation
- **FrameworkAuditor.tsx**: Validation tools (AI + Python)
- **StrategyDashboard.tsx**: Metrics and recommendations
- **IntegrationHub.tsx**: System status monitoring

### Utilities
- **Settings.tsx**: App configuration (API keys, providers)
- **TrackingStatus.tsx**: Tracking DB status component
- **ErrorBoundary.tsx**: Error handling wrapper

## API Endpoints Used

```typescript
// Backend API (via bridgeService)
GET  /api/v1/health              // System health
GET  /api/v1/tracking/status     // Tracking DB status
POST /api/v1/tools/exec          // Execute validation tools
GET  /api/v1/artifacts/list      // List artifacts
POST /api/v1/artifacts           // Create artifact
GET  /api/v1/compliance/check    // Compliance status

// AI Service (Gemini)
POST https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent
```

## Styling

- **Tailwind CSS** (CDN): Used for rapid development
  - Warning suppression configured in `index.html`
  - Consider PostCSS setup for production

- **Color Scheme**: Dark theme
  - Primary: Blue (#3b82f6)
  - Background: Slate (#0f172a, #1e293b)
  - Text: White/Slate shades

## State Management

Currently using React hooks for local state:
- `useState` for component state
- `useEffect` for side effects
- No global state library (Redux, Zustand) needed yet

## Troubleshooting

### Console Warnings

See [../CONSOLE_WARNINGS_RESOLUTION.md](../CONSOLE_WARNINGS_RESOLUTION.md) for:
- Tailwind CDN warning fix
- Recharts chart sizing fix
- React DevTools suggestion

### Backend Connection

If you see connection errors:
1. Verify backend is running: `lsof -i :8000`
2. Check proxy config in `vite.config.ts`
3. Ensure CORS is configured in backend

### Tool Execution Shows "(No output)"

This was fixed - ensure:
- Backend is using correct response format (`success`, `output`, `error`)
- Frontend is reading `result.output` not `result.stdout`

## Testing

```bash
npm test                 # Run tests
npm run test:coverage    # Coverage report
```

Current status: Manual testing complete, automated tests pending.

## Build & Deploy

```bash
# Build production bundle
npm run build

# Output: dist/
# Contains optimized static files

# Preview build locally
npm run preview
```

**Note**: Update Tailwind from CDN to PostCSS for production.

## Documentation

- **Architecture**: [docs/architecture/](docs/architecture/)
- **API Contracts**: [docs/api/](docs/api/)
- **Implementation Plans**: [docs/plans/](docs/plans/)
- **Progress Tracking**: [docs/meta/](docs/meta/)

## Contributing

Follow AgentQMS conventions:
- TypeScript strict mode
- Component-level documentation
- Integration with AgentQMS validation tools

---

**Stack**: React 19.2 • TypeScript 5.6 • Vite 7.2 • Tailwind CSS • Recharts 3.5

