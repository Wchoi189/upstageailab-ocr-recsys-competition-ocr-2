---
title: "AgentQMS Frontend Architecture Guidelines"
type: architecture
status: complete
created: 2025-12-08 14:30 (KST)
updated: 2025-12-08 14:30 (KST)
phase: 2
priority: high
tags: [architecture, frontend, react, patterns, separation-of-concerns]
---

# AgentQMS Frontend Architecture Guidelines

## 1. Core Philosophy: Separation of Concerns
To avoid "spaghetti code" and massive files, we enforce a strict **Container/Presentation** pattern.

### The Rule of Three
1.  **Logic (Hooks):** State management, API calls, and effects go into custom hooks (`useMyFeature.ts`).
2.  **Structure (Container):** The Parent component handles layout and passes data from hooks to children.
3.  **Visuals (Presentation):** Dumb components receive data via props and render UI. They have NO business logic.

---

## 2. Directory Structure
```
src/
├── components/          # Reusable UI elements (Buttons, Cards, Inputs)
├── features/            # Feature-specific modules
│   ├── auditor/         # Feature Folder
│   │   ├── AuditorContainer.tsx  # Main Entry
│   │   ├── AuditorView.tsx       # UI Render
│   │   └── useAuditor.ts         # Logic/State
│   └── integration/
├── services/            # Pure API/Backend Logic (No React code)
├── types/               # TypeScript Definitions
└── utils/               # Helper functions (Dates, Strings)
```

## 3. State Management Standards
-   **Local State:** Use `useState` for UI state (isModalOpen, activeTab).
-   **Complex State:** Use `useReducer` if a component has >3 `useState` calls that change together.
-   **Global State:** Use React Context ONLY for:
    -   User Preferences (Theme, Settings)
    -   Authentication
    -   Notification Toasts
-   **Data State:** Do not store "database" data in Redux/Context if it can be fetched fresh. Use `swr` or `react-query` patterns (or our `aiService`).

## 4. Coding Standards

### A. Component Composition over Monoliths
**Bad:**
```tsx
const Dashboard = () => {
  // 500 lines of mixed chart code, button logic, and api calls
  return <div>...</div>
}
```

**Good:**
```tsx
const Dashboard = () => {
  const { metrics, loading } = useDashboardMetrics(); // Logic extracted
  return (
    <DashboardLayout>
       <MetricCards data={metrics} />
       <ActivityFeed />
    </DashboardLayout>
  )
}
```

### B. Scalability Rules
1.  **Strict Typing:** No `any`. Define interfaces in `types.ts` or feature-local `types.ts`.
2.  **Constants:** Do not hardcode magic strings or colors. Use `constants.ts` or Tailwind classes.
3.  **Error Boundaries:** Every major feature must handle its own errors and not crash the whole app.

## 5. Performance
-   **Memoization:** Wrap expensive calculations in `useMemo`.
-   **Callback Stability:** Wrap functions passed to children in `useCallback` to prevent re-renders.
-   **Lazy Loading:** Use `React.lazy` for major route components (Dashboard, Auditor, Settings).
