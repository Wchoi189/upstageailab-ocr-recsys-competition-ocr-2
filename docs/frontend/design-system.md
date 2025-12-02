---
title: "Playground Design System"
status: "draft"
---

# Playground Design System

This system defines the initial component/visual language for the Albumentations/Upstage-inspired playground. It will evolve with usability studies, but these primitives unblock SPA scaffolding immediately.

## Foundations

- **Grid**: 12-column fluid grid with 24px gutters, 72px max content padding.
- **Spacing scale**: 4px base unit (4, 8, 12, 16, 24, 32, 40, 64).
- **Typography**:
  - Display: Inter 24/32 semi-bold (section headings).
  - Body: Inter 14/20 regular; captions 12/16.
  - Mono: JetBrains Mono for command console/log output.
- **Color tokens**:
  - `surface.base`: `#0F1115`
  - `surface.raise`: `#161922`
  - `border.default`: `rgba(255,255,255,0.08)`
  - `accent.primary`: `#5AE4A7` (interactive sliders/buttons)
  - `accent.warning`: `#FFB155`
  - `accent.danger`: `#FF6B6B`
  - `text.primary`: `#F5F7FA`
  - `text.secondary`: `#B4BCD0`
- **Elevation**:
  - Card: `0 8px 30px rgba(0,0,0,0.35)`
  - Floating panels (worker queue, logs) use blurred backdrop with 0.6 alpha.

## Layout Patterns

1. **Playground Canvas**
   - Left: persistent controls column (max 420px) with sticky header for dataset + checkpoint selectors.
   - Center: before/after canvas with split-view toggle and zoom controls.
   - Right: Insight rail for metrics, worker statuses, command diff.
2. **Command Console Drawer**
   - Slides from bottom; 480px height.
   - Tabs: Generated Command, Execution Log, Parameters (JSON).
3. **Worker Queue HUD**
   - Floating widget in bottom-right summarizing active jobs, CPU usage, route (client vs backend).
4. **Document OCR Showcase**
   - Multi-panel view (similar to Upstage console) combining page preview, detected layout tree, extracted key-value tables.

## Component Library

| Component | Description | Notes |
| --- | --- | --- |
| `ImageUploader` | Drag-drop + paste support, previews, metadata pill (resolution, size). | Accepts rembg toggle, dataset quick-pick actions. |
| `SliderField` | Debounced slider with `<100 ms` response requirement. | Emits `onCommit` + `onLiveChange` events for workers. |
| `SegmentedControl` | Mode selectors (single/batch). | Keyboard navigable. |
| `MetricCard` | Displays value, delta, sparkline. | Used for KPIs (hmean, latency). |
| `CanvasDiff` | WebGL canvas showing before/after, polygon overlays, heatmaps. | Must support streaming frames + zoom/pan gestures. |
| `CommandPreview` | Syntax-highlighted CLI string with copy/download buttons. | Pulls metadata from `/api/commands/build`. |
| `WorkerStatusList` | Table of active worker tasks, backend routes, durations. | Feeds from worker blueprint instrumentation. |
| `ComparisonTimeline` | Horizontal timeline of configurations, similar to Upstage "Document OCR" timeline. | Shows parameter sets + results badges. |

## Accessibility & Responsiveness

- Minimum contrast ratio 4.5:1 for primary text.
- Keyboard focus ring: `2px accent.primary`.
- Breakpoints: 1280 (desktop), 1024 (tablet, collapses insights), <768 (stacked layout with condensed console).

## Visual Inspirations

- Albumentations AutoContrast playground for immediate preview feedback.
- Upstage Playground (Document OCR & Parsing) for layout overlays, three-panel document breakdown.

## Implementation Notes

- Implement tokens via CSS custom properties exported from `packages/design-tokens`.
- Wrap high-frequency controls (sliders, switches) with memoized React components to avoid re-renders when worker events stream back.
- Provide skeleton states for every panel to avoid flash during worker warm-up.
