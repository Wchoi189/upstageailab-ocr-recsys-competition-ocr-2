---
type: reference
component: design_system
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Playground Design System

**Purpose**: Component and visual language for Albumentations/Upstage-inspired playground; defines tokens, layouts, components for SPA scaffolding.

---

## Foundation Tokens

| Category | Token | Value | Purpose |
|----------|-------|-------|---------|
| **Grid** | Columns | 12-column fluid | Layout structure |
| **Grid** | Gutters | 24px | Column spacing |
| **Grid** | Max padding | 72px | Content padding |
| **Spacing** | Base unit | 4px | Spacing scale: 4, 8, 12, 16, 24, 32, 40, 64 |

---

## Typography

| Type | Font | Size/Height | Weight | Usage |
|------|------|-------------|--------|-------|
| **Display** | Inter | 24/32 | Semi-bold | Section headings |
| **Body** | Inter | 14/20 | Regular | Main text |
| **Caption** | Inter | 12/16 | Regular | Secondary text |
| **Mono** | JetBrains Mono | N/A | Regular | Command console, logs |

---

## Color Tokens

| Token | Value | Usage |
|-------|-------|-------|
| `surface.base` | `#0F1115` | Base background |
| `surface.raise` | `#161922` | Elevated surfaces |
| `border.default` | `rgba(255,255,255,0.08)` | Default borders |
| `accent.primary` | `#5AE4A7` | Interactive elements (sliders, buttons) |
| `accent.warning` | `#FFB155` | Warning states |
| `accent.danger` | `#FF6B6B` | Error states |
| `text.primary` | `#F5F7FA` | Primary text |
| `text.secondary` | `#B4BCD0` | Secondary text |

---

## Elevation

| Surface | Shadow | Alpha |
|---------|--------|-------|
| **Card** | `0 8px 30px rgba(0,0,0,0.35)` | N/A |
| **Floating panels** | Blurred backdrop | 0.6 |

---

## Layout Patterns

| Pattern | Structure | Notes |
|---------|-----------|-------|
| **Playground Canvas** | Left (controls, max 420px) + Center (before/after canvas) + Right (insights rail) | Sticky header for dataset/checkpoint selectors |
| **Command Console Drawer** | Slides from bottom, 480px height | Tabs: Generated Command, Execution Log, Parameters (JSON) |
| **Worker Queue HUD** | Floating widget, bottom-right | Active jobs, CPU usage, route (client vs backend) |
| **Document OCR Showcase** | Multi-panel view | Page preview, layout tree, key-value tables |

---

## Component Library

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `ImageUploader` | Drag-drop + paste, previews, metadata pill | Resolution, size; rembg toggle, dataset quick-pick |
| `SliderField` | Debounced slider (<100ms response) | `onCommit` + `onLiveChange` events for workers |
| `SegmentedControl` | Mode selectors (single/batch) | Keyboard navigable |
| `MetricCard` | Value, delta, sparkline | KPIs (hmean, latency) |
| `CanvasDiff` | WebGL before/after, polygon overlays, heatmaps | Streaming frames, zoom/pan gestures |
| `CommandPreview` | Syntax-highlighted CLI, copy/download buttons | Pulls from `/api/commands/build` |
| `WorkerStatusList` | Active worker tasks table | Backend routes, durations, instrumentation |
| `ComparisonTimeline` | Horizontal timeline (Upstage-style) | Parameter sets, results badges |

---

## Accessibility & Responsiveness

| Requirement | Implementation |
|-------------|----------------|
| **Contrast ratio** | Minimum 4.5:1 for primary text |
| **Keyboard focus** | 2px `accent.primary` ring |
| **Breakpoints** | 1280 (desktop), 1024 (tablet, collapses insights), <768 (stacked layout, condensed console) |

---

## Dependencies

| Component | Dependencies |
|-----------|-------------|
| **Design tokens** | CSS custom properties from `packages/design-tokens` |
| **Canvas** | WebGL, polygon rendering |
| **Workers** | Worker blueprint instrumentation |

---

## Constraints

- Memoized React components for high-frequency controls (sliders, switches)
- Skeleton states for every panel (avoid flash during worker warm-up)
- Token implementation via CSS custom properties

---

## Backward Compatibility

**Status**: New system (v1.0)

**Breaking Changes**: N/A (new implementation)

---

## References

- [Worker Blueprint](worker-blueprint.md)
- [High Performance Playground](high-performance-playground.md)
- [Testing Observability](testing-observability.md)
