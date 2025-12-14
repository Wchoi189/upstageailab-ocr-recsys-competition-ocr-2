# Progress Tracker

This directory contains the state, progress, and documentation for the project.

## Structure

### 01-status/
The single source of truth for the project's current state.
- [Current State](01-status/current-state.md): What works, what doesn't, and environment details.
- [Roadmap](01-status/roadmap.md): High-level project phases.

### 02-work-logs/
Daily or session-based logs of work performed.
- [Latest Session](02-work-logs/2025-11-21-session-01.md)

### 03-tasks/
Task management.
- [Backlog](03-tasks/backlog.md): Tasks waiting to be picked up.
- [In Progress](03-tasks/in-progress.md): Active tasks.
- [Completed](03-tasks/completed/): Archive of finished work.

### 04-decisions/
Architecture Decision Records (ADRs). Significant technical decisions are recorded here.

### 05-components/
Component-specific documentation (Backend, Frontend, Database, JNI).

## Workflow
1. **Start Session:** Read `01-status/current-state.md`.
2. **Pick Task:** Move item from `backlog.md` to `in-progress.md`.
3. **Work:** Update `in-progress.md` with notes.
4. **End Session:**
    - Move task to `completed/` if done.
    - Create new entry in `02-work-logs/`.
    - Update `01-status/current-state.md`.
