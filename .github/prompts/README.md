# Prompt Library

This directory contains reusable system prompts and agent personas for the OCR/RecSys project. Use these files to context-load AI sessions for specific tasks.

## 📂 Organization Strategy

We categorize prompts by their primary function:

### 🎭 Roles (Personas)
*Define the AI's perspective and expertise level.*
- `role-cv-engineer.prompt.md`: Specialist in CV experiments, metric tracking, and hypothesis testing.
- `role-mlops-arch.prompt.md`: Layouts high-level system architecture, pipelines, and infrastructure.

### 🔍 Audits & QA
*Checklists and strict verification rules.*
- `audit-performance.prompt.md`: Checks for startup latency, eager imports, and async blocking.
- `audit-code-quality.prompt.md`: Enforces linting, typing, and style rules.
- `audit-security.prompt.md`: (New) Checklist for input validation and secret leaks.

### 🛠️ Tasks & Workflows
*Step-by-step guides for specific activities.*
- `task-experiment.prompt.md`: General guide for running and tracking experiments.
- `task-requirements.prompt.md`: Generates minimal, machine-parseable specs.
- `task-testing.prompt.md`: (New) Guidelines for reliable, fast pytest suites.
- `task-refactor.prompt.md`: (New) Strategies for safe code modernization.
- `task-writing.prompt.md`: (New) Standards for concise documentation.

### 🎨 Standards
*Formatting and style constraints.*
- `style-api.prompt.md`: Defines API contract styles and conventions.

## usage
Load these files into your context when starting a relevant task.
Example:
> "Read `.github/prompts/performance-optimization.prompt.md` and audit `main.py`."
