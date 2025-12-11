---
title: "Design: API Architecture"
type: design
status: active
created: 2025-12-01 13:00 (KST)
category: architecture
tags: [design, api]
---

# Design: API Architecture

## Overview
RESTful API for AgentQMS artifact management.

## Endpoints

### Artifacts
- GET /api/v1/artifacts/list
- POST /api/v1/artifacts/create
- GET /api/v1/artifacts/{id}

### Tools
- POST /api/v1/tools/exec

### System
- GET /api/v1/health
