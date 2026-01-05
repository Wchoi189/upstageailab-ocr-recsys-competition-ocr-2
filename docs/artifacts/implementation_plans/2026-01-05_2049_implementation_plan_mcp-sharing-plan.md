---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['implementation', 'plan', 'development']
title: "MCP Sharing Implementation Plan"
date: "2026-01-05 20:49 (KST)"
branch: "main"
description: "Plan to reorganize and share MCP configurations between AI agents."
---

# Implementation Plan -# Sharing MCP Servers Between AI Agents

The goal is to centralize MCP (Model Context Protocol) server definitions into a shared directory within the workspace and provide a synchronization tool to update various AI agents (Gemini, Claude, etc.) with these shared servers.

Currently, MCP servers are defined in agent-specific hidden directories (like `~/.gemini/antigravity/mcp_config.json`), making them difficult to manage across different agents. This plan moves the server definitions to `scripts/mcp/` and provides a `sync_configs.py` script to automate the configuration of all local agents.

## User Review Required

> [!IMPORTANT]
> This plan will modify your Gemini MCP configuration and potentially Claude's configuration by adding a "unified_project" server entry. It will also move the `mcp/` directory from the root to `scripts/mcp/`.

## Proposed Changes

### Configuration
- [ ] Change 1

### Code
- [ ] Change 1

## Verification Plan

### Automated Tests
- [ ] `pytest ...`

### Manual Verification
- [ ] Verify ...
