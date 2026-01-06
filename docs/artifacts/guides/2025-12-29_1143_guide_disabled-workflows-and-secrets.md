---
ads_version: "1.0"
type: "guide"
category: "infrastructure"
status: "active"
version: "1.0"
tags: ['ci', 'workflows', 'secrets', 'github-actions']
title: "Guide: Disabled Workflows & Required Secrets"
date: "2025-12-29 11:43"
branch: "main"
---

# Guide: Disabled Workflows & Required Secrets

The following workflows have been temporarily disabled to reduce CI noise and stabilize the quality gate. They can be re-enabled by renaming them from `.yml.disabled` to `.yml` and ensuring the required secrets are configured.

## 1. AI CI Healer Integration
**File:** `.github/workflows/ai-ci-healer-integration.yml.disabled`
**Purpose:** Automatically attempts to fix CI failures using an LLM.
**Required Secrets:**
- `OPENAI_API_KEY`: Required for the LLM to process logs and generate fixes.

## 2. Claude Issue Helper
**File:** `.github/workflows/claude-issue-helper.yml.disabled`
**Purpose:** Assists with issue triage and documentation using Claude.
**Required Secrets:**
- `CLAUDE_CODE_OAUTH_TOKEN`: Required for authentication with the Claude Code API.

## 3. AgentQMS Validation (Redundant)
**File:** `.github/workflows/agentqms-validation.yml.disabled`
**Purpose:** Validates artifact compliance.
**Status:** This workflow is redundant with `agentqms-ci.yml`, which provides a more comprehensive validation suite. It should remain disabled unless a lightweight check is specifically needed.

## How to Re-enable
1. Rename the file (remove `.disabled` extension)
2. Configure the required secrets in GitHub repo settings
3. Push the change to enable the workflow

## Current Status
All workflows are intentionally disabled. Enable only when needed and secrets are available.
