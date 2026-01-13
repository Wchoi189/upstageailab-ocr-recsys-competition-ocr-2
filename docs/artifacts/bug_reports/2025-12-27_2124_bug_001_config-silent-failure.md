---
ads_version: "1.0"
type: "bug_report"
category: "troubleshooting"
status: "completed"
severity: "critical"
version: "1.0"
tags: ['configuration', 'hydra', 'omegaconf', 'bugfix', 'refactor']
title: "Silent Configuration Failures via isinstance(dict)"
date: "2025-12-27 21:24 (KST)"
branch: "main"
summary: "OmegaConf DictConfig objects were failing isinstance(obj, dict) checks, leading to silent failures where default configurations were used instead of user-provided ones, or validation logic was skipped entirely."
---

# Details

The issue was pervasive across `ocr/models`, `ocr/inference`, `ocr/datasets`, and `ocr/utils`. By replacing brittle type checks with centralized utilities, we have hardened the application against this class of errors.
