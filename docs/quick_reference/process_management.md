# Streamlit Process Management

> **📖 This is a quick reference wrapper. For the complete documentation, see [Process Management Guide](../maintainers/process_management.md).**

This document provides quick access to process management documentation. The authoritative version with full details is located at [`docs/maintainers/process_management.md`](../maintainers/process_management.md).

## Quick Links

- **[Full Documentation →](../maintainers/process_management.md)** - Complete process management guide
- **Streamlit Debugging Protocol** - Debugging protocol for agents

## Quick Reference

```bash
# Start UI with full logging (MANDATORY for debugging)
make serve-inference-ui

# View logs immediately when errors occur
make logs-inference-ui

# Follow logs in real-time during reproduction
make logs-inference-ui -- --follow

# Document everything in context logs
make context-log-start LABEL="streamlit_debug"
```

---

**For complete documentation, troubleshooting, and detailed examples, please see [maintainers/process_management.md](../maintainers/process_management.md).**
