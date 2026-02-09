# ADR 003: POSIX-Only Platform Support

**Date**: 2026-02-07
**Status**: Accepted
**Decider**: User

---

## Context

Need to decide platform support scope. Windows support adds significant path handling complexity.

## Decision

**Support POSIX platforms only (Linux, macOS). No Windows support.**

## Rationale

> "Use POSIX paths. Windows support makes this much more complex."

- Path handling is simpler with POSIX conventions
- Primary development environment is Linux (Docker)
- Windows support requires extensive testing and edge case handling
- User base is primarily Linux/macOS

## Consequences

**Positive:**
- **Simpler code**: No `pathlib` vs `os.path` vs Windows quirks
- **Faster development**: No cross-platform testing needed
- **Fewer bugs**: POSIX paths are consistent

**Negative:**
- **Limited reach**: Windows users cannot use `aqms`
- Mitigation: Document WSL2 as workaround for Windows users

## Technical Details

### Path Handling
```python
# Always use pathlib with POSIX assumptions
from pathlib import Path

# Safe to use forward slashes
project_root = Path.cwd() / ".aqms"

# No need for os.path.join or handling backslashes
```

### CI/CD
- Test only on Linux (Ubuntu) and macOS runners
- No Windows runners in GitHub Actions

### Documentation
- Clearly state "Linux/macOS only" in README
- Suggest WSL2 for Windows users who want to use tool
