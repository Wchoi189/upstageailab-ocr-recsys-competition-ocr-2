---
title: "Timestamps Utility"
tier: "1-critical"
priority: "highest"
key_benefit: "KST timezone handling + consistent formatting"
ai_facing: true
---

# Timestamps Utility — KST Timestamp Handling

## Summary

**What**: Creates and formats timestamps in KST (Korea Standard Time)
**Why**: Consistent timezone handling for artifact metadata
**When**: Creating artifact metadata, logging, timestamps
**Where**: `AgentQMS/tools/utils/timestamps.py`

## Import

```python
from AgentQMS.tools.utils.timestamps import (
    get_kst_timestamp,
    format_kst,
    get_timestamp_age,
)
```

## API Reference

### get_kst_timestamp() → datetime

Get the current timestamp in KST timezone.

```python
from AgentQMS.tools.utils.timestamps import get_kst_timestamp

now = get_kst_timestamp()
# Returns: datetime object with KST timezone
```

**Returns**: `datetime.datetime` object (timezone-aware, KST)

**Use for**: Getting current time in KST

---

### format_kst(timestamp: datetime, format_str: str) → str

Format a KST timestamp as a string.

```python
from AgentQMS.tools.utils.timestamps import get_kst_timestamp, format_kst

now = get_kst_timestamp()
formatted = format_kst(now, "%Y-%m-%d %H:%M:%S")
# Returns: "2026-01-11 14:30:45"
```

**Parameters**:
- `timestamp` (datetime): Timestamp to format
- `format_str` (str): Python strftime format string

**Returns**: Formatted string

**Common format strings**:

| Format | Example | Use Case |
|--------|---------|----------|
| `"%Y-%m-%d %H:%M:%S"` | `2026-01-11 14:30:45` | ISO-like (human readable) |
| `"%Y%m%d_%H%M%S"` | `20260111_143045` | Filenames |
| `"%Y-%m-%d"` | `2026-01-11` | Date only |
| `"%H:%M:%S"` | `14:30:45` | Time only |
| `"%Y-%m-%dT%H:%M:%S"` | `2026-01-11T14:30:45` | ISO 8601 |

---

### get_timestamp_age(timestamp: datetime) → float

Calculate how old a timestamp is (in hours).

```python
from AGentQMS.tools.utils.timestamps import get_kst_timestamp, get_timestamp_age

old_time = get_kst_timestamp()
# ... do something ...
age_hours = get_timestamp_age(old_time)
# Returns: 0.5 (30 minutes old)
```

**Returns**: Age in hours (float)

**Use for**: Checking if data is stale, calculating cache validity

---

## Usage Examples

### Example 1: Artifact Metadata Timestamp

```python
from AgentQMS.tools.utils.timestamps import get_kst_timestamp, format_kst

timestamp = get_kst_timestamp()
metadata = {
    'created_at': format_kst(timestamp, "%Y-%m-%d %H:%M:%S"),
    'created_at_iso': format_kst(timestamp, "%Y-%m-%dT%H:%M:%S"),
}
```

### Example 2: Timestamped Filename

```python
from AgentQMS.tools.utils.timestamps import get_kst_timestamp, format_kst
from AGentQMS.tools.utils.paths import get_artifacts_dir
import os

timestamp = get_kst_timestamp()
filename = f"report_{format_kst(timestamp, '%Y%m%d_%H%M%S')}.md"
filepath = os.path.join(get_artifacts_dir(), filename)
# → "/path/to/docs/artifacts/report_20260111_143045.md"
```

### Example 3: Check Data Freshness

```python
from AGentQMS.tools.utils.timestamps import get_kst_timestamp, get_timestamp_age

data_timestamp = get_kst_timestamp()
# ... use data ...

age = get_timestamp_age(data_timestamp)
if age > 24:  # Older than 24 hours
    print("Data is stale, refresh recommended")
else:
    print(f"Data is {age:.1f} hours old")
```

### Example 4: Log with Timestamp

```python
from AGentQMS.tools.utils.timestamps import get_kst_timestamp, format_kst

timestamp = get_kst_timestamp()
log_entry = f"[{format_kst(timestamp, '%H:%M:%S')}] Process completed"
print(log_entry)
# → "[14:30:45] Process completed"
```

## Timezone Details

### Why KST?

```
Project Location: South Korea
Standard Timezone: KST (UTC+9)
Daylight Saving: Not observed in Korea
```

**Benefits**:
- ✅ Consistent across team (all in KST)
- ✅ Avoids timezone confusion
- ✅ Matches project's timezone
- ✅ No DST surprises

### Timezone-Aware vs Naive

```python
from datetime import datetime
from AGentQMS.tools.utils.timestamps import get_kst_timestamp

# WRONG (naive timezone)
naive = datetime.now()
# → no timezone info, ambiguous

# CORRECT (aware timezone)
aware = get_kst_timestamp()
# → includes KST timezone, unambiguous
```

**Always use**: `get_kst_timestamp()` for timezone-aware timestamps

---

## Format String Reference

### Common Strftime Codes

| Code | Meaning | Example |
|------|---------|---------|
| `%Y` | 4-digit year | 2026 |
| `%m` | Month (01-12) | 01 |
| `%d` | Day (01-31) | 11 |
| `%H` | Hour (00-23) | 14 |
| `%M` | Minute (00-59) | 30 |
| `%S` | Second (00-59) | 45 |
| `%A` | Full weekday | Saturday |
| `%B` | Full month | January |

### Format Examples

```python
from AGentQMS.tools.utils.timestamps import get_kst_timestamp, format_kst

ts = get_kst_timestamp()

# Human-readable
format_kst(ts, "%A, %B %d, %Y")
# → "Saturday, January 11, 2026"

# ISO-like
format_kst(ts, "%Y-%m-%d %H:%M:%S")
# → "2026-01-11 14:30:45"

# Filename-safe
format_kst(ts, "%Y%m%d_%H%M%S")
# → "20260111_143045"

# Compact
format_kst(ts, "%m/%d %H:%M")
# → "01/11 14:30"
```

## Integration Examples

### With Paths and ConfigLoader

```python
from AGentQMS.tools.utils.timestamps import get_kst_timestamp, format_kst
from AGentQMS.tools.utils.paths import get_artifacts_dir
from AGentQMS.tools.utils.config_loader import ConfigLoader
import os

# Load config
config = ConfigLoader().load('configs/train.yaml')

# Create artifact with timestamp
timestamp = get_kst_timestamp()
artifact_file = f"training_result_{format_kst(timestamp, '%Y%m%d')}.md"
artifact_path = os.path.join(get_artifacts_dir(), artifact_file)

# Metadata
metadata = {
    'timestamp': format_kst(timestamp, "%Y-%m-%d %H:%M:%S"),
    'config': config,
}
```

### With Git Info

```python
from AGentQMS.tools.utils.timestamps import get_kst_timestamp, format_kst
from AGentQMS.tools.utils.git import get_current_branch, get_commit_hash

timestamp = get_kst_timestamp()
metadata = {
    'created_at': format_kst(timestamp, "%Y-%m-%d %H:%M:%S"),
    'branch': get_current_branch(),
    'commit': get_commit_hash(),
}
```

## Common Mistakes

### ❌ Using Naive Timestamps

```python
from datetime import datetime

# WRONG - no timezone info
naive = datetime.now()
# Problem: Ambiguous (local time vs UTC?)
```

### ✅ Use KST Timestamp Utility

```python
from AGentQMS.tools.utils.timestamps import get_kst_timestamp

# CORRECT - explicit KST timezone
aware = get_kst_timestamp()
# Clear: this is KST
```

---

### ❌ Hardcoding Format Strings

```python
# WRONG - format embedded in code
timestamp_str = "2026-01-11T14:30:45"
```

### ✅ Generate with Utility

```python
from AGentQMS.tools.utils.timestamps import get_kst_timestamp, format_kst

# CORRECT - generated dynamically
timestamp_str = format_kst(get_kst_timestamp(), "%Y-%m-%dT%H:%M:%S")
```

---

## Testing

```bash
# Run timestamps utility tests
pytest tests/utils/test_timestamps.py -v

# Test KST timezone
pytest tests/utils/test_timestamps.py::test_kst_timezone -v

# Test formatting
pytest tests/utils/test_timestamps.py::test_format_kst -v
```

## Key Takeaways

✅ **Use get_kst_timestamp()** for all current timestamps
✅ **Format with format_kst()** for string representation
✅ **KST explicit** (clear which timezone we're using)
✅ **Age calculations** for checking data freshness

❌ **Don't use** `datetime.datetime.now()` (naive timezone)
❌ **Don't hardcode** timezone offsets
❌ **Don't mix** timezone-aware and naive timestamps

## Reference

**Source**: `AGentQMS/tools/utils/timestamps.py`
**Tests**: `tests/utils/test_timestamps.py`
**Status**: ✅ Production-ready
**Timezone**: KST (UTC+9)
**Last Updated**: 2026-01-11
