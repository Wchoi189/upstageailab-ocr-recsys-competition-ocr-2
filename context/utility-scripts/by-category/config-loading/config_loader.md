---
title: "ConfigLoader Utility"
tier: "1-critical"
priority: "highest"
key_benefit: "~2000x speedup via LRU caching"
ai_facing: true
---

# ConfigLoader — YAML Configuration with Caching

## Summary

**What**: Loads YAML config files with automatic LRU caching
**Why**: ~2000x performance gain for repeated loads
**When**: Every time you load a YAML file
**Where**: `AgentQMS/tools/utils/config_loader.py`

## Import

```python
from AgentQMS.tools.utils.config_loader import ConfigLoader
```

## Basic Usage

```python
loader = ConfigLoader()
config = loader.load('configs/train.yaml')
```

## API Reference

### ConfigLoader.load(file_path: str) → dict[str, Any]

Load a YAML file with caching.

**Parameters**:
- `file_path` (str): Path to YAML file (relative or absolute)

**Returns**:
- `dict[str, Any]`: Parsed YAML content

**Behavior**:
- First call: Reads from disk (~5ms)
- Subsequent calls: Returns from cache (~0.002ms)
- Missing file: Returns empty dict `{}`

**Raises**: `None` (graceful failure)

---

### ConfigLoader.clear_cache() → None

Clear the LRU cache.

**Use when**: You need to force reload (e.g., config file changed)

```python
loader.clear_cache()
config = loader.load('configs/train.yaml')  # Reloads from disk
```

---

### ConfigLoader.get_cache_info() → dict

Get cache statistics.

```python
loader = ConfigLoader()
loader.load('configs/train.yaml')
loader.load('configs/train.yaml')  # Cached
info = loader.get_cache_info()
# Returns: {'hits': 1, 'misses': 1, 'size': 1, 'maxsize': 100}
```

## Performance Details

### Cache Configuration

```
Cache size: 100 items (LRU)
TTL: 1 hour
Thread-safe: Yes (internal locking)
```

### Benchmark

| Scenario | Time | Note |
|----------|------|------|
| First load (disk) | ~5ms | I/O bound |
| Cached load (memory) | ~0.002ms | 2500x faster |
| Clear + reload | ~5ms | Forces disk I/O |

### Real-World Impact

```python
# Without caching (naive approach)
for i in range(1000):
    config = yaml.safe_load(open('config.yaml'))
    # Total: ~5000ms

# With ConfigLoader
loader = ConfigLoader()
for i in range(1000):
    config = loader.load('config.yaml')
    # Total: ~2ms (1 disk load + 999 cache hits)

# Speedup: ~2500x faster ✓
```

## Common Use Cases

### Case 1: Load Training Config

```python
from AgentQMS.tools.utils.config_loader import ConfigLoader

loader = ConfigLoader()
train_config = loader.load('configs/train.yaml')

# Access fields
batch_size = train_config.get('batch_size', 32)
epochs = train_config.get('epochs', 100)
```

### Case 2: Load Multiple Configs

```python
loader = ConfigLoader()

base_config = loader.load('configs/base.yaml')
model_config = loader.load('configs/model.yaml')
train_config = loader.load('configs/train.yaml')

# All cached after first load
```

### Case 3: Handle Missing Files

```python
loader = ConfigLoader()
config = loader.load('nonexistent.yaml')

if not config:
    print("Config not found, using defaults")
    config = {'default': True}
```

## Error Handling

### What Happens on Error

```
File not found
  → Returns empty dict: {}

Invalid YAML syntax
  → Returns empty dict: {}

No exception raised
  → Graceful failure
```

### Best Practice: Validate After Loading

```python
config = loader.load('configs/train.yaml')

if not config:
    raise ValueError("Config file missing or invalid")

# Now safe to use
batch_size = config['batch_size']
```

## Advanced: Custom Loader

For advanced configuration:

```python
from AgentQMS.tools.utils.config_loader import ConfigLoader

loader = ConfigLoader(cache_size=200, cache_ttl_hours=2)
config = loader.load('configs/train.yaml')
```

## Integration with Other Utils

### With paths utility:

```python
from AgentQMS.tools.utils.config_loader import ConfigLoader
from AgentQMS.tools.utils.paths import get_configs_dir
import os

loader = ConfigLoader()
config_path = os.path.join(get_configs_dir(), 'train.yaml')
config = loader.load(config_path)
```

### With timestamps utility:

```python
from AgentQMS.tools.utils.config_loader import ConfigLoader
from AgentQMS.tools.utils.timestamps import get_kst_timestamp

loader = ConfigLoader()
config = loader.load('configs/train.yaml')

# Add load timestamp to artifact metadata
metadata = {
    'loaded_at': get_kst_timestamp(),
    'config': config
}
```

## Testing

```bash
# Run ConfigLoader tests
pytest tests/utils/test_config_loader.py -v

# Test specific scenarios
pytest tests/utils/test_config_loader.py::test_cache_hit_ratio -v
pytest tests/utils/test_config_loader.py::test_missing_file -v
```

## Key Takeaways

✅ **Use ConfigLoader** for all YAML file loads
✅ **Automatic caching** (~2000x speedup)
✅ **Graceful failures** (no exceptions)
✅ **Thread-safe** (safe in multi-threaded apps)

❌ **Don't use** `yaml.safe_load(open(...))` directly
❌ **Don't ignore** missing file returns
❌ **Don't assume** YAML validity without validation

## Reference

**Source**: `AgentQMS/tools/utils/config_loader.py`
**Tests**: `tests/utils/test_config_loader.py`
**Status**: ✅ Production-ready
**Last Updated**: 2026-01-11
