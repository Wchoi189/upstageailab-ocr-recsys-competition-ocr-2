# Safe State Manager

A utility for safely reading and writing experiment state files to prevent corruption.

## Overview

The Safe State Manager provides atomic operations for experiment state files, preventing corruption that can occur during direct JSON/YAML editing. It uses YAML format for better readability and error tolerance.

## Features

- **Atomic Writes**: Changes are written to temporary files and atomically moved
- **Automatic Backups**: Original files are backed up before modifications
- **YAML Support**: More human-readable and forgiving than JSON
- **Validation**: Built-in validation of state structure
- **CLI Interface**: Command-line tools for safe operations

## Usage

### Command Line

```bash
# Validate a state file
python scripts/safe_state_manager.py path/to/state.yml --validate

# Get a section
python scripts/safe_state_manager.py path/to/state.yml --get tasks

# Set a section
python scripts/safe_state_manager.py path/to/state.yml --set status completed
```

### Python API

```python
from experiment_tracker.scripts.safe_state_manager import SafeStateManager

manager = SafeStateManager(Path("path/to/state.yml"))

# Load state
state = manager.load_state()

# Update section
manager.update_section("status", "completed")

# Validate
issues = manager.validate_state()
```

## Migration from JSON

To migrate existing JSON state files to YAML:

```bash
python scripts/migrate_state_to_yaml.py path/to/state.json
```

This will create `state.yml` and backup the original as `state.json.backup`.

## Integration with ETK

The safe state manager integrates with the Experiment Tracker Kit (ETK):

```bash
cd experiment-tracker
etk sync --all  # Syncs state.yml with metadata
etk validate    # Validates consistency
```

## Safety Guidelines

1. **Always use the safe manager** for programmatic changes
2. **Run validation** after manual edits
3. **Check backups** are created before risky operations
4. **Use ETK sync** to keep metadata consistent

## File Format

State files use YAML format with the following structure:

```yaml
id: "experiment_id"
status: "active"
created_at: "2025-12-18T00:00:00"
updated_at: "2025-12-18T19:00:00"
type: "experiment_type"
description: "Experiment description"
# ... other fields
```

## Error Handling

- Invalid YAML/JSON triggers automatic rollback to backup
- Validation failures prevent writes
- All operations are logged for debugging
