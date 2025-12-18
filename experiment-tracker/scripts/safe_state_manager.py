#!/usr/bin/env python3
"""
Safe State Manager for experiment-tracker state.json

Provides atomic read/write operations with validation to prevent corruption.
Uses YAML format for better readability and error tolerance.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml


class SafeStateManager:
    """Manages experiment state with corruption prevention."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.backup_file = state_file.with_suffix('.backup.json')

    def load_state(self) -> dict[str, Any]:
        """Load state with validation."""
        if not self.state_file.exists():
            return {}

        try:
            with open(self.state_file) as f:
                if self.state_file.suffix == '.json':
                    data = json.load(f)
                else:  # Assume YAML
                    data = yaml.safe_load(f) or {}
            return data
        except Exception as e:
            print(f"Error loading state: {e}")
            # Try backup
            if self.backup_file.exists():
                print("Loading from backup...")
                with open(self.backup_file) as f:
                    return json.load(f)
            return {}

    def save_state(self, data: dict[str, Any]) -> bool:
        """Save state atomically with backup."""
        try:
            # Create backup
            if self.state_file.exists():
                self.state_file.replace(self.backup_file)

            # Write to temp file first
            with tempfile.NamedTemporaryFile(mode='w', suffix=self.state_file.suffix,
                                          dir=self.state_file.parent, delete=False) as temp:
                if self.state_file.suffix == '.json':
                    json.dump(data, temp, indent=2, ensure_ascii=False)
                else:
                    yaml.dump(data, temp, default_flow_style=False, allow_unicode=True)

                temp_path = Path(temp.name)

            # Atomic move
            temp_path.replace(self.state_file)
            return True

        except Exception as e:
            print(f"Error saving state: {e}")
            # Restore backup
            if self.backup_file.exists():
                self.backup_file.replace(self.state_file)
            return False

    def update_section(self, section: str, data: Any) -> bool:
        """Update a specific section safely."""
        state = self.load_state()
        state[section] = data
        return self.save_state(state)

    def get_section(self, section: str) -> Any:
        """Get a specific section."""
        state = self.load_state()
        return state.get(section)

    def validate_state(self) -> list[str]:
        """Validate state structure."""
        issues = []
        state = self.load_state()

        required_keys = ['id', 'status', 'created_at']
        for key in required_keys:
            if key not in state:
                issues.append(f"Missing required key: {key}")

        # Add more validation as needed
        return issues


# CLI interface
if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Safe State Manager')
    parser.add_argument('state_file', help='Path to state file')
    parser.add_argument('--format', choices=['json', 'yaml'], default='json',
                       help='File format')
    parser.add_argument('--validate', action='store_true', help='Validate state')
    parser.add_argument('--get', help='Get section')
    parser.add_argument('--set', nargs=2, help='Set section (key value)')

    args = parser.parse_args()

    state_file = Path(args.state_file)
    if args.format == 'yaml':
        state_file = state_file.with_suffix('.yml')

    manager = SafeStateManager(state_file)

    if args.validate:
        issues = manager.validate_state()
        if issues:
            print("Validation issues:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        else:
            print("State is valid")

    elif args.get:
        section = manager.get_section(args.get)
        print(json.dumps(section, indent=2))

    elif args.set:
        key, value = args.set
        try:
            data = json.loads(value)
        except:
            data = value
        if manager.update_section(key, data):
            print(f"Updated {key}")
        else:
            print("Update failed")
            sys.exit(1)
