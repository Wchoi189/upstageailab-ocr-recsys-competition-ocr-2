#!/usr/bin/env python3
"""
Migrate state.json to state.yml for better safety and readability.
"""

from pathlib import Path

from safe_state_manager import SafeStateManager


def migrate_to_yaml(state_file: Path) -> bool:
    """Migrate JSON state file to YAML."""
    yaml_file = state_file.with_suffix('.yml')

    # Load current state
    manager = SafeStateManager(state_file)
    state = manager.load_state()

    if not state:
        print("No state to migrate")
        return False

    # Save as YAML
    yaml_manager = SafeStateManager(yaml_file)
    if yaml_manager.save_state(state):
        print(f"Migrated to {yaml_file}")
        # Optionally backup and remove old file
        backup = state_file.with_suffix('.json.backup')
        if state_file.exists():
            state_file.replace(backup)
            print(f"Original backed up to {backup}")
        return True

    return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Migrate state to YAML')
    parser.add_argument('state_file', help='Path to state.json')

    args = parser.parse_args()
    state_file = Path(args.state_file)

    if migrate_to_yaml(state_file):
        print("Migration successful")
    else:
        print("Migration failed")
