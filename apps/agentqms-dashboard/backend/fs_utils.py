import os
from typing import Any


def list_files(path: str) -> list[dict[str, Any]]:
    """
    List files in a directory.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    items = []
    for entry in os.scandir(path):
        items.append({
            "name": entry.name,
            "type": "directory" if entry.is_dir() else "file",
            "size": entry.stat().st_size,
            "last_modified": entry.stat().st_mtime
        })
    return items

def read_file(path: str) -> str:
    """
    Read file content.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, encoding='utf-8') as f:
        return f.read()
