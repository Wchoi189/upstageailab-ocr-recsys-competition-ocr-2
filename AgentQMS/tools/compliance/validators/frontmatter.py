from datetime import datetime
from pathlib import Path

DATE_FORMAT = "%Y-%m-%d %H:%M (KST)"

def _extract_frontmatter(file_path: Path) -> dict[str, str]:
    """Extract frontmatter from a file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        if not content.startswith("---"):
            return {}

        frontmatter_end = content.find("---", 3)
        if frontmatter_end == -1:
            return {}

        frontmatter_content = content[3:frontmatter_end]
        frontmatter = {}
        for line in frontmatter_content.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                frontmatter[key] = value
        return frontmatter
    except Exception:
        return {}

def validate_frontmatter(
    file_path: Path,
    valid_statuses: list[str],
    valid_categories: list[str],
    valid_types: list[str],
    required_frontmatter: list[str]
) -> tuple[bool, str]:
    """Validate frontmatter structure and content."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return False, f"Error reading file: {e}"

    # Check for frontmatter
    if not content.startswith("---"):
        return False, "Missing frontmatter (file should start with '---')"

    # Extract frontmatter
    frontmatter_end = content.find("---", 3)
    if frontmatter_end == -1:
        return False, "Malformed frontmatter (missing closing '---')"

    frontmatter_content = content[3:frontmatter_end]

    # Parse frontmatter (simple YAML-like parsing)
    frontmatter = {}
    for line in frontmatter_content.split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            frontmatter[key] = value

    # Check required fields
    missing_fields = []
    for field in required_frontmatter:
        if field not in frontmatter:
            missing_fields.append(field)

    if missing_fields:
        return (
            False,
            f"Missing required frontmatter fields: {', '.join(missing_fields)}",
        )

    # Validate field values
    validation_errors = []

    date_value = frontmatter.get("date", "").strip()
    if date_value:
        try:
            datetime.strptime(date_value, DATE_FORMAT)
        except ValueError:
            validation_errors.append("Date must use 'YYYY-MM-DD HH:MM (KST)' format (24-hour clock).")

    if "type" in frontmatter and frontmatter["type"] not in valid_types:
        validation_errors.append(f"Invalid type '{frontmatter['type']}'. Valid types: {', '.join(valid_types)}")

    if "category" in frontmatter and frontmatter["category"] not in valid_categories:
        validation_errors.append(f"Invalid category '{frontmatter['category']}'. Valid categories: {', '.join(valid_categories)}")

    if "status" in frontmatter and frontmatter["status"] not in valid_statuses:
        validation_errors.append(f"Invalid status '{frontmatter['status']}'. Valid statuses: {', '.join(valid_statuses)}")

    if validation_errors:
        return False, "; ".join(validation_errors)

    return True, "Valid frontmatter"
