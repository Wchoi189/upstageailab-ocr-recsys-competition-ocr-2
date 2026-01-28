from typing import Any

from AgentQMS.tools.utils.paths import get_project_root

try:
    from AgentQMS.tools.core.context.context_bundle import (
        is_fresh,
        list_available_bundles,
        load_bundle_definition,
        validate_bundle_files,
    )

    CONTEXT_BUNDLES_AVAILABLE = True
except ImportError:
    CONTEXT_BUNDLES_AVAILABLE = False


def validate_bundles() -> list[dict]:
    """
    Validate context bundle definitions.

    Checks:
    - All bundle definition files exist and are valid YAML
    - All files referenced in bundles exist
    - All bundle files are fresh (modified within last 30 days)

    Returns:
        List of validation result dictionaries
    """
    if not CONTEXT_BUNDLES_AVAILABLE:
        return []

    results = []

    try:
        project_root = get_project_root()
        available_bundles = list_available_bundles()

        for bundle_name in available_bundles:
            # Determine bundle file path (framework or plugin)
            framework_bundle_path = project_root / "AgentQMS" / ".agentqms" / "plugins" / "context_bundles" / f"{bundle_name}.yaml"
            plugin_bundle_path = project_root / ".agentqms" / "plugins" / "context_bundles" / f"{bundle_name}.yaml"

            if framework_bundle_path.exists():
                bundle_file_display = f"AgentQMS/.agentqms/plugins/context_bundles/{bundle_name}.yaml"
            elif plugin_bundle_path.exists():
                bundle_file_display = f".agentqms/plugins/context_bundles/{bundle_name}.yaml"
            else:
                bundle_file_display = f"context_bundles/{bundle_name}.yaml"

            bundle_result: dict[str, Any] = {
                "file": bundle_file_display,
                "valid": True,
                "errors": [],
                "warnings": [],
            }

            try:
                # Load bundle definition
                bundle_def = load_bundle_definition(bundle_name)

                # Validate bundle files
                validate_bundle_files(bundle_def)

                # Check for missing files
                tiers = bundle_def.get("tiers", {})

                for tier_key, tier in tiers.items():
                    tier_files = tier.get("files", [])
                    for file_spec in tier_files:
                        if isinstance(file_spec, str):
                            file_path_str = file_spec
                            is_optional = False
                        elif isinstance(file_spec, dict):
                            file_path_str = file_spec.get("path", "")
                            is_optional = file_spec.get("optional", False)
                        else:
                            continue

                        # Skip glob patterns (handled by validate_bundle_files)
                        if "*" in file_path_str or "**" in file_path_str:
                            continue

                        # Check if file exists
                        file_path = project_root / file_path_str
                        if not file_path.exists():
                            if is_optional:
                                # Optional files don't fail validation
                                bundle_result["warnings"].append(f"Optional file missing in {bundle_name} bundle: {file_path_str}")
                            else:
                                bundle_result["valid"] = False
                                bundle_result["errors"].append(f"Missing file in {bundle_name} bundle: {file_path_str}")
                        elif not is_fresh(file_path, days=30):
                            bundle_result["warnings"].append(
                                f"Stale file in {bundle_name} bundle: {file_path_str} (not modified in last 30 days)"
                            )

            except FileNotFoundError:
                bundle_result["valid"] = False
                bundle_result["errors"].append(f"Bundle definition file not found: {bundle_name}.yaml")
            except Exception as e:
                bundle_result["valid"] = False
                bundle_result["errors"].append(f"Error validating bundle {bundle_name}: {e!s}")

            results.append(bundle_result)

    except Exception as e:
        # Add error result if bundle system fails
        results.append(
            {
                "file": "context_bundles/",
                "valid": False,
                "errors": [f"Error validating bundles: {e!s}"],
                "warnings": [],
            }
        )

    return results
