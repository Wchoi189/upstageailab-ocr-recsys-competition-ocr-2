"""Concrete policies for Telemetry Middleware."""
import re
from pathlib import Path
from typing import Any

from .telemetry import PolicyViolation
from AgentQMS.tools.utils.paths import get_project_root


class RedundancyInterceptor:
    """Detects if AgentQMS is duplicating work managed by the provider."""

    def __init__(self) -> None:
        # Map AgentQMS artifact types to their shadow filenames/types in .gemini/
        self.redundancy_map = {
            "implementation_plan": "implementation_plan",
            "task_list": "task",
            "walkthrough": "walkthrough"
        }
        # The opaque managed directory
        self.shadow_dir = Path(".gemini")

    def validate(self, tool_name: str, arguments: dict[str, Any]) -> None:
        # Only check artifact creation
        if tool_name not in ("create_artifact", "mcp_unified_project_create_artifact"):
            return

        artifact_type = arguments.get("artifact_type")
        if not artifact_type:
            return

        # Check if this type is managed
        if artifact_type in self.redundancy_map:
            # Check if a shadow artifact exists
            # We look for a file that indicates the provider already did this
            try:
                # We assume the user's home .gemini structure for now as per prompt
                # user_prompt: "If the system detects the .gemini/ directory has been updated with a plan"
                # We'll check the project root .gemini directory which the user mentioned
                shadow_root = get_project_root() / ".gemini"

                # Heuristic: Check if there's a recent file with the artifact key name
                # e.g. .gemini/brain/.../task.md
                # Since we don't know the exact UUID used by the *provider*, we scan subdirs.
                if shadow_root.exists():
                    found_shadow = False
                    # DFS for the artifact type name
                    search_pattern = f"*{self.redundancy_map[artifact_type]}*"
                    for path in shadow_root.rglob(search_pattern):
                         if path.is_file():
                             found_shadow = True
                             break

                    if found_shadow:
                        raise PolicyViolation(
                            message=f"Redundancy detected for {artifact_type}",
                            feedback_to_ai=(
                                f"NOTICE: The '{artifact_type}' is already managed "
                                "by the internal Antigravity service in .gemini/. "
                                "To prevent memory bloat, please reference the managed version "
                                "instead of creating a duplicate via AgentQMS."
                            )
                        )
            except PolicyViolation:
                # Re-raise the policy violation so it propagates to the server
                raise
            except Exception:
                # Be resilient to FS errors so we don't crash the server
                pass


class ComplianceInterceptor:
    """Enforces coding standards at runtime."""

    def validate(self, tool_name: str, arguments: dict[str, Any]) -> None:
        # Check for force override
        if arguments.get("force") is True or str(arguments.get("force")).lower() == "true":
            # Determine complexity to warn, but allow it.
            # Ideally we log this override.
            return

        # We only care about tools that execute code or write code
        code_content = arguments.get("code") or arguments.get("content") or arguments.get("CodeContent")

        if not code_content or not isinstance(code_content, str):
            # Check for command execution
            command = arguments.get("command") or arguments.get("CommandLine")
            if command and isinstance(command, str):
                self._check_python_execution(command)
            return

        # Check code content
        self._check_python_execution(code_content)
        self._check_path_usage(code_content)

    def _check_python_execution(self, text: str) -> None:
        # Robust check for bare 'python' not preceded by 'uv run'
        # Matches "python script.py" or "python -m ..." but ignores "import python_module"
        # We look for the command at start of string or following common shell delimiters like &&, ;, |

        # Regex explanation:
        # (?<=^|;|\||&)\s*  -> Lookbehind: Start of line OR separator (; | &), followed by optional whitespace
        # (?<!uv run )      -> Negative Lookbehind: MUST NOT be preceded by "uv run "
        # python3?          -> Match "python" or "python3"
        # \s+               -> Must be followed by whitespace (avoids matching python_variable)

        if re.search(r"(?:^|[;|\|&])\s*(?<!uv run )python3?\s+", text):
             raise PolicyViolation(
                message="Internal Violation: Plain 'python' used.",
                feedback_to_ai="Internal Violation: Plain 'python' used. You MUST use 'uv run python' for environment consistency."
            )

    def _check_path_usage(self, text: str) -> None:
        if "sys.path.append" in text or "sys.path.insert" in text or "sys.path.extend" in text:
             raise PolicyViolation(
                message="Internal Violation: sys.path modified.",
                feedback_to_ai="PROTOCOL ERROR: Do not use sys.path modifications (append/insert). Use 'AgentQMS.tools.utils.paths' or correct environment setup (PYTHONPATH)."
            )

        if ".parent.parent.parent" in text:
             raise PolicyViolation(
                message="Internal Violation: Excessive parent chaining.",
                feedback_to_ai="PROTOCOL ERROR: Excessive .parent chaining detected. Use 'AgentQMS.tools.utils.paths.get_project_root()'."
            )


class StandardsInterceptor:
    """Enforces ADS v1.0 frontmatter on standard documents."""

    REQUIRED_KEYS = {
        "ads_version", "type", "agent", "tier", "priority",
        "validates_with", "compliance_status", "memory_footprint"
    }

    def validate(self, tool_name: str, arguments: dict[str, Any]) -> None:
        # Check for force override
        if arguments.get("force") is True or str(arguments.get("force")).lower() == "true":
            return

        if tool_name != "write_to_file":
            return

        target_file = arguments.get("TargetFile")
        if not target_file:
            return

        path = Path(target_file)

        # Only enforce on AgentQMS/standards/*.yaml
        if "AgentQMS/standards" not in str(path) or path.suffix != ".yaml":
            return

        content = arguments.get("CodeContent")
        if not content:
            return

        try:
            # Simple check for frontmatter keys to avoid heavyweight parsing if possible,
            # but regex/parsing is safer. Let's do a quick YAML load if possible,
            # or just regex for the keys since we can't import yaml easily here without overhead?
            # Actually, we can assume standard imports.
            import yaml

            # Handle multi-document streams (frontmatter often uses ---)
            # ADS spec says "YAML structured data only", implies the whole file is YAML.
            data = yaml.safe_load(content)

            if not isinstance(data, dict):
                 raise PolicyViolation(
                    message="Standards Violation: Root must be a dict.",
                    feedback_to_ai="ADS VIOLATION: Standards files must be a valid YAML dictionary."
                )

            missing = self.REQUIRED_KEYS - data.keys()
            if missing:
                raise PolicyViolation(
                    message=f"Standards Violation: Missing keys {missing}",
                    feedback_to_ai=f"ADS VIOLATION: Missing required ADS v1.0 frontmatter keys: {missing}. See AgentQMS/standards/schemas/ads-v1.0-spec.yaml"
                )

            if str(data.get("ads_version")) != "1.0":
                 raise PolicyViolation(
                    message="Standards Violation: Wrong version",
                    feedback_to_ai="ADS VIOLATION: ads_version must be '1.0'"
                )

        except ImportError:
            pass # yaml not available? Should be.
        except yaml.YAMLError as e:
             raise PolicyViolation(
                message=f"Standards Violation: Invalid YAML: {e}",
                feedback_to_ai=f"ADS VIOLATION: Invalid YAML format: {e}"
            )
        except PolicyViolation:
            raise
        except Exception:
            # Don't block if something weird happens
            pass
