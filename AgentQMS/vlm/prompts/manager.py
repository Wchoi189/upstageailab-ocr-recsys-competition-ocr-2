"""Prompt Template Manager.

Handles loading and rendering of markdown and Jinja2 prompt templates.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    Environment = None
    FileSystemLoader = None

if TYPE_CHECKING:
    from jinja2 import Template

from AgentQMS.vlm.core.contracts import AnalysisMode
from AgentQMS.vlm.utils.paths import get_path_resolver


class PromptManager:
    """Manages prompt templates and rendering."""

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize prompt manager.

        Args:
            templates_dir: Directory containing prompt templates
        """
        resolver = get_path_resolver()
        self.templates_dir = templates_dir or resolver.get_prompt_templates_path()
        self.markdown_dir = self.templates_dir / "markdown"
        self.jinja2_dir = self.templates_dir / "jinja2"
        self.examples_file = self.templates_dir / "few_shot_examples.json"

        # Initialize Jinja2 environment if available
        if Environment is not None and FileSystemLoader is not None:
            self.jinja2_env = Environment(
                loader=FileSystemLoader(str(self.jinja2_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self.jinja2_env = None

        # Cache for loaded templates
        self._template_cache: Dict[str, str] = {}
        self._examples_cache: Optional[List[Dict[str, Any]]] = None

    def load_markdown_template(self, mode: AnalysisMode) -> str:
        """Load a markdown template for the given analysis mode.

        Args:
            mode: Analysis mode

        Returns:
            Template content as string

        Raises:
            FileNotFoundError: If template file not found
        """
        cache_key = f"markdown_{mode.value}"
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        template_file = self.markdown_dir / f"{mode.value}_analysis.md"
        if not template_file.exists():
            raise FileNotFoundError(f"Markdown template not found: {template_file}")

        content = template_file.read_text()
        self._template_cache[cache_key] = content
        return content

    def load_jinja2_template(self, mode: AnalysisMode) -> "Template":
        """Load a Jinja2 template for the given analysis mode.

        Args:
            mode: Analysis mode

        Returns:
            Jinja2 Template object

        Raises:
            FileNotFoundError: If template file not found
            ImportError: If Jinja2 not installed
        """
        if self.jinja2_env is None:
            raise ImportError(
                "Jinja2 is required for template rendering. Install with: pip install Jinja2"
            )

        template_file = f"{mode.value}_analysis.j2"
        try:
            return self.jinja2_env.get_template(template_file)
        except Exception as e:
            raise FileNotFoundError(f"Jinja2 template not found or invalid: {template_file}") from e

    def render_template(
        self,
        mode: AnalysisMode,
        context: Optional[Dict[str, Any]] = None,
        use_jinja2: bool = True,
    ) -> str:
        """Render a prompt template with context.

        Args:
            mode: Analysis mode
            context: Template context variables
            use_jinja2: Whether to use Jinja2 (True) or markdown (False)

        Returns:
            Rendered prompt text
        """
        if context is None:
            context = {}

        if use_jinja2 and self.jinja2_env is not None:
            try:
                template = self.load_jinja2_template(mode)
                return template.render(**context)
            except (FileNotFoundError, ImportError):
                # Fallback to markdown
                use_jinja2 = False

        if not use_jinja2:
            # Use markdown template (no rendering, just return content)
            return self.load_markdown_template(mode)

        # Should not reach here, but just in case
        return self.load_markdown_template(mode)

    def load_few_shot_examples(self) -> List[Dict[str, Any]]:
        """Load few-shot examples from JSON file.

        Returns:
            List of example dictionaries
        """
        if self._examples_cache is not None:
            return self._examples_cache

        if not self.examples_file.exists():
            return []

        try:
            data = json.loads(self.examples_file.read_text())
            examples = data.get("examples", [])
            self._examples_cache = examples
            return examples
        except (json.JSONDecodeError, KeyError) as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load few-shot examples: {e}")
            return []

    def inject_few_shot_examples(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Inject few-shot examples into template context.

        Args:
            context: Template context

        Returns:
            Updated context with few-shot examples
        """
        examples = self.load_few_shot_examples()
        if examples:
            context["few_shot_examples"] = examples
        return context
