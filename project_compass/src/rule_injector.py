"""
Project Compass V2 - Rule Injector

Loads directives from vault and injects into active pulse instructions.
Ensures rules persist across conversation boundaries.
"""

from __future__ import annotations

from pathlib import Path

from project_compass.src.state_schema import VesselState


def load_directives(vault_path: Path) -> list[str]:
    """
    Load all core directives from vault/directives/.

    Returns list of markdown content strings.
    """
    directives_dir = vault_path / "directives"
    if not directives_dir.exists():
        return []

    rules = []
    for directive_file in sorted(directives_dir.glob("*.md")):
        content = directive_file.read_text(encoding="utf-8")
        rules.append(f"# Directive: {directive_file.stem}\n{content}")

    return rules


def load_milestone_rules(vault_path: Path, milestone_id: str) -> str | None:
    """
    Load milestone-specific rules from vault/milestones/{milestone_id}.md.

    Returns markdown content or None if not found.
    """
    milestone_file = vault_path / "milestones" / f"{milestone_id}.md"
    if milestone_file.exists():
        return milestone_file.read_text(encoding="utf-8")
    return None


def inject_rules(state: VesselState, vault_path: Path) -> VesselState:
    """
    Inject vault rules into active pulse instructions.

    This is the core mechanism for preserving context across sessions.

    Args:
        state: Current VesselState with active pulse
        vault_path: Path to vault/ directory

    Returns:
        Updated VesselState with injected instructions
    """
    if not state.active_pulse:
        return state

    # Clear existing instructions (re-inject fresh from vault)
    instructions = []

    # 1. Load core directives
    directives = load_directives(vault_path)
    instructions.extend(directives)

    # 2. Load milestone-specific rules
    milestone_rules = load_milestone_rules(vault_path, state.active_pulse.milestone_id)
    if milestone_rules:
        instructions.append(f"# Milestone Rules: {state.active_pulse.milestone_id}\n{milestone_rules}")

    # 3. Update pulse
    state.active_pulse.instructions = instructions

    return state


def get_injected_context(state: VesselState) -> str:
    """
    Format injected instructions for prompt context.

    Returns a single string suitable for inclusion in AI prompts.
    """
    if not state.active_pulse or not state.active_pulse.instructions:
        return "No active pulse or instructions."

    return "\n\n---\n\n".join(state.active_pulse.instructions)
