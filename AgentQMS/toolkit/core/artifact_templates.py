#!/usr/bin/env python3
"""
Artifact Template System for AI Agents

This module provides templates and utilities for creating properly formatted
artifacts that follow the project's naming conventions and structure.

Supports extension via plugin system - see .agentqms/plugins/artifact_types/

Usage:
    from artifact_templates import create_artifact, get_template

    # Create a new implementation plan
    create_artifact('implementation_plan', 'my-feature', 'docs/artifacts/')

    # Get template content (including plugin-registered types)
    template = get_template('assessment')
    template = get_template('change_request')  # Plugin-registered type
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Try to import plugin registry for extensibility
try:
    from AgentQMS.agent_tools.core.plugins import get_plugin_registry

    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False
# Try to import new utilities for branch and timestamp handling
try:
    from AgentQMS.agent_tools.utils.git import get_current_branch
    from AgentQMS.agent_tools.utils.timestamps import get_kst_timestamp

    UTILITIES_AVAILABLE = True
except ImportError:
    UTILITIES_AVAILABLE = False
    get_current_branch = None
    get_kst_timestamp = None


class ArtifactTemplates:
    """Templates for creating properly formatted artifacts.

    Supports extension via plugin system. Additional artifact types can be
    registered in .agentqms/plugins/artifact_types/*.yaml
    """

    def __init__(self):
        self.templates = {
            "implementation_plan": {
                "filename_pattern": "YYYY-MM-DD_HHMM_implementation_plan_{name}.md",
                "directory": "implementation_plans/",
                "frontmatter": {
                    "type": "implementation_plan",
                    "category": "development",
                    "status": "active",
                    "version": "1.0",
                    "tags": ["implementation", "plan", "development"],
                },
                "content_template": """# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **{title}**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: {title}

## Progress Tracker
- **STATUS:** Not Started
- **CURRENT STEP:** Phase 1, Task 1.1 - [Initial Task Name]
- **LAST COMPLETED TASK:** None
- **NEXT TASK:** [Description of the immediate next task]

### Implementation Outline (Checklist)

#### **Phase 1: Foundation (Week 1-2)**
1. [ ] **Task 1.1: [Task 1.1 Title]**
   - [ ] [Sub-task 1.1.1 description]
   - [ ] [Sub-task 1.1.2 description]
   - [ ] [Sub-task 1.1.3 description]

2. [ ] **Task 1.2: [Task 1.2 Title]**
   - [ ] [Sub-task 1.2.1 description]
   - [ ] [Sub-task 1.2.2 description]

#### **Phase 2: Core Implementation (Week 3-4)**
3. [ ] **Task 2.1: [Task 2.1 Title]**
   - [ ] [Sub-task 2.1.1 description]
   - [ ] [Sub-task 2.1.2 description]

4. [ ] **Task 2.2: [Task 2.2 Title]**
   - [ ] [Sub-task 2.2.1 description]
   - [ ] [Sub-task 2.2.2 description]

#### **Phase 3: Testing & Validation (Week 5-6)**
5. [ ] **Task 3.1: [Task 3.1 Title]**
   - [ ] [Sub-task 3.1.1 description]
   - [ ] [Sub-task 3.1.2 description]

6. [ ] **Task 3.2: [Task 3.2 Title]**
   - [ ] [Sub-task 3.2.1 description]
   - [ ] [Sub-task 3.2.2 description]

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] [Architectural Principle 1 (e.g., Modular Design)]
- [ ] [Data Model Requirement (e.g., Pydantic V2 Integration)]
- [ ] [Configuration Method (e.g., YAML-Driven)]
- [ ] [State Management Strategy]

### **Integration Points**
- [ ] [Integration with System X]
- [ ] [API Endpoint Definition]
- [ ] [Use of Existing Utility/Library]

### **Quality Assurance**
- [ ] [Unit Test Coverage Goal (e.g., > 90%)]
- [ ] [Integration Test Requirement]
- [ ] [Performance Test Requirement]
- [ ] [UI/UX Test Requirement]

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] [Key Feature 1 Works as Expected]
- [ ] [Key Feature 2 is Fully Implemented]
- [ ] [Performance Metric is Met (e.g., <X ms latency)]
- [ ] [User-Facing Outcome is Achieved]

### **Technical Requirements**
- [ ] [Code Quality Standard is Met (e.g., Documented, type-hinted)]
- [ ] [Resource Usage is Within Limits (e.g., <X GB memory)]
- [ ] [Compatibility with System Y is Confirmed]
- [ ] [Maintainability Goal is Met]

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW / MEDIUM / HIGH
### **Active Mitigation Strategies**:
1. [Mitigation Strategy 1 (e.g., Incremental Development)]
2. [Mitigation Strategy 2 (e.g., Comprehensive Testing)]
3. [Mitigation Strategy 3 (e.g., Regular Code Quality Checks)]

### **Fallback Options**:
1. [Fallback Option 1 if Risk A occurs (e.g., Simplified version of a feature)]
2. [Fallback Option 2 if Risk B occurs (e.g., CPU-only mode)]
3. [Fallback Option 3 if Risk C occurs (e.g., Phased Rollout)]

---

## 🔄 **Blueprint Update Protocol**

**Update Triggers:**
- Task completion (move to next task)
- Blocker encountered (document and propose solution)
- Technical discovery (update approach if needed)
- Quality gate failure (address issues before proceeding)

**Update Format:**
1. Update Progress Tracker (STATUS, CURRENT STEP, LAST COMPLETED TASK, NEXT TASK)
2. Mark completed items with [x]
3. Add any new discoveries or changes to approach
4. Update risk assessment if needed

---

## 🚀 **Immediate Next Action**

**TASK:** [Description of the immediate next task]

**OBJECTIVE:** [Clear, concise goal of the task]

**APPROACH:**
1. [Step 1 to execute the task]
2. [Step 2 to execute the task]
3. [Step 3 to execute the task]

**SUCCESS CRITERIA:**
- [Measurable outcome 1 that defines task completion]
- [Measurable outcome 2 that defines task completion]

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*""",
            },
            "assessment": {
                "filename_pattern": "YYYY-MM-DD_HHMM_assessment-{name}.md",
                "directory": "assessments/",
                "frontmatter": {
                    "type": "assessment",
                    "category": "evaluation",
                    "status": "active",
                    "version": "1.0",
                    "tags": ["assessment", "evaluation", "analysis"],
                },
                "content_template": """# {title}

## Purpose

This assessment evaluates {subject} and provides recommendations for improvement.

## Scope

- **Subject**: {subject}
- **Assessment Date**: {assessment_date}
- **Assessor**: AI Agent
- **Methodology**: {methodology}

## Findings

### Key Findings
1. Finding 1
2. Finding 2
3. Finding 3

### Detailed Analysis

#### Area 1
- **Current State**: Description
- **Issues Identified**: List of issues
- **Impact**: High/Medium/Low

#### Area 2
- **Current State**: Description
- **Issues Identified**: List of issues
- **Impact**: High/Medium/Low

## Recommendations

### High Priority
1. **Recommendation 1**
   - **Action**: Specific action
   - **Timeline**: When to complete
   - **Owner**: Who is responsible

2. **Recommendation 2**
   - **Action**: Specific action
   - **Timeline**: When to complete
   - **Owner**: Who is responsible

### Medium Priority
1. **Recommendation 3**
   - **Action**: Specific action
   - **Timeline**: When to complete

## Implementation Plan

### Phase 1: Immediate Actions (Week 1-2)
- [ ] Action 1
- [ ] Action 2

### Phase 2: Short-term Improvements (Week 3-4)
- [ ] Action 1
- [ ] Action 2

### Phase 3: Long-term Enhancements (Month 2+)
- [ ] Action 1
- [ ] Action 2

## Success Metrics

- **Metric 1**: Target value
- **Metric 2**: Target value
- **Metric 3**: Target value

## Conclusion

Summary of assessment findings and next steps.

---

*This assessment follows the project's standardized format for evaluation and analysis.*""",
            },
            "design": {
                "filename_pattern": "YYYY-MM-DD_HHMM_design-{name}.md",
                "directory": "design_documents/",
                "frontmatter": {
                    "type": "design",
                    "category": "architecture",
                    "status": "active",
                    "version": "1.0",
                    "tags": ["design", "architecture", "specification"],
                },
                "content_template": """# {title}

## Overview

This document describes the design for {component/system}.

## Problem Statement

What problem does this design solve?

## Design Goals

- Goal 1
- Goal 2
- Goal 3

## Architecture

### High-Level Architecture

```
[Architecture Diagram or Description]
```

### Components

#### Component 1
- **Purpose**: What it does
- **Responsibilities**: Key responsibilities
- **Interfaces**: How it interacts with other components

#### Component 2
- **Purpose**: What it does
- **Responsibilities**: Key responsibilities
- **Interfaces**: How it interacts with other components

## Design Decisions

### Decision 1
- **Context**: Why this decision was needed
- **Options Considered**: Alternative approaches
- **Decision**: What was chosen
- **Rationale**: Why this option was selected
- **Consequences**: Implications of this choice

### Decision 2
- **Context**: Why this decision was needed
- **Options Considered**: Alternative approaches
- **Decision**: What was chosen
- **Rationale**: Why this option was selected
- **Consequences**: Implications of this choice

## Implementation Considerations

### Technical Requirements
- Requirement 1
- Requirement 2

### Dependencies
- Dependency 1
- Dependency 2

### Constraints
- Constraint 1
- Constraint 2

## Testing Strategy

### Unit Testing
- Test approach for individual components

### Integration Testing
- Test approach for component interactions

### End-to-End Testing
- Test approach for complete workflows

## Deployment

### Deployment Strategy
- How this will be deployed

### Rollback Plan
- How to rollback if issues occur

## Monitoring & Observability

### Metrics
- Key metrics to monitor

### Logging
- Logging strategy

### Alerting
- Alert conditions and thresholds

## Future Considerations

### Scalability
- How this design will scale

### Extensibility
- How this design can be extended

### Maintenance
- Maintenance considerations

---

*This design document follows the project's standardized format for architectural documentation.*""",
            },
            "research": {
                "filename_pattern": "YYYY-MM-DD_HHMM_research-{name}.md",
                "directory": "research/",
                "frontmatter": {
                    "type": "research",
                    "category": "research",
                    "status": "active",
                    "version": "1.0",
                    "tags": ["research", "investigation", "analysis"],
                },
                "content_template": """# {title}

## Research Question

What question is this research trying to answer?

## Hypothesis

What do we expect to find?

## Methodology

### Research Approach
- Approach 1
- Approach 2

### Data Sources
- Source 1
- Source 2

### Analysis Methods
- Method 1
- Method 2

## Findings

### Key Findings
1. Finding 1
2. Finding 2
3. Finding 3

### Detailed Results

#### Result 1
- **Description**: What was found
- **Evidence**: Supporting data/observations
- **Implications**: What this means

#### Result 2
- **Description**: What was found
- **Evidence**: Supporting data/observations
- **Implications**: What this means

## Analysis

### Patterns Identified
- Pattern 1
- Pattern 2

### Trends Observed
- Trend 1
- Trend 2

### Anomalies
- Anomaly 1
- Anomaly 2

## Conclusions

### Primary Conclusions
1. Conclusion 1
2. Conclusion 2

### Secondary Conclusions
1. Conclusion 3
2. Conclusion 4

## Recommendations

### Immediate Actions
- Action 1
- Action 2

### Future Research
- Research direction 1
- Research direction 2

## Limitations

### Research Limitations
- Limitation 1
- Limitation 2

### Data Limitations
- Limitation 1
- Limitation 2

## References

- Reference 1
- Reference 2

---

*This research document follows the project's standardized format for research documentation.*""",
            },
            "template": {
                "filename_pattern": "YYYY-MM-DD_HHMM_template-{name}.md",
                "directory": "templates/",
                "frontmatter": {
                    "type": "template",
                    "category": "reference",
                    "status": "active",
                    "version": "1.0",
                    "tags": ["template", "reference", "guidelines"],
                },
                "content_template": """# {title}

## Purpose

This template provides a standardized format for {purpose}.

## Usage Instructions

1. Copy this template
2. Replace placeholder text with actual content
3. Follow the structure and formatting guidelines
4. Ensure all required sections are completed

## Template Structure

### Section 1: Overview
Brief description of what this document covers.

### Section 2: Main Content
The primary content of the document.

### Section 3: Implementation
How to implement or use this information.

### Section 4: Examples
Concrete examples of usage.

## Guidelines

### Content Guidelines
- Guideline 1
- Guideline 2

### Formatting Guidelines
- Guideline 1
- Guideline 2

## Examples

### Example 1
```markdown
Example content here
```

### Example 2
```markdown
Another example here
```

## Best Practices

- Practice 1
- Practice 2

## Common Pitfalls

- Pitfall 1: Description and how to avoid
- Pitfall 2: Description and how to avoid

---

*This template follows the project's standardized format for reusable templates.*""",
            },
            "bug_report": {
                "filename_pattern": "YYYY-MM-DD_HHMM_BUG_NNN_{name}.md",
                "directory": "bug_reports/",
                "frontmatter": {
                    "type": "bug_report",
                    "category": "troubleshooting",
                    "status": "active",
                    "severity": "medium",
                    "version": "1.0",
                    "tags": ["bug", "issue", "troubleshooting"],
                },
                "content_template": """# Bug Report: {title}

## Bug ID
BUG-{bug_id}

<!-- REQUIRED: Fill these sections when creating the initial bug report -->
## Summary
Brief description of the bug.

## Environment
- **OS**: Operating system
- **Python Version**: Python version
- **Dependencies**: Key dependencies and versions
- **Browser**: Browser and version (if applicable)

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Error Messages
```
Error message here
```

## Screenshots/Logs
If applicable, include screenshots or relevant log entries.

## Impact
- **Severity**: High/Medium/Low
- **Affected Users**: Who is affected
- **Workaround**: Any temporary workarounds

<!-- OPTIONAL: Resolution sections - fill these during investigation and fixing -->
## Investigation

### Root Cause Analysis
- **Cause**: What is causing the issue
- **Location**: Where in the code
- **Trigger**: What triggers the issue

### Related Issues
- Related issue 1
- Related issue 2

## Proposed Solution

### Fix Strategy
How to fix the issue.

### Implementation Plan
1. Step 1
2. Step 2

### Testing Plan
How to test the fix.

## Status
- [ ] Confirmed
- [ ] Investigating
- [ ] Fix in progress
- [ ] Fixed
- [ ] Verified

## Assignee
Who is working on this bug.

## Priority
High/Medium/Low (urgency for fixing, separate from severity above)

---

*This bug report follows the project's standardized format for issue tracking.*""",
            },
        }

        # Load additional templates from plugin registry
        self._load_plugin_templates()

    def _load_plugin_templates(self) -> None:
        """Load additional artifact templates from plugin registry."""
        if not PLUGINS_AVAILABLE:
            return

        try:
            registry = get_plugin_registry()
            artifact_types = registry.get_artifact_types()

            for name, plugin_def in artifact_types.items():
                # Skip if already defined (builtin takes precedence)
                if name in self.templates:
                    continue

                # Convert plugin schema to template format
                template = self._convert_plugin_to_template(name, plugin_def)
                if template:
                    self.templates[name] = template

        except Exception:
            # Plugin loading is non-critical - continue with builtins
            pass

    def _convert_plugin_to_template(
        self, name: str, plugin_def: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Convert a plugin artifact type definition to template format.

        Plugin schema format:
            metadata:
              filename_pattern: "CR_{date}_{name}.md"
              directory: change_requests/
              frontmatter: {...}
            template: "# Content..."
            template_variables: {...}

        Template format:
            filename_pattern: "YYYY-MM-DD_HHMM_type_{name}.md"
            directory: "directory/"
            frontmatter: {...}
            content_template: "# Content..."
        """
        try:
            metadata = plugin_def.get("metadata", {})

            # Required fields
            filename_pattern = metadata.get("filename_pattern")
            directory = metadata.get("directory")
            template_content = plugin_def.get("template")

            if not all([filename_pattern, directory, template_content]):
                return None

            # Build template dict
            template: dict[str, Any] = {
                "filename_pattern": filename_pattern,
                "directory": directory,
                "frontmatter": metadata.get("frontmatter", {
                    "type": name,
                    "category": "development",
                    "status": "active",
                    "version": "1.0",
                    "tags": [name],
                }),
                "content_template": template_content,
            }

            # Store template variables for use in create_content
            if "template_variables" in plugin_def:
                template["_plugin_variables"] = plugin_def["template_variables"]

            return template

        except Exception:
            return None

    def get_template(self, template_type: str) -> dict | None:
        """Get template configuration for a specific type."""
        return self.templates.get(template_type)

    def get_available_templates(self) -> list:
        """Get list of available template types."""
        return list(self.templates.keys())

    def create_filename(self, template_type: str, name: str) -> str:
        """Create a properly formatted filename for an artifact."""
        template = self.get_template(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")

        # Normalize name to lowercase kebab-case (artifacts must be lowercase)
        # Convert to lowercase and replace spaces/underscores with hyphens
        normalized_name = (
            name.lower()
            .replace(" ", "-")
            .replace("_", "-")
            .replace("--", "-")
            .strip("-")
        )

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H%M")

        # Handle special case for bug reports (need bug ID)
        if template_type == "bug_report":
            # Extract bug ID from name or generate one
            if "_" in name:
                bug_id = name.split("_")[0]
                descriptive_name = normalized_name.replace(bug_id + "-", "").strip("-")
            else:
                bug_id = "001"  # Default bug ID
                descriptive_name = normalized_name
            return str(
                template["filename_pattern"]
                .format(name=descriptive_name)
                .replace("YYYY-MM-DD_HHMM", timestamp)
                .replace("NNN", bug_id)
            )
        else:
            # For plugin-based templates, use .format() with available variables
            # Build context with all possible filename variables
            filename_context = {
                "name": normalized_name,
                "date": timestamp,  # Plugin {date} gets full timestamp
            }

            filename = template["filename_pattern"]

            # Try to format with context (for plugin templates)
            try:
                filename = filename.format(**filename_context)
            except KeyError:
                # Fallback: format with just name
                filename = filename.format(name=normalized_name)

            # Replace builtin pattern for legacy compatibility
            filename = filename.replace("YYYY-MM-DD_HHMM", timestamp)

            return str(filename)

    def create_frontmatter(self, template_type: str, title: str, **kwargs) -> str:
        """Create frontmatter for an artifact."""
        template = self.get_template(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")

        frontmatter = template["frontmatter"].copy()
        frontmatter["title"] = title

        # Add timestamp using new utility if available, fallback to old method
        if UTILITIES_AVAILABLE and get_kst_timestamp:
            frontmatter["date"] = get_kst_timestamp()
        else:
            from datetime import timedelta, timezone
            kst = timezone(timedelta(hours=9))  # KST is UTC+9
            frontmatter["date"] = datetime.now(kst).strftime("%Y-%m-%d %H:%M (KST)")

        # Add branch name if not explicitly provided in kwargs
        if "branch" not in kwargs:
            if UTILITIES_AVAILABLE and get_current_branch:
                try:
                    frontmatter["branch"] = get_current_branch()
                except Exception:
                    frontmatter["branch"] = "main"  # Fallback
            else:
                frontmatter["branch"] = "main"  # Fallback

        # Add any additional frontmatter fields (may override defaults including branch)
        for key, value in kwargs.items():
            frontmatter[key] = value

        # Convert to YAML-like format
        lines = ["---"]
        for key, value in frontmatter.items():
            if isinstance(value, list):
                lines.append(f"{key}: {value}")
            else:
                lines.append(f'{key}: "{value}"')
        lines.append("---")

        return "\n".join(lines)

    def create_content(self, template_type: str, title: str, **kwargs) -> str:
        """Create content for an artifact using the template."""
        template = self.get_template(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")

        content_template = template["content_template"]

        # Default values
        now = datetime.now()
        defaults = {
            "title": title,
            "start_date": now.strftime("%Y-%m-%d"),
            "target_date": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
            "assessment_date": now.strftime("%Y-%m-%d"),
            "subject": "the system",
            "methodology": "systematic analysis",
            "component/system": "the component",
            "purpose": "documentation",
            "bug_id": "001",
        }

        # Add plugin-defined template variables if present
        if "_plugin_variables" in template:
            defaults.update(template["_plugin_variables"])

        # Merge with provided kwargs (user values override defaults)
        context = {**defaults, **kwargs}

        return str(content_template.format(**context))

    def create_artifact(
        self,
        template_type: str,
        name: str,
        title: str,
        output_dir: str = "docs/artifacts/",
        **kwargs,
    ) -> str:
        """Create a complete artifact file."""
        template = self.get_template(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")

        # Create output directory
        output_path = Path(output_dir) / template["directory"]
        output_path.mkdir(parents=True, exist_ok=True)

        # Check for recently created files with the same base name to prevent duplicates
        # Look for files created within the last 5 minutes with matching type and name
        normalized_name = (
            name.lower()
            .replace(" ", "-")
            .replace("_", "-")
            .replace("--", "-")
            .strip("-")
        )

        # Build pattern to match files based on artifact type
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")

        # Search for files with same date and base name pattern
        # Different patterns for different artifact types
        if template_type == "bug_report":
            # Bug reports: BUG_YYYY-MM-DD_HHMM_NNN_{name}.md
            # Extract bug ID if present in name (format: "NNN_descriptive-name")
            if "_" in name:
                bug_id = name.split("_")[0]
                descriptive_name = normalized_name.replace(bug_id + "-", "").strip("-")
            else:
                descriptive_name = normalized_name
            # Match files with same descriptive name (bug ID may differ)
            # Pattern needs to account for: BUG_DATE_TIME_BUGID_NAME.md
            pattern_base = f"BUG_{current_date}_*_*_{descriptive_name}.md"
        else:
            # Other types: YYYY-MM-DD_HHMM_{template_type}_{normalized-name}.md
            pattern_base = f"{current_date}_*_{template_type}_{normalized_name}.md"

        existing_files = list(output_path.glob(pattern_base))

        # Check if any existing file was created recently (within 5 minutes)
        if existing_files:
            for existing_file in sorted(
                existing_files, key=lambda p: p.stat().st_mtime, reverse=True
            ):
                file_mtime = datetime.fromtimestamp(existing_file.stat().st_mtime)
                time_diff = (now - file_mtime).total_seconds()

                # If file was created within the last 5 minutes, reuse it
                if time_diff < 300:  # 5 minutes = 300 seconds
                    print(f"⚠️  Found recently created file: {existing_file.name}")
                    print("   Reusing existing file instead of creating duplicate.")
                    return str(existing_file)

        # Create filename
        filename = self.create_filename(template_type, name)
        file_path = output_path / filename

        # Create content
        frontmatter = self.create_frontmatter(template_type, title, **kwargs)
        content = self.create_content(template_type, title, **kwargs)

        # Write file
        full_content = frontmatter + "\n\n" + content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        return str(file_path)


# Convenience functions
def get_template(template_type: str) -> dict | None:
    """Get template configuration for a specific type."""
    templates = ArtifactTemplates()
    return templates.get_template(template_type)


def create_artifact(
    template_type: str,
    name: str,
    title: str,
    output_dir: str = "docs/artifacts/",
    **kwargs,
) -> str:
    """Create a complete artifact file."""
    templates = ArtifactTemplates()
    return templates.create_artifact(template_type, name, title, output_dir, **kwargs)


def get_available_templates() -> list:
    """Get list of available template types."""
    templates = ArtifactTemplates()
    return templates.get_available_templates()


if __name__ == "__main__":
    # Example usage
    templates = ArtifactTemplates()

    print("Available templates:")
    for template_type in templates.get_available_templates():
        print(f"  - {template_type}")

    # Example: Create an implementation plan
    try:
        file_path = create_artifact(
            "implementation_plan",
            "my-feature",
            "My Feature Implementation Plan",
            subject="feature implementation",
            methodology="agile development",
        )
        print(f"\nCreated artifact: {file_path}")
    except Exception as e:
        print(f"Error creating artifact: {e}")
