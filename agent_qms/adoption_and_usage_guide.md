# AI Agent Guide: Adopting and Using the Quality Management Framework

This document provides structured prompts for AI agents to either integrate the Quality Management Framework (QMF) into an existing project or to use it once it has been adopted.

---

## 1. Prompt for Adopting the QMF in an Existing Project

**Objective:** To integrate the Quality Management Framework into a project that does not currently use it.

**Instructions for AI Agent:**

```
**System Prompt: Adopt Quality Management Framework**

Your task is to integrate our standard Quality Management Framework (QMF) into the current project. The framework is designed to standardize artifact creation, ensure quality, and improve your operational efficiency.

**Step-by-Step Instructions:**

1.  **Locate and Copy Framework:**
    *   The QMF is located in a reference directory. You will need to copy the entire `quality_management_framework` directory into the project's `.claude/export_package/` directory.
    *   Verify that the following structure is now present:
        ```
        .claude/export_package/quality_management_framework/
        ├── q-manifest.yaml
        ├── schemas/
        ├── templates/
        └── toolbelt/
        ```

2.  **Install Dependencies:**
    *   The QMF `toolbelt` requires the following Python libraries: `PyYAML`, `jinja2`, `jsonschema`, and `pytz`.
    *   Check the project's `pyproject.toml` or `requirements.txt`. If these dependencies are missing, add them and run the appropriate installation command (e.g., `uv pip install -r requirements.txt`).

3.  **Understand the Core Components:**
    *   **`q-manifest.yaml`**: This is your single source of truth. Read this file to understand available artifact types, their schemas, and the location of tool entrypoints.
    *   **`toolbelt/core.py`**: This contains the `QualityManagementToolbelt` class. This is your primary interface for all QMF operations. **Do not use shell commands or manual file edits for framework tasks.**

4.  **Confirm Integration:**
    *   Once you have copied the files and installed the dependencies, confirm that you can import the toolbelt successfully:
        ```python
        from .claude.export_package.quality_management_framework.toolbelt import QualityManagementToolbelt
        try:
            toolbelt = QualityManagementToolbelt()
            print("QMF Toolbelt loaded successfully.")
            print("Available artifact types:", toolbelt.list_artifact_types())
        except Exception as e:
            print(f"Error loading QMF Toolbelt: {e}")
        ```

5.  **Proceed with your primary task**, using the QMF for all relevant artifact creation and management.
```

---

## 2. Prompt for Using the QMF in an Integrated Project

**Objective:** To use the already-integrated Quality Management Framework for development tasks.

**Instructions for AI Agent:**

```
**System Prompt: Utilize the Quality Management Framework**

This project uses the Quality Management Framework (QMF) for all artifact and documentation tasks. You must use the provided `QualityManagementToolbelt` for these operations.

**Core Principles:**

1.  **Programmatic Interface Only:** All framework interactions MUST go through the `QualityManagementToolbelt` Python class. Do not use `make`, `python` scripts from the command line, or manual file I/O for creating or managing artifacts.

2.  **Discover, Don't Assume:** The `q-manifest.yaml` file, located in `.claude/export_package/quality_management_framework/`, is the single source of truth. Before creating an artifact, you can consult this file or use the toolbelt's methods to understand what artifact types are available.

**Workflow Example: Creating a New 'Assessment' Artifact**

1.  **Import and Instantiate the Toolbelt:**
    ```python
    from .claude.export_package.quality_management_framework.toolbelt import QualityManagementToolbelt
    qmf_toolbelt = QualityManagementToolbelt()
    ```

2.  **Discover Available Artifact Types (Optional but Recommended):**
    ```python
    available_types = qmf_toolbelt.list_artifact_types()
    print(f"Available artifact types: {available_types}")
    # Expected output might include: ['implementation_plan', 'assessment']
    ```

3.  **Create the Artifact:**
    *   Use the `create_artifact` method. Provide the `artifact_type`, `title`, and `content`.
    ```python
    artifact_title = "Analysis of User Login Performance"
    artifact_content = """
    ## 1. Summary
    The user login process is currently experiencing a 15% slowdown during peak hours.

    ## 2. Assessment
    The root cause appears to be a bottleneck in the database query for user authentication.

    ## 3. Recommendations
    We recommend adding an index to the 'users' table on the 'email' column.
    """

    new_artifact_path = qmf_toolbelt.create_artifact(
        artifact_type="assessment",
        title=artifact_title,
        content=artifact_content,
        author="ai-agent-beta",
        tags=["performance", "database", "auth"]
    )

    print(f"Successfully created new assessment at: {new_artifact_path}")
    ```

4.  **Validate the Artifact (Optional):**
    *   You can use the `validate_artifact` method to ensure the new artifact's frontmatter conforms to its schema.
    ```python
    is_valid = qmf_toolbelt.validate_artifact(new_artifact_path)
    print(f"Artifact is valid: {is_valid}")
    ```

**Your primary directive is to follow this programmatic approach for all quality and documentation tasks.**
```
