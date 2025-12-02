"""Audio message templates for AI agents.

This module provides pre-generated audio messages organized by category
and utilities for generating custom messages.
"""

# Pre-generated messages organized by category
MESSAGES: dict[str, list[str]] = {
    "task_completion": [
        "Task complete.",
        "All done.",
        "Finished successfully.",
        "That's complete.",
        "Ready for the next task.",
    ],
    "process_completion": [
        "Process finished.",
        "Operation complete.",
        "All set.",
        "Done and ready.",
    ],
    "success_status": [
        "Success.",
        "Everything looks good.",
        "No issues found.",
        "All systems operational.",
    ],
    "progress_updates": [
        "Working on it.",
        "Almost there.",
        "Just a moment.",
        "Processing now.",
    ],
    "file_operations": [
        "File saved.",
        "Download complete.",
        "Export finished.",
        "Upload complete.",
    ],
    "code_operations": [
        "Build complete.",
        "Tests passed.",
        "Deployment ready.",
        "Code compiled successfully.",
    ],
    "warnings": [
        "Please check this.",
        "Something needs attention.",
        "Review required.",
        "Take a look when you can.",
    ],
    "general_status": [
        "All set.",
        "Ready to go.",
        "You're all set.",
        "Standing by.",
    ],
}

# All messages flattened for random selection
ALL_MESSAGES: list[str] = [
    message for messages in MESSAGES.values() for message in messages
]


def get_message(category: str, index: int | None = None) -> str:
    """Get a message from a specific category.

    Args:
        category: Category name (e.g., "task_completion", "success_status")
        index: Optional index to select specific message (default: random)

    Returns:
        Selected message string

    Raises:
        KeyError: If category doesn't exist
        IndexError: If index is out of range
    """
    if category not in MESSAGES:
        raise KeyError(
            f"Unknown category: {category}. Available: {list(MESSAGES.keys())}"
        )

    messages = MESSAGES[category]
    if index is None:
        import random

        return random.choice(messages)
    return messages[index]


def get_random_message() -> str:
    """Get a random message from any category.

    Returns:
        Random message string
    """
    import random

    return random.choice(ALL_MESSAGES)


def list_categories() -> list[str]:
    """List all available message categories.

    Returns:
        List of category names
    """
    return list(MESSAGES.keys())


def list_messages(category: str) -> list[str]:
    """List all messages in a category.

    Args:
        category: Category name

    Returns:
        List of messages in the category

    Raises:
        KeyError: If category doesn't exist
    """
    if category not in MESSAGES:
        raise KeyError(
            f"Unknown category: {category}. Available: {list(MESSAGES.keys())}"
        )
    return MESSAGES[category]


def validate_message(message: str, max_sentences: int = 3) -> bool:
    """Validate that a custom message meets guidelines.

    Args:
        message: Message to validate
        max_sentences: Maximum number of sentences allowed (default: 3)

    Returns:
        True if message is valid, False otherwise
    """
    if not message or not message.strip():
        return False

    # Count sentences (simple heuristic: count periods, exclamation marks, question marks)
    sentence_count = sum(1 for char in message if char in ".!?")
    if sentence_count > max_sentences:
        return False

    # Check length (rough estimate: should be under 200 characters for ~10 seconds)
    return not len(message) > 200


def suggest_message(event_type: str) -> str:
    """Suggest an appropriate message for an event type.

    Args:
        event_type: Type of event (e.g., "task_complete", "build_success", "error")

    Returns:
        Suggested message string
    """
    # Map common event types to categories
    event_mapping = {
        "task_complete": "task_completion",
        "task_finished": "task_completion",
        "process_complete": "process_completion",
        "build_success": "code_operations",
        "build_complete": "code_operations",
        "test_passed": "code_operations",
        "tests_passed": "code_operations",
        "deployment_ready": "code_operations",
        "file_saved": "file_operations",
        "download_complete": "file_operations",
        "export_complete": "file_operations",
        "upload_complete": "file_operations",
        "success": "success_status",
        "warning": "warnings",
        "needs_attention": "warnings",
        "in_progress": "progress_updates",
        "processing": "progress_updates",
    }

    category = event_mapping.get(event_type.lower(), "general_status")
    return get_message(category)
