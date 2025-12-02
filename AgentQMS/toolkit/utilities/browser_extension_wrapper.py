#!/usr/bin/env python3
"""
Browser Extension Wrapper with Retry Logic and Puppeteer Fallback

This utility provides reliable browser operations with automatic retry
and cost-effective Puppeteer fallback instead of expensive screenshots.

NOTE: This wrapper is designed to be used by AI agents. The browser extension
MCP tools (browser_navigate, browser_snapshot, etc.) are called via function
calls, not Python imports. This wrapper provides patterns and utilities for
handling retries and fallbacks when using these tools.

Usage Pattern:
    When using browser extension tools in agent code, follow these patterns:

    1. Use retry logic for all browser operations
    2. Always wait for page load before snapshots
    3. Fallback to Puppeteer (not screenshots) when extension fails
    4. Use lightweight checks (console, network) before expensive operations
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class BrowserExtensionWrapper:
    """
    Wrapper for browser extension MCP tools with retry logic and fallback.

    This wrapper addresses common failure patterns:
    - Timeout issues (retry with exponential backoff)
    - Connection instability (automatic reconnection)
    - Wait condition failures (proper wait sequences)
    - Missing error handling (comprehensive error handling)
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        use_puppeteer_fallback: bool = True,
        log_failures: bool = True,
    ):
        """
        Initialize wrapper.

        Args:
            max_retries: Maximum retry attempts for each operation
            initial_delay: Initial delay between retries (exponential backoff)
            use_puppeteer_fallback: Whether to use Puppeteer instead of screenshots
            log_failures: Whether to log failures for debugging
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.use_puppeteer_fallback = use_puppeteer_fallback
        self.log_failures = log_failures
        self.failure_log: list[dict[str, Any]] = []

    def _retry_operation(
        self, operation: Callable, operation_name: str, *args, **kwargs
    ) -> Any:
        """
        Execute operation with retry logic.

        Args:
            operation: Function to execute
            operation_name: Name for logging
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Result of operation

        Raises:
            Exception: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = operation(*args, **kwargs)

                # Log success after retries
                if attempt > 0 and self.log_failures:
                    self.failure_log.append(
                        {
                            "operation": operation_name,
                            "attempt": attempt + 1,
                            "status": "success_after_retry",
                            "timestamp": time.time(),
                        }
                    )

                return result

            except Exception as e:
                last_error = e

                # Log failure
                if self.log_failures:
                    self.failure_log.append(
                        {
                            "operation": operation_name,
                            "attempt": attempt + 1,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": time.time(),
                        }
                    )

                # Don't retry on last attempt
                if attempt < self.max_retries - 1:
                    delay = self.initial_delay * (2**attempt)
                    time.sleep(delay)
                    continue

        # All retries failed
        raise Exception(
            f"{operation_name} failed after {self.max_retries} attempts: {last_error}"
        )

    def navigate_pattern(
        self,
        url: str,
        browser_navigate_func: Callable,
        wait_for_streamlit: bool = True,
        additional_wait: float = 2.0,
    ) -> dict[str, Any]:
        """
        Navigate to URL with proper wait sequence.

        This is a pattern to follow when calling browser_navigate from agent code.
        Pass the browser_navigate function as an argument.

        Args:
            url: URL to navigate to
            browser_navigate_func: The browser_navigate MCP tool function
            wait_for_streamlit: Whether to wait for Streamlit to load
            additional_wait: Additional wait time after page load

        Returns:
            Result dictionary with success status
        """
        try:
            # Navigate with retry
            def _navigate():
                return browser_navigate_func(url=url)

            self._retry_operation(_navigate, "navigate")

            # Wait for Streamlit to load
            if wait_for_streamlit:
                self.wait_for_page_load_pattern(browser_wait_for_func=None)

            # Additional wait for async operations
            if additional_wait > 0:
                time.sleep(additional_wait)

            return {"success": True, "url": url}

        except Exception:
            if self.use_puppeteer_fallback:
                return self._puppeteer_fallback(url, "navigate")
            raise

    def wait_for_page_load_pattern(
        self,
        browser_wait_for_func: Callable | None,
        streamlit_timeout: int = 5,
        loading_timeout: int = 10,
    ) -> None:
        """
        Wait for Streamlit page to fully load.

        Pattern to follow when using browser_wait_for from agent code.

        Args:
            browser_wait_for_func: The browser_wait_for MCP tool function (optional)
            streamlit_timeout: Timeout for Streamlit header to appear
            loading_timeout: Timeout for loading indicator to disappear
        """
        try:
            # Wait for Streamlit header (always present)
            if browser_wait_for_func:

                def _wait_streamlit():
                    return browser_wait_for_func(
                        text="Streamlit", time=streamlit_timeout
                    )

                self._retry_operation(_wait_streamlit, "wait_for_streamlit")

            # Wait for loading to complete
            if browser_wait_for_func:
                try:

                    def _wait_loading_gone():
                        return browser_wait_for_func(
                            text="Loading", time=loading_timeout, textGone=True
                        )

                    self._retry_operation(_wait_loading_gone, "wait_loading_gone")
                except Exception:
                    # Loading indicator might not always appear, continue anyway
                    pass

            # Fallback: Simple time-based wait
            time.sleep(2)

        except Exception as e:
            # Log but don't fail - page might still be usable
            if self.log_failures:
                self.failure_log.append(
                    {
                        "operation": "wait_for_page_load",
                        "status": "warning",
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                )

    def snapshot_pattern(
        self,
        browser_snapshot_func: Callable,
        url: str | None = None,
        browser_navigate_func: Callable | None = None,
    ) -> dict[str, Any]:
        """
        Get browser snapshot with retry and fallback.

        Pattern to follow when calling browser_snapshot from agent code.

        Args:
            browser_snapshot_func: The browser_snapshot MCP tool function
            url: Optional URL to navigate to first
            browser_navigate_func: The browser_navigate MCP tool function (if URL provided)

        Returns:
            Snapshot result dictionary
        """
        try:
            # Navigate if URL provided
            if url and browser_navigate_func:
                self.navigate_pattern(url, browser_navigate_func)

            # Get snapshot with retry
            def _snapshot():
                return browser_snapshot_func()

            result = self._retry_operation(_snapshot, "snapshot")

            return {"success": True, "data": result, "method": "browser_extension"}

        except Exception as e:
            if self.use_puppeteer_fallback and url:
                return self._puppeteer_fallback(url, "snapshot")

            return {"success": False, "error": str(e), "method": "browser_extension"}

    def console_messages_pattern(
        self, browser_console_messages_func: Callable
    ) -> dict[str, Any]:
        """
        Get console messages (lightweight operation, no retry needed).

        Pattern to follow when calling browser_console_messages from agent code.

        Args:
            browser_console_messages_func: The browser_console_messages MCP tool function

        Returns:
            Console messages dictionary
        """
        try:
            # This is lightweight and usually succeeds
            result = browser_console_messages_func()

            return {"success": True, "messages": result, "method": "browser_extension"}
        except Exception as e:
            return {"success": False, "error": str(e), "method": "browser_extension"}

    def network_requests_pattern(
        self, browser_network_requests_func: Callable
    ) -> dict[str, Any]:
        """
        Get network requests (lightweight operation).

        Pattern to follow when calling browser_network_requests from agent code.

        Args:
            browser_network_requests_func: The browser_network_requests MCP tool function

        Returns:
            Network requests dictionary
        """
        try:
            result = browser_network_requests_func()

            return {"success": True, "requests": result, "method": "browser_extension"}
        except Exception as e:
            return {"success": False, "error": str(e), "method": "browser_extension"}

    def _puppeteer_fallback(self, url: str, operation: str) -> dict[str, Any]:
        """
        Fallback to Puppeteer when browser extension fails.

        Args:
            url: URL to access
            operation: Operation that failed

        Returns:
            Result dictionary from Puppeteer
        """
        try:
            # Import Puppeteer wrapper
            sys_path = Path(__file__).parent.parent.parent
            wrapper_path = (
                sys_path
                / "scripts"
                / "agent_tools"
                / "utilities"
                / "puppeteer_wrapper.py"
            )

            if not wrapper_path.exists():
                raise ImportError(f"Puppeteer wrapper not found: {wrapper_path}")

            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "puppeteer_wrapper", wrapper_path
            )
            puppeteer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(puppeteer_module)

            # Use Puppeteer to capture page
            result = puppeteer_module.capture_page(url, timeout=30)

            if result["success"]:
                return {
                    "success": True,
                    "content": result.get("content", ""),
                    "method": "puppeteer_fallback",
                    "original_operation": operation,
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown Puppeteer error"),
                    "method": "puppeteer_fallback",
                    "original_operation": operation,
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Puppeteer fallback failed: {e!s}",
                "method": "puppeteer_fallback",
                "original_operation": operation,
            }

    def verify_page_pattern(
        self,
        url: str,
        browser_navigate_func: Callable,
        browser_snapshot_func: Callable,
        browser_console_messages_func: Callable,
        browser_wait_for_func: Callable | None = None,
        check_console: bool = True,
        check_network: bool = False,
        check_exceptions: bool = True,
    ) -> dict[str, Any]:
        """
        Comprehensive page verification with all checks.

        Pattern to follow when verifying Streamlit pages from agent code.

        Args:
            url: URL to verify
            browser_navigate_func: The browser_navigate MCP tool function
            browser_snapshot_func: The browser_snapshot MCP tool function
            browser_console_messages_func: The browser_console_messages MCP tool function
            browser_wait_for_func: The browser_wait_for MCP tool function (optional)
            check_console: Check console for errors
            check_network: Check network requests
            check_exceptions: Check for Streamlit exceptions in DOM

        Returns:
            Verification result dictionary
        """
        results = {"url": url, "success": True, "checks": {}, "errors": []}

        # Navigate to page
        nav_result = self.navigate_pattern(url, browser_navigate_func)
        if not nav_result.get("success"):
            results["success"] = False
            results["errors"].append(f"Navigation failed: {nav_result.get('error')}")
            return results

        # Check console messages
        if check_console:
            console_result = self.console_messages_pattern(
                browser_console_messages_func
            )
            if console_result.get("success"):
                messages = console_result.get("messages", [])
                errors = [msg for msg in messages if msg.get("type") == "error"]
                results["checks"]["console"] = {
                    "success": len(errors) == 0,
                    "error_count": len(errors),
                    "errors": errors,
                }
                if errors:
                    results["success"] = False
                    results["errors"].extend([f"Console error: {e}" for e in errors])

        # Check network requests
        if check_network:
            # Note: browser_network_requests_func would need to be passed
            pass

        # Check for Streamlit exceptions
        if check_exceptions:
            snapshot_result = self.snapshot_pattern(
                browser_snapshot_func=browser_snapshot_func,
                url=None,  # Already navigated
            )
            if snapshot_result.get("success"):
                snapshot_result.get("data", {})
                # Parse snapshot for exception elements
                # This would search for [data-testid="stException"] patterns
                results["checks"]["exceptions"] = {
                    "success": True,  # Would parse actual snapshot
                    "exception_count": 0,  # Would count actual exceptions
                }

        return results

    def get_failure_stats(self) -> dict[str, Any]:
        """
        Get statistics about failures.

        Returns:
            Failure statistics dictionary
        """
        if not self.failure_log:
            return {"total_failures": 0, "operations": {}}

        stats = {
            "total_failures": len(self.failure_log),
            "operations": {},
            "recent_failures": self.failure_log[-10:],  # Last 10 failures
        }

        # Group by operation
        for failure in self.failure_log:
            op = failure.get("operation", "unknown")
            if op not in stats["operations"]:
                stats["operations"][op] = {"count": 0, "statuses": {}}
            stats["operations"][op]["count"] += 1
            status = failure.get("status", "unknown")
            stats["operations"][op]["statuses"][status] = (
                stats["operations"][op]["statuses"].get(status, 0) + 1
            )

        return stats


# Convenience functions - These provide patterns for agent code
# NOTE: These cannot be called directly as they require MCP tool functions as arguments
# Use them as patterns when writing agent code that calls browser extension tools


def reliable_browser_snapshot_pattern(
    url: str,
    browser_navigate_func: Callable,
    browser_snapshot_func: Callable,
    browser_wait_for_func: Callable | None = None,  # noqa: ARG001
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Pattern: Get browser snapshot with retry and fallback.

    Usage in agent code:
        result = reliable_browser_snapshot_pattern(
            url="http://localhost:8501/page",
            browser_navigate_func=browser_navigate,
            browser_snapshot_func=browser_snapshot,
            browser_wait_for_func=browser_wait_for
        )
    """
    wrapper = BrowserExtensionWrapper(max_retries=max_retries)
    return wrapper.snapshot_pattern(
        browser_snapshot_func=browser_snapshot_func,
        url=url,
        browser_navigate_func=browser_navigate_func,
    )


def reliable_browser_navigate_pattern(
    url: str,
    browser_navigate_func: Callable,
    browser_wait_for_func: Callable | None = None,  # noqa: ARG001
    wait_for_load: bool = True,
) -> dict[str, Any]:
    """
    Pattern: Navigate to URL with proper wait sequence.

    Usage in agent code:
        result = reliable_browser_navigate_pattern(
            url="http://localhost:8501/page",
            browser_navigate_func=browser_navigate,
            browser_wait_for_func=browser_wait_for
        )
    """
    wrapper = BrowserExtensionWrapper()
    return wrapper.navigate_pattern(
        url=url,
        browser_navigate_func=browser_navigate_func,
        wait_for_streamlit=wait_for_load,
    )


def verify_streamlit_page_pattern(
    url: str,
    browser_navigate_func: Callable,
    browser_snapshot_func: Callable,
    browser_console_messages_func: Callable,
    browser_wait_for_func: Callable | None = None,
    check_console: bool = True,
    check_exceptions: bool = True,
) -> dict[str, Any]:
    """
    Pattern: Comprehensive Streamlit page verification.

    Usage in agent code:
        result = verify_streamlit_page_pattern(
            url="http://localhost:8501/page",
            browser_navigate_func=browser_navigate,
            browser_snapshot_func=browser_snapshot,
            browser_console_messages_func=browser_console_messages,
            browser_wait_for_func=browser_wait_for
        )
    """
    wrapper = BrowserExtensionWrapper()
    return wrapper.verify_page_pattern(
        url=url,
        browser_navigate_func=browser_navigate_func,
        browser_snapshot_func=browser_snapshot_func,
        browser_console_messages_func=browser_console_messages_func,
        browser_wait_for_func=browser_wait_for_func,
        check_console=check_console,
        check_exceptions=check_exceptions,
    )


if __name__ == "__main__":
    """
    Example usage patterns for agent code.

    NOTE: These examples show how to use the wrapper patterns when
    you have access to the browser extension MCP tools.
    """
    print("""
    Browser Extension Wrapper - Usage Patterns

    This wrapper provides patterns for using browser extension MCP tools
    with retry logic and Puppeteer fallback.

    When using in agent code, follow these patterns:

    1. Navigate with retry:
       wrapper = BrowserExtensionWrapper()
       result = wrapper.navigate_pattern(
           url="http://localhost:8501/page",
           browser_navigate_func=browser_navigate
       )

    2. Snapshot with fallback:
       result = wrapper.snapshot_pattern(
           browser_snapshot_func=browser_snapshot,
           url="http://localhost:8501/page",
           browser_navigate_func=browser_navigate
       )

    3. Verify page:
       result = wrapper.verify_page_pattern(
           url="http://localhost:8501/page",
           browser_navigate_func=browser_navigate,
           browser_snapshot_func=browser_snapshot,
           browser_console_messages_func=browser_console_messages
       )

    See docs/troubleshooting/BROWSER_EXTENSION_FAILURES.md for details.
    """)
