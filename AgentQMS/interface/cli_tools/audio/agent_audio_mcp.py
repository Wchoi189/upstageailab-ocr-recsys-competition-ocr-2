#!/usr/bin/env python3
"""
Audio Playback MCP Server for Claude Desktop

This MCP server exposes audio playback capabilities to Claude.
It wraps the play_audio.py utility for MCP protocol compatibility.
"""

import json
import sys
from pathlib import Path
from typing import Any

from AgentQMS.toolkit.utils.runtime import ensure_project_root_on_sys_path
from agent.tools.audio.message_templates import (
    get_message,
    get_random_message,
    list_categories,
    list_messages,
    suggest_message,
    validate_message,
)

PROJECT_ROOT = ensure_project_root_on_sys_path()
PLAY_AUDIO_PATH = PROJECT_ROOT / "AgentQMS" / "scripts" / "maintenance" / "play_audio.py"

if PLAY_AUDIO_PATH.exists():
    import importlib.util

    spec = importlib.util.spec_from_file_location("play_audio", PLAY_AUDIO_PATH)
    if spec and spec.loader:
        play_audio_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(play_audio_module)
        play_audio = play_audio_module.play_audio
    else:
        raise ImportError(f"Failed to load play_audio from {PLAY_AUDIO_PATH}")
else:
    raise ImportError(f"Could not find play_audio at {PLAY_AUDIO_PATH}")


class AudioMCPServer:
    """MCP Server wrapper for audio playback."""

    def __init__(self):
        # Store project root for path resolution
        # Use global project root set at module level
        self.project_root = PROJECT_ROOT

        self.tools = {
            "play_audio": {
                "name": "play_audio",
                "description": "Play an audio file using the PulseAudio bridge to Windows speakers. Supports MP3, WAV, and other common audio formats. Automatically detects WSL environment and configures PulseAudio connection.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_file": {
                            "type": "string",
                            "description": "Path to the audio file to play. Can be absolute or relative to the project root. Common locations: outputs/elevenlabs/*.mp3",
                        },
                        "prefer_ffplay": {
                            "type": "boolean",
                            "description": "If true, prefer ffplay over paplay (default: false, prefers paplay for better PulseAudio integration)",
                            "default": False,
                        },
                    },
                    "required": ["audio_file"],
                },
            },
            "get_audio_message": {
                "name": "get_audio_message",
                "description": "Get a pre-generated audio message from a category or suggest one based on event type. Returns message text that can be used with ElevenLabs TTS.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Message category: task_completion, process_completion, success_status, progress_updates, file_operations, code_operations, warnings, general_status",
                        },
                        "event_type": {
                            "type": "string",
                            "description": "Event type to suggest message for (e.g., 'task_complete', 'build_success', 'file_saved'). If provided, category is ignored.",
                        },
                        "index": {
                            "type": "integer",
                            "description": "Optional index to select specific message from category (default: random)",
                        },
                    },
                },
            },
            "list_audio_categories": {
                "name": "list_audio_categories",
                "description": "List all available audio message categories.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            "list_audio_messages": {
                "name": "list_audio_messages",
                "description": "List all messages in a specific category.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Category name to list messages from",
                        },
                    },
                    "required": ["category"],
                },
            },
            "validate_audio_message": {
                "name": "validate_audio_message",
                "description": "Validate that a custom message meets guidelines (max 3 sentences, under 200 characters).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to validate",
                        },
                    },
                    "required": ["message"],
                },
            },
        }

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP protocol request (JSON-RPC format)."""
        request_id = request.get("id")
        method = request.get("method")
        params: dict[str, Any] = request.get("params") or {}

        def make_response(
            result: Any = None, error: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            """Create JSON-RPC response."""
            response = {"jsonrpc": "2.0", "id": request_id}
            if error:
                response["error"] = error
            else:
                response["result"] = result
            return response

        if method == "initialize":
            return make_response(
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "audio-mcp-server", "version": "1.0.0"},
                }
            )

        elif method == "tools/list":
            return make_response({"tools": list(self.tools.values())})

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            try:
                if tool_name == "play_audio":
                    audio_file = arguments.get("audio_file")
                    prefer_ffplay = arguments.get("prefer_ffplay", False)

                    # Resolve path relative to project root if not absolute
                    audio_path = Path(audio_file)
                    if not audio_path.is_absolute():
                        audio_path = self.project_root / audio_path

                    # Play the audio
                    success = play_audio(audio_path, prefer_pulse=not prefer_ffplay)

                    if success:
                        return make_response(
                            {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Successfully played audio file: {audio_path}",
                                    }
                                ]
                            }
                        )
                    else:
                        return make_response(
                            error={
                                "code": -32603,
                                "message": f"Failed to play audio file: {audio_path}. Check PulseAudio setup and ensure the file exists.",
                            }
                        )

                elif tool_name == "get_audio_message":
                    event_type = arguments.get("event_type")
                    category = arguments.get("category")
                    index = arguments.get("index")

                    if event_type:
                        message = suggest_message(event_type)
                    elif category:
                        message = get_message(category, index)
                    else:
                        message = get_random_message()

                    return make_response(
                        {
                            "content": [
                                {
                                    "type": "text",
                                    "text": message,
                                }
                            ]
                        }
                    )

                elif tool_name == "list_audio_categories":
                    categories = list_categories()
                    return make_response(
                        {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(categories, indent=2),
                                }
                            ]
                        }
                    )

                elif tool_name == "list_audio_messages":
                    category = arguments.get("category")
                    messages = list_messages(category)
                    return make_response(
                        {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(messages, indent=2),
                                }
                            ]
                        }
                    )

                elif tool_name == "validate_audio_message":
                    message = arguments.get("message")
                    is_valid = validate_message(message)
                    return make_response(
                        {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {"valid": is_valid, "message": message},
                                        indent=2,
                                    ),
                                }
                            ]
                        }
                    )

                else:
                    return make_response(
                        error={"code": -32601, "message": f"Unknown tool: {tool_name}"}
                    )

            except Exception as e:
                return make_response(
                    error={"code": -32603, "message": f"Internal error: {e!s}"}
                )

        else:
            return make_response(
                error={"code": -32601, "message": f"Unknown method: {method}"}
            )


def main():
    """Main entry point for MCP server."""
    server = AudioMCPServer()

    # Read from stdin (MCP protocol uses JSON-RPC over stdio)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = server.handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e!s}"},
            }
            print(json.dumps(error_response))
            sys.stdout.flush()
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": f"Internal error: {e!s}"},
            }
            print(json.dumps(error_response))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
