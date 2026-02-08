# Audio Playback MCP Server Configuration Guide

This guide explains how to register the Audio Playback tool with Claude Desktop's MCP (Model Context Protocol).

## Prerequisites

1. **Python 3.9+** with access to the project's scripts
2. **PulseAudio bridge** configured for WSL (see main project documentation)
3. **Audio playback tools** installed (`paplay` or `ffplay`)

## Claude Desktop Configuration

Claude Desktop reads MCP server configuration from:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

### Configuration Steps

1. **Create or edit the config file** (create if it doesn't exist):

```json
{
  "mcpServers": {
    "audio": {
      "command": "python3",
      "args": [
        "/workspaces/upstage-prompt-hack-a-thon-dev/agent_interface/tools/audio/agent_audio_mcp.py"
      ],
      "env": {
        "PYTHONPATH": "/workspaces/upstage-prompt-hack-a-thon-dev"
      }
    }
  }
}
```

2. **Adjust paths** for your system:
   - Replace `/workspaces/upstage-prompt-hack-a-thon-dev` with your actual project path
   - Use `python` instead of `python3` if that's your Python command

3. **Restart Claude Desktop** to load the new MCP server

## Available Tools

Once registered, Claude will have access to these audio tools:

### `play_audio`

Play an audio file using the PulseAudio bridge to Windows speakers.

**Parameters:**
- `audio_file` (required): Path to the audio file to play. Can be absolute or relative to project root.
- `prefer_ffplay` (optional): If true, prefer ffplay over paplay (default: false)

**Example:**
```json
{
  "name": "play_audio",
  "arguments": {
    "audio_file": "outputs/elevenlabs/All_phases_have_been_implemented_success_80203.mp3"
  }
}
```

### `get_audio_message`

Get a pre-generated audio message from a category or suggest one based on event type.

**Parameters:**
- `event_type` (optional): Event type to suggest message for (e.g., "task_complete", "build_success", "file_saved")
- `category` (optional): Message category (task_completion, process_completion, success_status, progress_updates, file_operations, code_operations, warnings, general_status)
- `index` (optional): Index to select specific message from category (default: random)

**Example:**
```json
{
  "name": "get_audio_message",
  "arguments": {
    "event_type": "build_success"
  }
}
```

### `list_audio_categories`

List all available audio message categories.

**Example:**
```json
{
  "name": "list_audio_categories",
  "arguments": {}
}
```

### `list_audio_messages`

List all messages in a specific category.

**Parameters:**
- `category` (required): Category name to list messages from

**Example:**
```json
{
  "name": "list_audio_messages",
  "arguments": {
    "category": "task_completion"
  }
}
```

### `validate_audio_message`

Validate that a custom message meets guidelines (max 3 sentences, under 200 characters).

**Parameters:**
- `message` (required): Message to validate

**Example:**
```json
{
  "name": "validate_audio_message",
  "arguments": {
    "message": "Task complete."
  }
}
```

**Common Use Cases:**
- Play generated ElevenLabs audio files: `outputs/elevenlabs/*.mp3`
- Play test audio files: `test_tone.wav`
- Play any audio file in the project

## Complete Workflow

### Option 1: Use Pre-Generated Message

1. Get a message using `get_audio_message`:
   ```json
   {
     "name": "get_audio_message",
     "arguments": {
       "event_type": "task_complete"
     }
   }
   ```

2. Generate audio with ElevenLabs TTS:
   ```json
   {
     "name": "elevenlabs_text_to_speech",
     "arguments": {
       "text": "[message from step 1]",
       "output_directory": "outputs/elevenlabs"
     }
   }
   ```

3. Play the generated audio:
   ```json
   {
     "name": "play_audio",
     "arguments": {
       "audio_file": "outputs/elevenlabs/[generated_filename].mp3"
     }
   }
   ```

### Option 2: Generate Custom Message

1. Create a custom message (max 3 sentences, under 200 characters)
2. Validate it using `validate_audio_message`
3. Generate audio with ElevenLabs TTS
4. Play the generated audio

## Integration with ElevenLabs TTS

This tool works seamlessly with the ElevenLabs TTS workflow:

1. Get or create a message
2. Generate audio with ElevenLabs TTS
3. Use `play_audio` tool to play the generated file
4. Audio plays through Windows speakers via PulseAudio bridge

## Troubleshooting

### Audio doesn't play

1. **Check PulseAudio setup**: Ensure PulseAudio server is running on Windows
2. **Verify environment variables**: Check that `PULSE_SERVER` is set correctly
3. **Test manually**: Try running `paplay` or `ffplay` directly from command line
4. **Check file path**: Ensure the audio file path is correct (absolute or relative to project root)

### Connection refused

- Verify PulseAudio server is listening on port 4713
- Check Windows firewall settings
- Ensure WSL can reach the Windows host IP
