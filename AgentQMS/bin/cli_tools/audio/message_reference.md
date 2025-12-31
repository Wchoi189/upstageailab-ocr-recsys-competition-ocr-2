# Audio Message Reference for AI Agents

This reference provides pre-generated audio messages and instructions for generating custom messages using the ElevenLabs TTS API.

## Quick Reference

### Pre-Generated Messages

Choose from these categories when you need to play audio notifications:

#### Task Completion
- "Task complete."
- "All done."
- "Finished successfully."
- "That's complete."
- "Ready for the next task."

#### Process Completion
- "Process finished."
- "Operation complete."
- "All set."
- "Done and ready."

#### Success Status
- "Success."
- "Everything looks good."
- "No issues found."
- "All systems operational."

#### Progress Updates
- "Working on it."
- "Almost there."
- "Just a moment."
- "Processing now."

#### File Operations
- "File saved."
- "Download complete."
- "Export finished."
- "Upload complete."

#### Code Operations
- "Build complete."
- "Tests passed."
- "Deployment ready."
- "Code compiled successfully."

#### Warnings/Attention
- "Please check this."
- "Something needs attention."
- "Review required."
- "Take a look when you can."

#### General Status
- "All set."
- "Ready to go."
- "You're all set."
- "Standing by."

## Usage Instructions

### Option 1: Use Pre-Generated Message

1. Select a message from the categories above
2. Use the ElevenLabs TTS tool to generate audio:
   ```json
   {
     "name": "elevenlabs_text_to_speech",
     "arguments": {
       "text": "Task complete.",
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

1. Create a concise message (1-3 sentences max)
2. Keep it simple and non-technical
3. Use the ElevenLabs TTS tool to generate audio
4. Play the generated audio

**Guidelines for Custom Messages:**
- Maximum 3 sentences
- Simple, conversational language
- No technical jargon or explanations
- Focus on status or completion
- Keep it under 10 seconds of audio

## Message Templates

### Completion Template
"[Action] complete." or "All [items] finished."

### Status Template
"[Status]." or "Everything is [status]."

### Progress Template
"[Current action]." or "Working on [task]."

## Examples

**Good:**
- "All tests passed."
- "Build complete and ready."
- "Files saved successfully."

**Avoid:**
- "The compilation process has completed successfully with no errors detected in the build output."
- "I've finished executing the Python script that processes the data files."
- "The deployment pipeline has successfully pushed the changes to the production environment."

## Integration Workflow

1. **Determine message type**: Completion, status, progress, or warning
2. **Choose or create message**: Select from pre-generated or create custom
3. **Generate audio**: Use ElevenLabs TTS tool
4. **Play audio**: Use play_audio tool
5. **Clean up** (optional): Remove temporary audio files if needed

## Best Practices

- Use pre-generated messages for common scenarios
- Generate custom messages only when needed
- Keep messages ultra-concise (preferred: single sentence)
- Test audio playback after generation
- Use appropriate voice settings for clarity
