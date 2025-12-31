#!/bin/bash
# Script to play MP3 files from ElevenLabs output directory

OUTPUT_DIR="/workspaces/upstage-prompt-hack-a-thon-dev/outputs/elevenlabs"
MP3_FILES="$OUTPUT_DIR/*.mp3"

# Check if directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory $OUTPUT_DIR does not exist."
    exit 1
fi

# Check if there are MP3 files
if ! compgen -G "$MP3_FILES" > /dev/null; then
    echo "No MP3 files found in $OUTPUT_DIR."
    exit 1
fi

# Play the files
mpg123 $MP3_FILES
