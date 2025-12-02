#!/usr/bin/env python3
"""Simple ElevenLabs text-to-speech helper CLI.

This utility wraps the ElevenLabs REST API so you can synthesize audio
from the terminal without running the full MCP server. It relies on
the `ELEVENLABS_API_KEY` environment variable and produces an audio file
on disk (MP3 by default). Optionally, it can attempt to play the audio
using `ffplay` if available.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel (ElevenLabs default demo voice)
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OUTPUT_DIR = Path.cwd() / "outputs" / "elevenlabs"


class ElevenLabsError(RuntimeError):
    """Raised when the ElevenLabs API returns an error response."""


def synthesize_speech(
    text: str,
    api_key: str,
    *,
    voice_id: str = DEFAULT_VOICE_ID,
    model_id: str = DEFAULT_MODEL_ID,
    stability: float | None = None,
    similarity_boost: float | None = None,
    style: float | None = None,
    output_path: Path,
) -> Path:
    """Send a synthesis request and write the resulting audio to *output_path*.

    Returns the path to the file on success.
    """

    headers = {
        "accept": "audio/mpeg",
        "content-type": "application/json",
        "xi-api-key": api_key,
    }

    voice_settings: dict[str, float] = {}
    if stability is not None:
        voice_settings["stability"] = float(stability)
    if similarity_boost is not None:
        voice_settings["similarity_boost"] = float(similarity_boost)
    if style is not None:
        voice_settings["style"] = float(style)

    payload: dict[str, object] = {
        "text": text,
        "model_id": model_id,
    }
    if voice_settings:
        payload["voice_settings"] = voice_settings

    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            audio_bytes = response.read()
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(message)
            message = parsed.get("detail") or parsed.get("message") or message
        except json.JSONDecodeError:
            pass
        raise ElevenLabsError(
            f"ElevenLabs API returned status {exc.code}: {message.strip()}"
        ) from exc
    except URLError as exc:
        raise ElevenLabsError(f"Failed to reach ElevenLabs API: {exc.reason}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)
    return output_path


def attempt_playback(audio_path: Path) -> None:
    """Attempt to play the synthesized audio using the integrated audio player."""

    # Import the play_audio module from the same directory
    play_audio_script = Path(__file__).parent / "play_audio.py"
    if not play_audio_script.exists():
        print("Audio player not found – skipping playback.", file=sys.stderr)
        return

    print(f"Playing audio: {audio_path}")
    try:
        result = subprocess.run(
            [sys.executable, str(play_audio_script), str(audio_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            print("Audio playback failed (check PulseAudio setup)", file=sys.stderr)
    except OSError as exc:
        print(f"Failed to launch audio player: {exc}", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Text to synthesize (wrap in quotes)")
    parser.add_argument(
        "-v",
        "--voice-id",
        default=DEFAULT_VOICE_ID,
        help=f"ElevenLabs voice ID to use (default: {DEFAULT_VOICE_ID})",
    )
    parser.add_argument(
        "-m",
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Model ID to use (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to save the synthesized audio. "
            "Defaults to ./outputs/elevenlabs/<timestamp>.mp3"
        ),
    )
    parser.add_argument(
        "--stability",
        type=float,
        help="Optional stability parameter (0.0 – 1.0)",
    )
    parser.add_argument(
        "--similarity-boost",
        type=float,
        help="Optional similarity boost parameter (0.0 – 1.0)",
    )
    parser.add_argument(
        "--style",
        type=float,
        help="Optional style parameter (0.0 – 1.0)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Attempt to play the resulting audio with ffplay",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ELEVENLABS_API_KEY environment variable is required.", file=sys.stderr)
        return 1

    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sanitized = "_".join(args.text.strip().split())[:40] or "audio"
        filename = f"{sanitized}_{int(os.getpid())}.mp3"
        output_path = DEFAULT_OUTPUT_DIR / filename

    try:
        audio_path = synthesize_speech(
            args.text,
            api_key,
            voice_id=args.voice_id,
            model_id=args.model_id,
            stability=args.stability,
            similarity_boost=args.similarity_boost,
            style=args.style,
            output_path=output_path,
        )
    except ElevenLabsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Saved audio to {audio_path}")

    if args.play:
        attempt_playback(audio_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
