#!/usr/bin/env python3
"""Audio playback utility for WSL with PulseAudio bridge support.

This utility plays audio files using the PulseAudio bridge to Windows,
automatically setting up the required environment variables.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_windows_host_ip() -> str | None:
    """Detect Windows host IP from WSL resolver configuration."""
    resolv_path = Path("/etc/resolv.conf")
    if not resolv_path.exists():
        return None

    try:
        with resolv_path.open() as f:
            for line in f:
                line = line.strip()
                if line.startswith("nameserver"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
    except (OSError, ValueError):
        pass

    return None


def setup_pulseaudio_env() -> dict[str, str]:
    """Set up PulseAudio environment variables for WSL bridge."""
    env = os.environ.copy()

    # If PULSE_SERVER is already set (e.g., from .bashrc), use it
    if "PULSE_SERVER" in env:
        return env

    # Check if we're in WSL
    if not Path("/proc/version").exists():
        return env

    try:
        with Path("/proc/version").open() as f:
            version_info = f.read().lower()
            if "microsoft" not in version_info and "wsl" not in version_info:
                return env
    except OSError:
        return env

    # Get Windows host IP
    windows_host = get_windows_host_ip()
    if not windows_host:
        return env

    # Set PulseAudio environment variables
    env["WINDOWS_HOST"] = windows_host
    env["PULSE_SERVER"] = f"tcp:{windows_host}:4713"
    env["PULSE_LATENCY_MSEC"] = "30"

    return env


def play_with_paplay(audio_path: Path, env: dict[str, str]) -> bool:
    """Attempt to play audio using paplay (PulseAudio client)."""
    paplay = shutil.which("paplay")
    if not paplay:
        return False

    try:
        subprocess.run(
            [paplay, str(audio_path)],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def play_with_ffplay(audio_path: Path, env: dict[str, str]) -> bool:
    """Attempt to play audio using ffplay (fallback)."""
    ffplay = shutil.which("ffplay")
    if not ffplay:
        return False

    try:
        subprocess.run(
            [ffplay, "-autoexit", "-nodisp", str(audio_path)],
            env=env,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except OSError:
        return False


def play_audio(audio_path: Path, prefer_pulse: bool = True) -> bool:
    """Play an audio file using available players.

    Args:
        audio_path: Path to the audio file to play
        prefer_pulse: If True, prefer paplay over ffplay

    Returns:
        True if playback succeeded, False otherwise
    """
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        return False

    env = setup_pulseaudio_env()

    # Try paplay first if preferred (better PulseAudio integration)
    if prefer_pulse and play_with_paplay(audio_path, env):
        return True

    # Fallback to ffplay
    if play_with_ffplay(audio_path, env):
        return True

    # If paplay wasn't tried yet, try it now
    if not prefer_pulse and play_with_paplay(audio_path, env):
        return True

    print("Error: No audio player available (paplay or ffplay)", file=sys.stderr)
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the audio file to play",
    )
    parser.add_argument(
        "--prefer-ffplay",
        action="store_true",
        help="Prefer ffplay over paplay (default: prefer paplay)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    audio_path = args.audio_file
    if not audio_path.is_absolute():
        audio_path = Path.cwd() / audio_path

    if args.verbose:
        env = setup_pulseaudio_env()
        if "PULSE_SERVER" in env:
            print(f"Using PulseAudio server: {env['PULSE_SERVER']}", file=sys.stderr)
        else:
            print("PulseAudio bridge not configured", file=sys.stderr)

    success = play_audio(audio_path, prefer_pulse=not args.prefer_ffplay)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
