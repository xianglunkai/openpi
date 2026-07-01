#!/usr/bin/env python3
"""Lightweight cross-platform text-to-speech helpers."""

from __future__ import annotations

import logging
import platform
import subprocess

_VOICE_AVAILABLE = True
_VOICE_ERROR_REPORTED = False


def say(text: str, blocking: bool = False) -> None:
    system = platform.system()

    if system == "Darwin":
        cmd = ["say", text]
    elif system == "Linux":
        cmd = ["spd-say", text]
        if blocking:
            cmd.append("--wait")
    elif system == "Windows":
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]
    else:
        raise RuntimeError("Unsupported operating system for text-to-speech.")

    if blocking:
        subprocess.run(cmd, check=True)
    else:
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if system == "Windows" else 0)


def log_say(text: str, play_sounds: bool = True, blocking: bool = False) -> None:
    """Log a message and optionally play it via system TTS."""
    global _VOICE_AVAILABLE
    global _VOICE_ERROR_REPORTED

    logging.info(text)
    if not play_sounds or not _VOICE_AVAILABLE:
        return

    try:
        say(text, blocking=blocking)
    except Exception as exc:
        if not _VOICE_ERROR_REPORTED:
            logging.warning("TTS disabled after failure: %s", exc)
            _VOICE_ERROR_REPORTED = True
        _VOICE_AVAILABLE = False
