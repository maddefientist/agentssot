from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

MediaType = Literal["video", "audio", "article", "thread"]

# Subprocess ceilings. The plan specifies "timeout" without numeric values;
# these are conservative defaults so a hung download/encode fails loudly instead
# of blocking the request forever.
_DOWNLOAD_TIMEOUT = 600  # seconds — yt-dlp fetch
_FFMPEG_TIMEOUT = 120  # seconds — ffmpeg wav conversion
_STT_TIMEOUT = 300  # seconds — matches plan STT multipart timeout


class IntakeExtractionError(RuntimeError):
    """Typed error for any failure in the extraction adapter.

    Boundary-level only: input validation, missing binaries, subprocess
    failure, or STT service error. Internal control flow never raises this.
    """


def _make_provenance(source_url: str | None, media_type: MediaType) -> dict:
    return {
        "source_url": source_url,
        "media_type": media_type,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_source_url(source_url: str | None) -> str:
    """Validate a source URL for video/audio extraction.

    SSRF guard: scheme MUST be http or https. Empty/None/blank rejected.
    """
    if not isinstance(source_url, str) or not source_url.strip():
        raise IntakeExtractionError("source_url is required for video/audio media")
    url = source_url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise IntakeExtractionError(
            f"unsupported URL scheme {parsed.scheme!r}; only http/https allowed"
        )
    if not parsed.netloc:
        raise IntakeExtractionError("invalid source_url: missing host")
    return url


def _run_subprocess(argv: list[str], *, timeout: int, label: str) -> None:
    """Run a subprocess with shell=False, list argv, check=False, captured IO.

    Non-zero exit or timeout -> IntakeExtractionError with stderr excerpt.
    """
    logger.debug("running subprocess %s", argv)
    try:
        result = subprocess.run(
            argv,
            shell=False,
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise IntakeExtractionError(
            f"{label} timed out after {timeout}s"
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        excerpt = stderr[-500:] if stderr else "(no stderr captured)"
        raise IntakeExtractionError(
            f"{label} failed (exit {result.returncode}): {excerpt}"
        )


def _resolve_binaries() -> tuple[str, str]:
    """Resolve yt-dlp and ffmpeg binaries; raise typed if missing.

    yt-dlp must be on PATH. ffmpeg uses PATH then falls back to /usr/bin/ffmpeg;
    if neither exists, raise.
    """
    yt_dlp_bin = shutil.which("yt-dlp")
    if not yt_dlp_bin:
        raise IntakeExtractionError("yt-dlp binary not found on PATH")
    ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    if not os.path.exists(ffmpeg_bin):
        raise IntakeExtractionError("ffmpeg binary not found")
    return yt_dlp_bin, ffmpeg_bin


def _transcribe(wav_path: Path, stt_url: str) -> str:
    """POST the wav to the STT service and return a non-empty transcript.

    Tries JSON keys in order: text, transcript, result. Raises on empty.
    """
    with open(wav_path, "rb") as fh:
        response = httpx.post(
            stt_url,
            files={"file": ("audio.wav", fh, "audio/wav")},
            timeout=_STT_TIMEOUT,
        )

    if response.status_code != 200:
        body = response.text[:200] if response.text else "(empty body)"
        raise IntakeExtractionError(
            f"STT service returned HTTP {response.status_code}: {body}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        body = response.text[:200] if response.text else "(empty body)"
        raise IntakeExtractionError(
            f"STT service returned non-JSON response: {body}"
        ) from exc

    transcript = ""
    if isinstance(data, dict):
        for key in ("text", "transcript", "result"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                transcript = value.strip()
                break

    if not transcript:
        raise IntakeExtractionError("STT service returned empty transcript")
    return transcript


def extract(
    source_url: str | None,
    text: str | None,
    media_type: MediaType,
    *,
    stt_url: str,
) -> tuple[str, dict]:
    """Extract a transcript from a source.

    article/thread: pass through stripped `text` (required, non-empty).
    video/audio: download via yt-dlp, convert to 16kHz mono wav via ffmpeg,
    then POST to the STT service. Never returns a silent empty transcript.
    """
    if media_type in ("article", "thread"):
        if not isinstance(text, str) or not text.strip():
            raise IntakeExtractionError(
                f"text is required for {media_type} media"
            )
        return text.strip(), _make_provenance(source_url, media_type)

    if media_type in ("video", "audio"):
        url = validate_source_url(source_url)
        _, ffmpeg_bin = _resolve_binaries()

        with tempfile.TemporaryDirectory() as tmp:
            download_path = Path(tmp) / "download"
            wav_path = Path(tmp) / "audio.wav"

            yt_dlp_argv = [
                "yt-dlp",
                "-f",
                "bestaudio/best",
                "-o",
                str(download_path),
                url,
            ]
            _run_subprocess(
                yt_dlp_argv,
                timeout=_DOWNLOAD_TIMEOUT,
                label="yt-dlp",
            )

            ffmpeg_argv = [
                ffmpeg_bin,
                "-y",
                "-i",
                str(download_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(wav_path),
            ]
            _run_subprocess(
                ffmpeg_argv,
                timeout=_FFMPEG_TIMEOUT,
                label="ffmpeg",
            )

            transcript = _transcribe(wav_path, stt_url)

        return transcript, _make_provenance(url, media_type)

    raise IntakeExtractionError(f"unsupported media_type: {media_type!r}")