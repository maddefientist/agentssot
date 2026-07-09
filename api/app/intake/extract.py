from __future__ import annotations

import ipaddress
import logging
import os
import shutil
import socket
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

# Cap the download so a hostile/huge source can't exhaust disk/CPU before the
# timeout fires. yt-dlp aborts a stream once it exceeds this.
_MAX_FILESIZE = "500M"


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


def _host_resolves_to_internal(host: str) -> bool:
    """True if any resolved address for `host` is internal/non-routable.

    Blocks the classic SSRF targets: loopback (127.0.0.0/8, ::1), private
    ranges (10/8, 172.16/12, 192.168/16, fc00::/7), link-local incl. the cloud
    metadata endpoint (169.254.169.254 / fe80::/10), plus reserved, multicast,
    and unspecified. Resolution happens here so an IP literal and a hostname
    pointing at an internal IP are both caught. Fails closed on resolution
    error (treated as blocked by the caller).
    """
    infos = socket.getaddrinfo(host, None)
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    return False


def validate_source_url(source_url: str | None) -> str:
    """Validate a source URL for video/audio extraction.

    SSRF guard (defense-in-depth; the endpoint is API-key gated to Corra):
      - scheme MUST be http or https,
      - host MUST be present, and
      - host MUST NOT resolve to a loopback/private/link-local/reserved address
        (blocks internal services and the 169.254.169.254 metadata endpoint).

    RESIDUAL RISK (documented): yt-dlp follows HTTP redirects itself and does
    not re-run this validator on the redirect target, so a public host that
    302-redirects to an internal address is not blocked here. A full fix needs
    an SSRF-filtering fetch proxy or a pinned resolver; tracked as a follow-up.
    The download cap + API-key gate bound the blast radius for v1.
    """
    if not isinstance(source_url, str) or not source_url.strip():
        raise IntakeExtractionError("source_url is required for video/audio media")
    url = source_url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise IntakeExtractionError(
            f"unsupported URL scheme {parsed.scheme!r}; only http/https allowed"
        )
    host = parsed.hostname
    if not host:
        raise IntakeExtractionError("invalid source_url: missing host")
    try:
        blocked = _host_resolves_to_internal(host)
    except socket.gaierror as exc:
        raise IntakeExtractionError(
            f"could not resolve source_url host {host!r}"
        ) from exc
    if blocked:
        raise IntakeExtractionError(
            "source_url resolves to a disallowed internal/private address"
        )
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
        yt_dlp_bin, ffmpeg_bin = _resolve_binaries()

        with tempfile.TemporaryDirectory() as tmp:
            download_path = Path(tmp) / "download"
            wav_path = Path(tmp) / "audio.wav"

            # Resolved binary path (not a bare "yt-dlp") + hard download cap and
            # no-playlist so a single hostile URL can't fan out or exhaust disk.
            yt_dlp_argv = [
                yt_dlp_bin,
                "--no-playlist",
                "--max-filesize",
                _MAX_FILESIZE,
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