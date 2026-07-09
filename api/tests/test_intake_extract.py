from __future__ import annotations

import subprocess
from pathlib import Path

import httpx
import pytest

from app.intake import extract as extract_mod
from app.intake.extract import IntakeExtractionError, extract, validate_source_url


# ---------------------------------------------------------------------------
# validate_source_url
# ---------------------------------------------------------------------------


def test_validate_source_url_accepts_https():
    assert validate_source_url("https://example.com/a") == "https://example.com/a"


def test_validate_source_url_accepts_http():
    assert validate_source_url("http://example.com/a") == "http://example.com/a"


def test_validate_source_url_strips_whitespace():
    assert validate_source_url("  https://example.com/a  ") == "https://example.com/a"


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_validate_source_url_rejects_empty(bad):
    with pytest.raises(IntakeExtractionError):
        validate_source_url(bad)


def test_validate_source_url_rejects_non_string():
    with pytest.raises(IntakeExtractionError):
        validate_source_url(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "scheme",
    ["ftp://example.com/a", "file:///etc/passwd", "javascript:alert(1)", "gopher://x"],
)
def test_validate_source_url_rejects_non_http_scheme(scheme):
    with pytest.raises(IntakeExtractionError):
        validate_source_url(scheme)


def test_validate_source_url_rejects_missing_host():
    with pytest.raises(IntakeExtractionError):
        validate_source_url("https:///path-only")


# ---------------------------------------------------------------------------
# article / thread passthrough
# ---------------------------------------------------------------------------


def test_article_passthrough_returns_stripped_text_and_provenance():
    transcript, provenance = extract(
        source_url="https://example.com/post",
        text="  Some article body text.  ",
        media_type="article",
        stt_url="http://stt.local/transcribe",
    )
    assert transcript == "Some article body text."
    assert provenance["media_type"] == "article"
    assert provenance["source_url"] == "https://example.com/post"
    assert "T" in provenance["captured_at"]  # ISO8601-ish UTC


def test_thread_passthrough_returns_stripped_text():
    transcript, provenance = extract(
        source_url=None,
        text="thread content\n\n",
        media_type="thread",
        stt_url="http://stt.local/transcribe",
    )
    assert transcript == "thread content"
    assert provenance["media_type"] == "thread"
    assert provenance["source_url"] is None


@pytest.mark.parametrize("media_type", ["article", "thread"])
def test_article_thread_empty_text_raises(media_type):
    with pytest.raises(IntakeExtractionError):
        extract(source_url="https://x", text="", media_type=media_type, stt_url="http://s")


@pytest.mark.parametrize("media_type", ["article", "thread"])
def test_article_thread_none_text_raises(media_type):
    with pytest.raises(IntakeExtractionError):
        extract(source_url="https://x", text=None, media_type=media_type, stt_url="http://s")


# ---------------------------------------------------------------------------
# video / audio: missing source_url, invalid scheme
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("media_type", ["video", "audio"])
def test_video_audio_missing_source_url_raises(media_type):
    with pytest.raises(IntakeExtractionError):
        extract(source_url=None, text=None, media_type=media_type, stt_url="http://s")


@pytest.mark.parametrize("media_type", ["video", "audio"])
def test_video_audio_invalid_scheme_raises(media_type):
    with pytest.raises(IntakeExtractionError):
        extract(
            source_url="ftp://example.com/v.mp4",
            text=None,
            media_type=media_type,
            stt_url="http://s",
        )


# ---------------------------------------------------------------------------
# missing binaries
# ---------------------------------------------------------------------------


def test_missing_yt_dlp_raises(monkeypatch):
    monkeypatch.setattr(extract_mod.shutil, "which", lambda name: None)
    with pytest.raises(IntakeExtractionError):
        extract(
            source_url="https://example.com/v.mp4",
            text=None,
            media_type="video",
            stt_url="http://s",
        )


def test_missing_ffmpeg_raises(monkeypatch):
    def fake_which(name):
        if name == "yt-dlp":
            return "/usr/bin/yt-dlp"
        return None  # ffmpeg not on PATH

    monkeypatch.setattr(extract_mod.shutil, "which", fake_which)
    # Force the /usr/bin/ffmpeg fallback to also appear missing.
    monkeypatch.setattr(extract_mod.os.path, "exists", lambda p: False if p == "/usr/bin/ffmpeg" else True)

    with pytest.raises(IntakeExtractionError):
        extract(
            source_url="https://example.com/v.mp4",
            text=None,
            media_type="video",
            stt_url="http://s",
        )


# ---------------------------------------------------------------------------
# mocked subprocess + httpx success path
# ---------------------------------------------------------------------------


def _fake_which_present(name):
    if name == "yt-dlp":
        return "/usr/bin/yt-dlp"
    if name == "ffmpeg":
        return "/usr/bin/ffmpeg"
    return None


def _make_subprocess_that_writes_wav():
    """yt-dlp succeeds with no file written; ffmpeg writes a dummy wav to
    argv[-1] so the real STT file-open works."""
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        assert kwargs.get("shell") is False
        assert kwargs.get("check") is False
        assert kwargs.get("capture_output") is True
        if argv[0] == "ffmpeg" or argv[0].endswith("ffmpeg"):
            wav = Path(argv[-1])
            wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE dummy")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=b"", stderr=b"")

    return fake_run, calls


def test_video_success_mocked_returns_transcript(monkeypatch):
    monkeypatch.setattr(extract_mod.shutil, "which", _fake_which_present)
    fake_run, calls = _make_subprocess_that_writes_wav()
    monkeypatch.setattr(extract_mod.subprocess, "run", fake_run)

    captured: dict = {}

    def fake_post(url, files=None, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        # files is a dict; ensure the multipart file tuple shape is correct.
        assert "file" in files
        fname, fh, ctype = files["file"]
        assert fname == "audio.wav"
        assert ctype == "audio/wav"
        return httpx.Response(200, json={"text": "  hello world transcript  "})

    monkeypatch.setattr(extract_mod.httpx, "post", fake_post)

    transcript, provenance = extract(
        source_url="https://example.com/v.mp4",
        text=None,
        media_type="video",
        stt_url="http://stt.local/transcribe",
    )

    assert transcript == "hello world transcript"
    assert provenance["media_type"] == "video"
    assert provenance["source_url"] == "https://example.com/v.mp4"
    assert "T" in provenance["captured_at"]
    assert captured["url"] == "http://stt.local/transcribe"
    assert captured["timeout"] == 300

    # Two subprocess invocations, both with shell=False and list argv.
    assert len(calls) == 2
    assert calls[0][0] == "yt-dlp"
    assert calls[0][1:4] == ["-f", "bestaudio/best", "-o"]
    assert calls[0][-1] == "https://example.com/v.mp4"
    assert calls[1][0] == "/usr/bin/ffmpeg"
    assert calls[1][1] == "-y"
    assert calls[1][2:4] == ["-i", calls[0][4]]  # -i <download_path>
    assert calls[1][4:8] == ["-ar", "16000", "-ac", "1"]
    assert calls[1][-1] == calls[0][4].replace("download", "audio.wav") or calls[1][-1].endswith("audio.wav")


def test_audio_success_uses_transcript_key(monkeypatch):
    monkeypatch.setattr(extract_mod.shutil, "which", _fake_which_present)
    fake_run, _ = _make_subprocess_that_writes_wav()
    monkeypatch.setattr(extract_mod.subprocess, "run", fake_run)

    monkeypatch.setattr(
        extract_mod.httpx,
        "post",
        lambda url, files=None, timeout=None: httpx.Response(
            200, json={"transcript": "via transcript key"}
        ),
    )

    transcript, _ = extract(
        source_url="https://example.com/a.mp3",
        text=None,
        media_type="audio",
        stt_url="http://stt.local/transcribe",
    )
    assert transcript == "via transcript key"


def test_stt_result_key_fallback(monkeypatch):
    monkeypatch.setattr(extract_mod.shutil, "which", _fake_which_present)
    fake_run, _ = _make_subprocess_that_writes_wav()
    monkeypatch.setattr(extract_mod.subprocess, "run", fake_run)

    monkeypatch.setattr(
        extract_mod.httpx,
        "post",
        lambda url, files=None, timeout=None: httpx.Response(
            200, json={"result": "via result key"}
        ),
    )

    transcript, _ = extract(
        source_url="https://example.com/v",
        text=None,
        media_type="video",
        stt_url="http://stt.local/transcribe",
    )
    assert transcript == "via result key"


# ---------------------------------------------------------------------------
# mocked failures -> typed errors
# ---------------------------------------------------------------------------


def test_yt_dlp_nonzero_raises(monkeypatch):
    monkeypatch.setattr(extract_mod.shutil, "which", _fake_which_present)

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            args=argv, returncode=1, stdout=b"", stderr=b"download error boom"
        )

    monkeypatch.setattr(extract_mod.subprocess, "run", fake_run)

    with pytest.raises(IntakeExtractionError, match="yt-dlp failed"):
        extract(
            source_url="https://example.com/v",
            text=None,
            media_type="video",
            stt_url="http://stt.local/transcribe",
        )


def test_ffmpeg_nonzero_raises(monkeypatch):
    monkeypatch.setattr(extract_mod.shutil, "which", _fake_which_present)
    call_count = {"n": 0}

    def fake_run(argv, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        return subprocess.CompletedProcess(argv, 2, b"", b"ffmpeg codec error")

    monkeypatch.setattr(extract_mod.subprocess, "run", fake_run)

    with pytest.raises(IntakeExtractionError, match="ffmpeg failed"):
        extract(
            source_url="https://example.com/v",
            text=None,
            media_type="video",
            stt_url="http://stt.local/transcribe",
        )


def test_subprocess_timeout_raises(monkeypatch):
    monkeypatch.setattr(extract_mod.shutil, "which", _fake_which_present)

    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=1)

    monkeypatch.setattr(extract_mod.subprocess, "run", fake_run)

    with pytest.raises(IntakeExtractionError, match="timed out"):
        extract(
            source_url="https://example.com/v",
            text=None,
            media_type="video",
            stt_url="http://stt.local/transcribe",
        )


def test_stt_non_200_raises(monkeypatch):
    monkeypatch.setattr(extract_mod.shutil, "which", _fake_which_present)
    fake_run, _ = _make_subprocess_that_writes_wav()
    monkeypatch.setattr(extract_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(
        extract_mod.httpx,
        "post",
        lambda url, files=None, timeout=None: httpx.Response(503, text="stt down"),
    )

    with pytest.raises(IntakeExtractionError, match="HTTP 503"):
        extract(
            source_url="https://example.com/v",
            text=None,
            media_type="video",
            stt_url="http://stt.local/transcribe",
        )


def test_stt_empty_transcript_raises(monkeypatch):
    monkeypatch.setattr(extract_mod.shutil, "which", _fake_which_present)
    fake_run, _ = _make_subprocess_that_writes_wav()
    monkeypatch.setattr(extract_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(
        extract_mod.httpx,
        "post",
        lambda url, files=None, timeout=None: httpx.Response(200, json={"text": "   "}),
    )

    with pytest.raises(IntakeExtractionError, match="empty transcript"):
        extract(
            source_url="https://example.com/v",
            text=None,
            media_type="video",
            stt_url="http://stt.local/transcribe",
        )


def test_stt_missing_all_keys_raises(monkeypatch):
    monkeypatch.setattr(extract_mod.shutil, "which", _fake_which_present)
    fake_run, _ = _make_subprocess_that_writes_wav()
    monkeypatch.setattr(extract_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(
        extract_mod.httpx,
        "post",
        lambda url, files=None, timeout=None: httpx.Response(200, json={"unrelated": "x"}),
    )

    with pytest.raises(IntakeExtractionError, match="empty transcript"):
        extract(
            source_url="https://example.com/v",
            text=None,
            media_type="video",
            stt_url="http://stt.local/transcribe",
        )


def test_stt_non_json_response_raises(monkeypatch):
    monkeypatch.setattr(extract_mod.shutil, "which", _fake_which_present)
    fake_run, _ = _make_subprocess_that_writes_wav()
    monkeypatch.setattr(extract_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(
        extract_mod.httpx,
        "post",
        lambda url, files=None, timeout=None: httpx.Response(
            200, text="<html>not json</html>", headers={"content-type": "text/html"}
        ),
    )

    with pytest.raises(IntakeExtractionError, match="non-JSON"):
        extract(
            source_url="https://example.com/v",
            text=None,
            media_type="video",
            stt_url="http://stt.local/transcribe",
        )


# ---------------------------------------------------------------------------
# unsupported media type
# ---------------------------------------------------------------------------


def test_unsupported_media_type_raises():
    with pytest.raises(IntakeExtractionError, match="unsupported media_type"):
        extract(
            source_url=None,
            text="x",
            media_type="podcast",  # type: ignore[arg-type]
            stt_url="http://s",
        )


# ---------------------------------------------------------------------------
# no real network / GPU: prove httpx + subprocess are patched at module level
# ---------------------------------------------------------------------------


def test_no_real_network_calls_in_test_module(monkeypatch):
    """Sanity: any httpx.post in this test goes through the monkeypatched fake,
    and subprocess.run never reaches the OS. Asserts the module symbols are
    patchable (i.e. we patch module attrs, not global builtins)."""
    sentinel_post = {"called": False}
    sentinel_run = {"called": False}

    monkeypatch.setattr(extract_mod.shutil, "which", _fake_which_present)

    def fake_post(url, files=None, timeout=None):
        sentinel_post["called"] = True
        return httpx.Response(200, json={"text": "ok"})

    def fake_run(argv, **kwargs):
        sentinel_run["called"] = True
        if argv[0].endswith("ffmpeg") or argv[0] == "ffmpeg":
            Path(argv[-1]).write_bytes(b"dummy")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(extract_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(extract_mod.httpx, "post", fake_post)

    transcript, _ = extract(
        source_url="https://example.com/v",
        text=None,
        media_type="video",
        stt_url="http://stt.local/transcribe",
    )
    assert transcript == "ok"
    assert sentinel_post["called"] is True
    assert sentinel_run["called"] is True