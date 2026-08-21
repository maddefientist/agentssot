"""Backup freshness reporting (main._backup_status) and provenance (git_sha).

Only stats filenames/mtimes -- never opens a dump file's contents. Exercised
directly against a tmp_path fixture standing in for the read-only ./backups
mount, so no real backup service or filesystem layout is required.
"""
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@127.0.0.1/test")

from app import main


def _settings(backup_dir, stale_after_hours=36):
    return SimpleNamespace(backup_dir=str(backup_dir), backup_stale_after_hours=stale_after_hours)


def test_no_backup_dir_reports_stale_with_no_dates(tmp_path):
    missing = tmp_path / "does-not-exist"
    status = main._backup_status(_settings(missing))
    assert status["newest_dump_at"] is None
    assert status["last_success_at"] is None
    assert status["stale"] is True


def test_fresh_dump_and_last_success_reports_not_stale(tmp_path):
    (tmp_path / "ssot_20260101_000000.dump").write_bytes(b"x" * 100)
    (tmp_path / ".last_success").write_text("2026-01-01T00:00:00Z")
    status = main._backup_status(_settings(tmp_path))
    assert status["newest_dump_at"] is not None
    assert status["newest_dump_age_seconds"] is not None
    assert status["last_success_at"] is not None
    assert status["last_success_age_seconds"] is not None
    assert status["stale"] is False


def test_stale_last_success_reports_stale(tmp_path):
    (tmp_path / "ssot_old.dump").write_bytes(b"x" * 100)
    marker = tmp_path / ".last_success"
    marker.write_text("stale")
    old = time.time() - (48 * 3600)  # older than default 36h stale window
    os.utime(marker, (old, old))
    status = main._backup_status(_settings(tmp_path, stale_after_hours=36))
    assert status["stale"] is True
    assert status["last_success_age_seconds"] > 36 * 3600


def test_partial_dump_file_is_ignored_by_newest_dump_scan(tmp_path):
    """A .dump.part in-progress/failed file must never count as a completed
    backup -- only *.dump (the post-atomic-rename name) is considered."""
    (tmp_path / "ssot_inprogress.dump.part").write_bytes(b"x" * 100)
    status = main._backup_status(_settings(tmp_path))
    assert status["newest_dump_at"] is None


def test_backup_status_never_reads_dump_contents(tmp_path, monkeypatch):
    """Corrupt/garbage bytes must not raise -- only mtimes are read."""
    (tmp_path / "ssot_corrupt.dump").write_bytes(b"\x00\x01not a real pg dump\xff")
    status = main._backup_status(_settings(tmp_path))
    assert status["newest_dump_at"] is not None  # stat succeeded, no parse attempted


def test_doctor_source_never_reads_backup_file_contents():
    """Static guard: the /doctor handler's backup wiring must call _backup_status
    (stat-only) and never open()/read() a dump file."""
    src = Path(__file__).resolve().parents[1] / "app" / "main.py"
    text = src.read_text()
    assert "_backup_status(s)" in text
    doctor_body = text.split("async def doctor(", 1)[1]
    assert ".open(" not in doctor_body.split("\n\n\n", 1)[0]


def test_health_endpoint_never_mentions_backup():
    """/health must stay DB-less and backup-status-less; only /doctor reports it."""
    src = Path(__file__).resolve().parents[1] / "app" / "main.py"
    text = src.read_text()
    m = re.search(r'@app\.get\("/health"\)\s*\n\s*async def health\(.*?\n(.*?)\n\n', text, re.S)
    assert m, "could not locate /health handler body"
    assert "_backup_status" not in m.group(1)
    assert "backup" not in m.group(1)


def test_doctor_reports_git_sha_from_environment(monkeypatch):
    monkeypatch.setenv("GIT_SHA", "deadbeefcafe")
    assert os.environ.get("GIT_SHA", "unknown") == "deadbeefcafe"
    # Static guard that the field is actually read from the environment in /doctor.
    src = Path(__file__).resolve().parents[1] / "app" / "main.py"
    assert 'os.environ.get("GIT_SHA", "unknown")' in src.read_text()


def test_dockerfile_declares_git_sha_arg_label_env():
    dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile"
    text = dockerfile.read_text()
    assert re.search(r"^ARG GIT_SHA=unknown", text, re.M)
    assert "org.opencontainers.image.revision=$GIT_SHA" in text
    assert re.search(r"GIT_SHA=\$GIT_SHA", text)


def test_deploy_script_passes_exact_revision_as_build_arg():
    deploy = Path(__file__).resolve().parents[2] / "scripts" / "deploy.sh"
    text = deploy.read_text()
    assert "git rev-parse HEAD" in text
    assert "--build-arg GIT_SHA=" in text


def test_backup_compose_uses_atomic_validated_dumps_and_runtime_retention():
    compose = Path(__file__).resolve().parents[2] / "docker-compose.yml"
    text = compose.read_text()
    assert 'BACKUP_RETENTION_DAYS: ${BACKUP_RETENTION_DAYS:-30}' in text
    assert 'partfile="$${outfile}.part"' in text
    assert "umask 077" in text
    assert "chmod 700 /backups" in text
    assert 'pg_restore --list "$$partfile"' in text
    assert 'mv "$$partfile" "$$outfile"' in text
    assert "/backups/.last_success" in text
    # These values exist inside the container, so Compose must preserve the
    # dollar signs instead of substituting possibly empty host variables.
    assert '-U "$$POSTGRES_USER"' in text
    assert '-d "$$POSTGRES_DB"' in text
    assert '-mtime +"$$BACKUP_RETENTION_DAYS"' in text
