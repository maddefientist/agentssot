import os
from pathlib import Path
import subprocess
import sys

import pytest


API_DIR = Path(__file__).resolve().parents[1]
ENTRYPOINT = API_DIR / "container_entrypoint.py"


def _load_entrypoint():
    import importlib.util

    spec = importlib.util.spec_from_file_location("container_entrypoint", ENTRYPOINT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gateway_rejects_uvicorn_cli_multiworker_before_server_start():
    env = os.environ.copy()
    env["GATEWAY_ENABLED"] = "true"
    env.pop("WEB_CONCURRENCY", None)
    env.pop("UVICORN_WORKERS", None)

    result = subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT),
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--workers",
            "2",
        ],
        cwd=API_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 78
    assert "requires exactly one API worker" in result.stderr
    assert "Uvicorn running" not in result.stderr


def test_gateway_direct_app_import_requires_owned_launcher_attestation():
    env = os.environ.copy()
    env.update(
        {
            "DATABASE_URL": "postgresql+psycopg://test:test@127.0.0.1/test",
            "GATEWAY_ENABLED": "true",
        }
    )
    env.pop("AGENTSSOT_LAUNCHER_ATTESTATION", None)
    result = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=API_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert "requires the owned container entrypoint" in result.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["uvicorn", "app.main:app", "--workers=3"],
        ["gunicorn", "app.main:app", "-w", "1"],
        ["uvicorn", "other.main:app"],
    ],
)
def test_gateway_rejects_unsafe_or_unknown_launchers(argv):
    entrypoint = _load_entrypoint()
    with pytest.raises(RuntimeError):
        entrypoint.validate_command(argv, {"GATEWAY_ENABLED": "true"})


def test_gateway_allows_owned_single_worker_command(monkeypatch):
    entrypoint = _load_entrypoint()
    executed = []
    monkeypatch.setattr(entrypoint.os, "execvp", lambda executable, argv: executed.append((executable, argv)))

    assert entrypoint.main(
        ["uvicorn", "app.main:app", "--workers", "1"],
        {"GATEWAY_ENABLED": "true"},
    ) == 0
    assert executed == [("uvicorn", ["uvicorn", "app.main:app", "--workers", "1"])]


def test_gateway_disabled_does_not_restrict_container_command(monkeypatch):
    entrypoint = _load_entrypoint()
    executed = []
    monkeypatch.setattr(entrypoint.os, "execvp", lambda executable, argv: executed.append((executable, argv)))

    assert entrypoint.main(["python", "-V"], {"GATEWAY_ENABLED": "false"}) == 0
    assert executed == [("python", ["python", "-V"])]
