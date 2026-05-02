"""SessionStart loadout hook — happy path + timeout fallback."""
import os
import subprocess
from pathlib import Path

import pytest

HOOK = Path("~/.claude/plugins/hari-hive/hooks/SessionStart.md").expanduser()


def _extract_bash(md_text: str) -> str:
    # Pull the first ```bash block out of the hook markdown.
    block = md_text.split("```bash", 1)[1].split("```", 1)[0]
    return block.lstrip("\n")


@pytest.mark.integration
def test_hook_emits_hive_block_on_success(tmp_path):
    script = tmp_path / "hook.sh"
    script.write_text(_extract_bash(HOOK.read_text()))
    out = subprocess.run(
        ["bash", str(script)],
        capture_output=True, text=True, timeout=8,
        env={**os.environ, "PWD": "/opt/agentssot"},
    )
    assert "<hive-loadout>" in out.stdout
    assert "</hive-loadout>" in out.stdout


@pytest.mark.integration
def test_hook_falls_back_under_timeout(tmp_path, monkeypatch):
    """If HIVE_API_BASE is bogus, hook must still emit the static fallback within 3s."""
    script = tmp_path / "hook.sh"
    script.write_text(_extract_bash(HOOK.read_text()))
    out = subprocess.run(
        ["bash", str(script)],
        capture_output=True, text=True, timeout=4,
        env={**os.environ, "HIVE_API_BASE": "http://127.0.0.1:1"},
    )
    assert "<hive-available>" in out.stdout  # static fallback marker
