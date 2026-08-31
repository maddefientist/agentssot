"""SessionStart hook stays lightweight and does not perform automatic recall."""
import os
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / "app" / "plugin" / "hooks" / "SessionStart.md"


def _extract_bash(md_text: str) -> str:
    # Pull the first ```bash block out of the hook markdown.
    block = md_text.split("```bash", 1)[1].split("```", 1)[0]
    return block.lstrip("\n")


@pytest.mark.integration
def test_hook_emits_bounded_hive_hint(tmp_path):
    script = tmp_path / "hook.sh"
    script.write_text(_extract_bash(HOOK.read_text()))
    out = subprocess.run(
        ["bash", str(script)],
        capture_output=True, text=True, timeout=8,
        env={**os.environ, "PWD": "/opt/agentssot"},
    )
    assert "<hive-available>" in out.stdout
    assert "</hive-available>" in out.stdout
    assert "Use hive_recall only when" in out.stdout
    assert "<hive-loadout>" not in out.stdout


@pytest.mark.integration
def test_hook_does_not_contact_the_api(tmp_path):
    script = tmp_path / "hook.sh"
    script.write_text(_extract_bash(HOOK.read_text()))
    out = subprocess.run(
        ["bash", str(script)],
        capture_output=True, text=True, timeout=4,
        env={**os.environ, "HIVE_API_BASE": "http://127.0.0.1:1"},
    )
    assert out.returncode == 0
    assert "<hive-available>" in out.stdout
