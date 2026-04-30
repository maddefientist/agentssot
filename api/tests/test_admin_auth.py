"""MCP plugin admin-auth resolution.

Unit-test the auth helper directly (no live API needed).
"""
import json
import os
import sys
from pathlib import Path

import pytest

# Make the plugin importable
PLUGIN_PATH = Path("~/.claude/plugins/hari-hive").expanduser()
sys.path.insert(0, str(PLUGIN_PATH))


def test_admin_role_requires_admin_json(tmp_path, monkeypatch):
    """If admin.json is missing, requesting admin role raises PermissionError."""
    fake_home = tmp_path / "home"
    (fake_home / ".claude/agentssot/local").mkdir(parents=True)
    (fake_home / ".claude/agentssot/local/agent.json").write_text(
        json.dumps({"api_key": "writer-key"})
    )
    monkeypatch.setenv("HOME", str(fake_home))
    # Re-import to refresh path resolution (skipping if Path expansion is cached)
    import importlib
    import mcp_server
    importlib.reload(mcp_server)
    with pytest.raises(PermissionError):
        mcp_server._api_key_for("admin")


def test_writer_role_uses_agent_json(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    (fake_home / ".claude/agentssot/local").mkdir(parents=True)
    (fake_home / ".claude/agentssot/local/agent.json").write_text(
        json.dumps({"api_key": "writer-key"})
    )
    monkeypatch.setenv("HOME", str(fake_home))
    import importlib, mcp_server
    importlib.reload(mcp_server)
    assert mcp_server._api_key_for("writer") == "writer-key"
