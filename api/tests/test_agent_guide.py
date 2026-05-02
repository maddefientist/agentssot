"""GET /agent-guide returns text/plain markdown tailored to the caller key."""
import json
import os
from pathlib import Path

import pytest
import httpx


@pytest.mark.integration
def test_agent_guide_writer_renders():
    # Use a known writer key from the local issued-keys file.
    keys_data = json.loads(
        Path(os.path.expanduser("~/.claude/agentssot/local/issued-keys.json")).read_text()
    )
    writer = next(k for k in keys_data["keys"] if k["role"] == "writer")
    with httpx.Client(base_url="http://localhost:8088") as client:
        r = client.get("/agent-guide", headers={"X-Api-Key": writer["api_key"]})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    # Per-key tailoring markers
    assert writer["name"] in body
    assert "writer" in body
    # Must include a tier cheat sheet
    for tier in ("command", "rule", "skill", "entity"):
        assert tier in body
    # Troubleshooting block
    assert "401" in body and "403" in body
