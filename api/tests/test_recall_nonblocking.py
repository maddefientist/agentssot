import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "app" / "routers" / "knowledge.py"


def test_embed_text_calls_are_offloaded_to_thread():
    """Every embed_text call in async handlers must be wrapped in
    asyncio.to_thread so it cannot block the event loop."""
    text = SRC.read_text()
    offenders = []
    for i, line in enumerate(text.splitlines(), 1):
        if "embed_text(" in line and "to_thread" not in line and not line.lstrip().startswith("#"):
            offenders.append((i, line.strip()))
    assert not offenders, f"Un-offloaded embed_text calls: {offenders}"