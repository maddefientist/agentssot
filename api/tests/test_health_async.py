import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_health_endpoint_is_async_and_dbless():
    text = SRC.read_text()
    m = re.search(r"@app\.get\(\"/health\"\)\s*\n\s*(async def|def) health\(", text)
    assert m, "could not locate /health handler"
    assert m.group(1) == "async def", "/health must be async so it never needs a threadpool token"
    body = text.split('def health(', 1)[1].split('\ndef ', 1)[0].split('\n@app', 1)[0]
    assert "get_session" not in body and "SessionLocal" not in body, "/health must not touch the DB"