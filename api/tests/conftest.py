import os

import httpx
import pytest

BASE_URL = os.environ.get("SSOT_TEST_URL", "http://YOUR_HOST:8088")


@pytest.fixture
def base_url():
    return BASE_URL


@pytest.fixture
def client():
    """Provide a reusable httpx client pointed at the running API."""
    with httpx.Client(base_url=BASE_URL, timeout=10) as c:
        yield c
