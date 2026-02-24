"""Tests for the enrollment endpoints: POST /enroll/auto, GET /enroll/portal, GET /enroll/bootstrap.sh"""

import uuid

import httpx
import pytest

from .conftest import BASE_URL


def _unique_name(prefix: str = "pytest") -> str:
    """Generate a unique device name to avoid collisions between test runs."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# -- POST /enroll/auto --


class TestEnrollAuto:
    def test_success_returns_key_and_namespaces(self):
        name = _unique_name()
        resp = httpx.post(f"{BASE_URL}/enroll/auto", json={"name": name, "passphrase": ""})
        assert resp.status_code == 200

        data = resp.json()
        assert data["role"] == "writer"
        assert data["api_key"].startswith("ssot_")
        assert "claude-shared" in data["namespaces"]
        assert f"device-{name}-private" in data["namespaces"]

    def test_agent_config_present(self):
        name = _unique_name()
        resp = httpx.post(f"{BASE_URL}/enroll/auto", json={"name": name, "passphrase": ""})
        assert resp.status_code == 200

        config = resp.json()["agent_config"]
        assert config["device_name"] == name
        assert config["base_url"].startswith("http://")
        assert config["api_key"].startswith("ssot_")
        assert config["default_namespace"] == "claude-shared"
        assert config["default_scope"] == "knowledge"
        assert "claude-shared" in config["namespaces"]
        assert f"device-{name}-private" in config["namespaces"]

    def test_empty_name_returns_422(self):
        resp = httpx.post(f"{BASE_URL}/enroll/auto", json={"name": "", "passphrase": ""})
        assert resp.status_code == 422

    def test_missing_name_returns_422(self):
        resp = httpx.post(f"{BASE_URL}/enroll/auto", json={"passphrase": ""})
        assert resp.status_code == 422

    def test_created_key_works_against_health(self):
        name = _unique_name("verify")
        resp = httpx.post(f"{BASE_URL}/enroll/auto", json={"name": name, "passphrase": ""})
        assert resp.status_code == 200

        api_key = resp.json()["api_key"]
        health = httpx.get(f"{BASE_URL}/health", headers={"X-API-Key": api_key})
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

    def test_created_key_can_query(self):
        """Verify the enrolled key can call an authenticated endpoint."""
        name = _unique_name("query")
        resp = httpx.post(f"{BASE_URL}/enroll/auto", json={"name": name, "passphrase": ""})
        assert resp.status_code == 200

        api_key = resp.json()["api_key"]
        qr = httpx.get(
            f"{BASE_URL}/query",
            params={"namespace": "claude-shared", "q": "test"},
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        assert qr.status_code == 200

    def test_name_is_lowercased_and_sanitized(self):
        name_raw = _unique_name("Upper")
        resp = httpx.post(f"{BASE_URL}/enroll/auto", json={"name": name_raw, "passphrase": ""})
        assert resp.status_code == 200

        data = resp.json()
        expected_lower = name_raw.lower()
        assert data["agent_config"]["device_name"] == expected_lower
        assert f"device-{expected_lower}-private" in data["namespaces"]

    def test_passphrase_ignored_when_not_set(self):
        """With no ENROLLMENT_PASSPHRASE set on the server, any client passphrase is ignored."""
        name = _unique_name("pass")
        resp = httpx.post(f"{BASE_URL}/enroll/auto", json={"name": name, "passphrase": "anything"})
        assert resp.status_code == 200


# -- GET /enroll/portal --


class TestEnrollPortal:
    def test_returns_html(self):
        resp = httpx.get(f"{BASE_URL}/enroll/portal")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_html_has_content(self):
        resp = httpx.get(f"{BASE_URL}/enroll/portal")
        assert len(resp.text) > 100


# -- GET /enroll/bootstrap.sh --


class TestEnrollBootstrap:
    def test_returns_shell_script(self):
        resp = httpx.get(f"{BASE_URL}/enroll/bootstrap.sh")
        assert resp.status_code == 200
        assert resp.text.startswith("#!/usr/bin/env bash")

    def test_content_type_is_plain_text(self):
        resp = httpx.get(f"{BASE_URL}/enroll/bootstrap.sh")
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_script_references_enroll_auto(self):
        resp = httpx.get(f"{BASE_URL}/enroll/bootstrap.sh")
        assert "enroll/auto" in resp.text

    def test_script_references_base_url(self):
        resp = httpx.get(f"{BASE_URL}/enroll/bootstrap.sh")
        assert "BASE_URL=" in resp.text
