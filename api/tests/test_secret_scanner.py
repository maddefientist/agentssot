"""Tests for Milestone 8: Secret Scanner + Ingest Rejection.

Covers:
1. Pattern detection — each secret category matched correctly
2. Clean content passes through
3. Ingest rejection — items with secrets rejected, clean items pass
4. Feature flag — scanning can be disabled
5. Error messages — clear without echoing the secret

NOTE: Test secrets are assembled at runtime to avoid tripping GitHub push
protection. Every "fake" token is built via string concatenation so the
literal pattern never appears in the source file.
"""

import os
import sys

import pytest

# Ensure app modules importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set dummy DATABASE_URL so Settings can be imported without a real DB
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"

from app.settings import get_settings
get_settings.cache_clear()

from app.secret_scanner import ScanResult, scan_batch, scan_text


# ── Helper: build fake secrets at runtime so they never appear literally ──
def _fake_openai_key():
    return "sk-" + "abc123def456ghi789jkl012mno345pqr678stu901"

def _fake_anthropic_key():
    return "sk-ant-" + "api03-abcdefghijklmnopqrstuvwxyz"

def _fake_aws_access():
    return "AKIA" + "IOSFODNN7EXAMPLE"

def _fake_aws_secret():
    return "wJalrXUtnFEMI/K7MDENG/" + "bPxRfiCYEXAMPLEKEY"

def _fake_github_pat():
    return "ghp_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"

def _fake_github_oauth():
    return "gho_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"

def _fake_github_app():
    return "ghs_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"

def _fake_github_refresh():
    return "ghr_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"

def _fake_gitlab_pat():
    return "glpat-" + "xxxxxxxxxxxxxxxxxxxx"

def _fake_slack_token():
    return "xoxb-" + "123456789012-1234567890123-abcdefghijklmnop"

def _fake_stripe_live():
    return "sk_live_" + "1234567890abcdefghij"

def _fake_stripe_test():
    return "sk_test_" + "1234567890abcdefghij"

def _fake_google_key():
    return "AIzaSy" + "A1234567890abcdefghijklmnopqrstuv"

def _fake_sendgrid_key():
    return "SG." + "abcdefghijklmnopqrstuv.wxyz1234567890abcdefghijkl"

def _fake_npm_token():
    return "npm_" + "aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789aa"

def _fake_pypi_token():
    return "pypi-" + "a" * 60

def _fake_bearer():
    return "Bearer " + "eyJhbGciOiJIUzI1NiIsInR5cCI6"

def _fake_db_url():
    return "postgresql://user:" + "password@host:5432/dbname"


# ═══════════════════════════════════════════════════════════════════
# PATTERN COVERAGE TESTS
# ═══════════════════════════════════════════════════════════════════


class TestAPIKeyPatterns:
    """Test detection of various API key formats."""

    def test_openai_key(self):
        result = scan_text(f"My key is {_fake_openai_key()}")
        assert result.has_secrets
        assert "openai_api_key" in result.matched_patterns

    def test_anthropic_key(self):
        result = scan_text(f"Using {_fake_anthropic_key()} for auth")
        assert result.has_secrets
        assert "anthropic_key" in result.matched_patterns

    def test_aws_access_key(self):
        result = scan_text(f"AWS key: {_fake_aws_access()}")
        assert result.has_secrets
        assert "aws_access_key" in result.matched_patterns

    def test_aws_secret_key(self):
        result = scan_text(f"aws_secret_access_key = {_fake_aws_secret()}")
        assert result.has_secrets
        assert "aws_secret_key" in result.matched_patterns

    def test_github_pat(self):
        result = scan_text(f"Token: {_fake_github_pat()}")
        assert result.has_secrets
        assert "github_pat" in result.matched_patterns

    def test_github_oauth(self):
        result = scan_text(_fake_github_oauth())
        assert result.has_secrets
        assert "github_oauth" in result.matched_patterns

    def test_github_app(self):
        result = scan_text(_fake_github_app())
        assert result.has_secrets
        assert "github_app" in result.matched_patterns

    def test_github_refresh(self):
        result = scan_text(_fake_github_refresh())
        assert result.has_secrets
        assert "github_refresh" in result.matched_patterns

    def test_gitlab_pat(self):
        result = scan_text(f"Token is {_fake_gitlab_pat()}")
        assert result.has_secrets
        assert "gitlab_pat" in result.matched_patterns

    def test_slack_token(self):
        result = scan_text(f"SLACK_TOKEN={_fake_slack_token()}")
        assert result.has_secrets
        assert "slack_token" in result.matched_patterns

    def test_stripe_live_key(self):
        result = scan_text(f"Stripe key: {_fake_stripe_live()}")
        assert result.has_secrets
        assert "stripe_key" in result.matched_patterns

    def test_stripe_test_key(self):
        result = scan_text(_fake_stripe_test())
        assert result.has_secrets
        assert "stripe_key" in result.matched_patterns

    def test_google_api_key(self):
        result = scan_text(f"Google key {_fake_google_key()}")
        assert result.has_secrets
        assert "google_api_key" in result.matched_patterns

    def test_sendgrid_key(self):
        result = scan_text(_fake_sendgrid_key())
        assert result.has_secrets
        assert "sendgrid_key" in result.matched_patterns

    def test_npm_token(self):
        result = scan_text(_fake_npm_token())
        assert result.has_secrets
        assert "npm_token" in result.matched_patterns

    def test_pypi_token(self):
        result = scan_text(f"Published with {_fake_pypi_token()}")
        assert result.has_secrets
        assert "pypi_token" in result.matched_patterns


class TestPasswordPatterns:
    """Test detection of password assignments."""

    def test_password_equals(self):
        result = scan_text("password=SuperSecretP@ss123")
        assert result.has_secrets
        assert "generic_password_assignment" in result.matched_patterns

    def test_passwd_colon(self):
        result = scan_text("passwd: my_secret_password_123")
        assert result.has_secrets
        assert "generic_password_assignment" in result.matched_patterns

    def test_pwd_equals_quoted(self):
        result = scan_text("pwd='longpassword123'")
        assert result.has_secrets
        assert "generic_password_assignment" in result.matched_patterns


class TestTokenPatterns:
    """Test detection of generic token assignments."""

    def test_api_key_assignment(self):
        result = scan_text("api_key=abcdef1234567890abcdef")
        assert result.has_secrets
        assert "generic_token_assignment" in result.matched_patterns

    def test_access_token_assignment(self):
        result = scan_text("access_token: eyJhbGciOiJIUzI1NiIs")
        assert result.has_secrets
        assert "generic_token_assignment" in result.matched_patterns

    def test_auth_token_equals(self):
        result = scan_text("AUTH_TOKEN=abcdef1234567890ab")
        assert result.has_secrets
        assert "generic_token_assignment" in result.matched_patterns

    def test_bearer_token(self):
        result = scan_text(f"Authorization: {_fake_bearer()}")
        assert result.has_secrets
        assert "authorization_bearer" in result.matched_patterns


class TestPrivateKeyPatterns:
    """Test detection of private key material."""

    def test_rsa_private_key(self):
        result = scan_text("-----BEGIN RSA PRIVATE KEY-----")
        assert result.has_secrets
        assert "private_key_pem" in result.matched_patterns

    def test_ec_private_key(self):
        result = scan_text("-----BEGIN EC PRIVATE KEY-----")
        assert result.has_secrets
        assert "private_key_pem" in result.matched_patterns

    def test_openssh_private_key(self):
        result = scan_text("-----BEGIN OPENSSH PRIVATE KEY-----")
        assert result.has_secrets
        assert "private_key_pem" in result.matched_patterns

    def test_generic_private_key(self):
        result = scan_text("-----BEGIN PRIVATE KEY-----")
        assert result.has_secrets
        assert "private_key_pem" in result.matched_patterns

    def test_hex_private_key(self):
        hex_key = "0x" + "a1b2c3d4" * 9  # 72 hex chars
        result = scan_text(f"Key: {hex_key}")
        assert result.has_secrets
        assert "private_key_hex" in result.matched_patterns


class TestConnectionStringPatterns:
    """Test detection of database/service connection strings."""

    def test_postgres_url(self):
        result = scan_text(_fake_db_url())
        assert result.has_secrets
        assert "database_url" in result.matched_patterns

    def test_mysql_url(self):
        result = scan_text("mysql://root:" + "secret@localhost:3306/app")
        assert result.has_secrets
        assert "database_url" in result.matched_patterns

    def test_mongodb_url(self):
        result = scan_text("mongodb+srv://admin:" + "pass@cluster.example.com/db")
        assert result.has_secrets
        assert "database_url" in result.matched_patterns

    def test_redis_url(self):
        result = scan_text("redis://default:" + "password@redis-host:6379/0")
        assert result.has_secrets
        assert "database_url" in result.matched_patterns


class TestEnvSecretPatterns:
    """Test detection of .env-style secret assignments."""

    def test_database_url_env(self):
        result = scan_text("DATABASE_URL=" + "postgresql://user:pass@host/db")
        assert result.has_secrets

    def test_openai_api_key_env(self):
        result = scan_text("OPENAI_API_KEY=" + "sk-abcdef1234567890abcdef")
        assert result.has_secrets

    def test_export_secret_key(self):
        result = scan_text("export SECRET_KEY=my-super-secret-value-here")
        assert result.has_secrets

    def test_jwt_secret_env(self):
        result = scan_text("JWT_SECRET=some-long-jwt-secret-value")
        assert result.has_secrets


class TestProjectTokenPatterns:
    """Test detection of project-specific tokens."""

    def test_cw_token(self):
        result = scan_text("cw_abcdef1234567890")
        assert result.has_secrets
        assert "project_tokens" in result.matched_patterns

    def test_ssot_token(self):
        result = scan_text("ssot_abcdef1234567890")
        assert result.has_secrets
        assert "project_tokens" in result.matched_patterns

    def test_moltbook_sk_token(self):
        result = scan_text("moltbook_sk_abcdef1234567890")
        assert result.has_secrets
        assert "project_tokens" in result.matched_patterns


class TestSolanaPatterns:
    """Test detection of crypto wallet keys."""

    def test_solana_private_key_ref(self):
        result = scan_text("SOLANA_PRIVATE_KEY is set")
        assert result.has_secrets
        assert "solana_private_key" in result.matched_patterns

    def test_base58_key(self):
        # 88-char base58 string (Solana keypair)
        key = "5" * 88
        result = scan_text(f"Key: {key}")
        assert result.has_secrets
        assert "base58_private_key" in result.matched_patterns


# ═══════════════════════════════════════════════════════════════════
# CLEAN CONTENT TESTS — Must NOT trigger false positives
# ═══════════════════════════════════════════════════════════════════


class TestCleanContent:
    """Verify that normal content passes through without false positives."""

    def test_normal_text(self):
        result = scan_text("We decided to use PostgreSQL for the database layer")
        assert not result.has_secrets

    def test_code_snippet(self):
        result = scan_text("def hello(): return 'world'")
        assert not result.has_secrets

    def test_technical_discussion(self):
        result = scan_text("The API key rotation policy should be every 90 days")
        assert not result.has_secrets

    def test_mention_of_passwords_conceptually(self):
        result = scan_text("Users should change their passwords regularly")
        assert not result.has_secrets

    def test_short_sk_prefix(self):
        # "sk-" followed by <20 chars should NOT match as OpenAI key
        result = scan_text("sk-short")
        assert not result.has_secrets

    def test_empty_text(self):
        result = scan_text("")
        assert not result.has_secrets

    def test_none_equivalent(self):
        result = scan_text("")
        assert not result.has_secrets
        assert result.matched_patterns == []

    def test_short_password_value(self):
        # Password with <8 char value shouldn't trigger
        result = scan_text("password=short")
        assert not result.has_secrets

    def test_normal_github_mention(self):
        result = scan_text("Push to GitHub and create a PR")
        assert not result.has_secrets

    def test_database_concept(self):
        result = scan_text("PostgreSQL supports vector embeddings via pgvector")
        assert not result.has_secrets


# ═══════════════════════════════════════════════════════════════════
# BATCH SCANNING TESTS
# ═══════════════════════════════════════════════════════════════════


class TestBatchScanning:
    """Test batch scanning across multiple texts."""

    def test_mixed_batch(self):
        texts = [
            "Clean content about architecture",
            f"My key is {_fake_openai_key()}",
            "Another clean piece of knowledge",
            "password=SuperSecretP@ss123",
        ]
        flagged = scan_batch(texts)
        assert 0 not in flagged  # clean
        assert 1 in flagged      # has openai key
        assert 2 not in flagged  # clean
        assert 3 in flagged      # has password

    def test_all_clean_batch(self):
        texts = [
            "Normal knowledge item",
            "Architecture decision about caching",
            "User prefers dark mode",
        ]
        flagged = scan_batch(texts)
        assert len(flagged) == 0

    def test_empty_batch(self):
        flagged = scan_batch([])
        assert len(flagged) == 0


# ═══════════════════════════════════════════════════════════════════
# SCAN RESULT API TESTS
# ═══════════════════════════════════════════════════════════════════


class TestScanResult:
    """Test the ScanResult data structure."""

    def test_reason_empty_when_clean(self):
        result = scan_text("Just normal text")
        assert result.reason == ""

    def test_reason_lists_patterns(self):
        result = scan_text(_fake_openai_key())
        assert "openai_api_key" in result.reason
        assert "secret pattern" in result.reason.lower()

    def test_reason_does_not_contain_actual_secret(self):
        secret = _fake_openai_key()
        result = scan_text(f"Key: {secret}")
        assert secret not in result.reason

    def test_multiple_patterns_in_reason(self):
        # Text with both an API key and a password
        text = f"{_fake_openai_key()} password=MySecret123!"
        result = scan_text(text)
        assert result.has_secrets
        assert len(result.matched_patterns) >= 2


# ═══════════════════════════════════════════════════════════════════
# INGEST REJECTION BEHAVIOR TESTS (unit-level, no DB)
# ═══════════════════════════════════════════════════════════════════


class TestIngestRejection:
    """Test the scan_ingest_payload helper from secret_scanner.py."""

    def _make_payload(self, knowledge_items=None, events=None, requirements=None):
        """Create a mock IngestRequest-like object."""
        from app.schemas import IngestRequest
        return IngestRequest(
            namespace="test-ns",
            knowledge_items=knowledge_items or [],
            events=events or [],
            requirements=requirements or [],
        )

    def test_clean_payload_passes(self):
        from app.secret_scanner import scan_ingest_payload
        payload = self._make_payload(
            knowledge_items=[
                {"content": "Normal fact about databases", "tags": ["test"]},
                {"content": "Another clean knowledge item", "tags": ["test"]},
            ],
        )
        rejections = scan_ingest_payload(payload)
        assert rejections == []

    def test_secret_in_knowledge_rejected(self):
        from app.secret_scanner import scan_ingest_payload
        fake_key = _fake_openai_key()
        payload = self._make_payload(
            knowledge_items=[
                {"content": f"Use key {fake_key}", "tags": ["test"]},
            ],
        )
        rejections = scan_ingest_payload(payload)
        assert len(rejections) == 1
        assert "knowledge_items[0]" in rejections[0]
        assert "openai_api_key" in rejections[0]
        # Must NOT contain the actual secret
        assert fake_key not in rejections[0]

    def test_secret_in_event_body_rejected(self):
        from app.secret_scanner import scan_ingest_payload
        payload = self._make_payload(
            events=[
                {"title": "Setup", "body": "password=SuperSecretValue123", "type": "note"},
            ],
        )
        rejections = scan_ingest_payload(payload)
        assert len(rejections) == 1
        assert "events[0].body" in rejections[0]

    def test_secret_in_event_title_rejected(self):
        from app.secret_scanner import scan_ingest_payload
        payload = self._make_payload(
            events=[
                {"title": f"Key: {_fake_github_pat()}", "type": "note"},
            ],
        )
        rejections = scan_ingest_payload(payload)
        assert len(rejections) == 1
        assert "events[0].title" in rejections[0]

    def test_secret_in_requirement_rejected(self):
        from app.secret_scanner import scan_ingest_payload
        payload = self._make_payload(
            requirements=[
                {"title": "Setup DB", "body": "postgresql://user:" + "pass@host:5432/db"},
            ],
        )
        rejections = scan_ingest_payload(payload)
        assert len(rejections) == 1
        assert "requirements[0].body" in rejections[0]

    def test_mixed_clean_and_dirty_items(self):
        from app.secret_scanner import scan_ingest_payload
        payload = self._make_payload(
            knowledge_items=[
                {"content": "Clean item 1", "tags": []},
                {"content": f"Has secret: {_fake_aws_access()}", "tags": []},
                {"content": "Clean item 3", "tags": []},
            ],
        )
        rejections = scan_ingest_payload(payload)
        assert len(rejections) == 1
        assert "knowledge_items[1]" in rejections[0]


class TestSettingsSecretScanning:
    """Verify the ingest_secret_scanning feature flag."""

    def test_default_is_enabled(self):
        from app.settings import Settings
        s = Settings(DATABASE_URL="postgresql://test:test@localhost/test")
        assert s.ingest_secret_scanning is True

    def test_can_disable(self):
        from app.settings import Settings
        s = Settings(DATABASE_URL="postgresql://test:" + "test@localhost/test", INGEST_SECRET_SCANNING="false")
        assert s.ingest_secret_scanning is False


# ═══════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY TESTS
# ═══════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Ensure existing ingest workflows without secrets still work."""

    def test_typical_fact_ingest_passes(self):
        texts = [
            "We decided to use FastAPI for the backend",
            "PostgreSQL 16 with pgvector extension for embeddings",
            "Deployment target is Docker Compose on a home lab server",
            "User prefers dark mode in all terminal applications",
            "The architecture follows a monorepo pattern with shared packages",
        ]
        for text in texts:
            result = scan_text(text)
            assert not result.has_secrets, f"False positive on: {text}"

    def test_code_discussion_passes(self):
        texts = [
            "The function signature is def ingest_batch(session, payload, provider)",
            "Using SQLAlchemy ORM with mapped_column for type safety",
            "Embedding dimension is 768 for nomic-embed-text model",
            "Compaction threshold: 80 events or 24000 characters",
        ]
        for text in texts:
            result = scan_text(text)
            assert not result.has_secrets, f"False positive on: {text}"

    def test_config_discussion_without_values(self):
        texts = [
            "Set the EMBEDDING_PROVIDER env var to ollama",
            "The LLM_PROVIDER should be set to none for testing",
            "Enable COMPACTION_ENABLED=true in production",
            "The default TOP_K is 5",
        ]
        for text in texts:
            result = scan_text(text)
            assert not result.has_secrets, f"False positive on: {text}"
