"""Server-side secret scanning for ingest content.

Rejects knowledge items, events, and requirements containing likely secrets
before they are persisted to the database. This is the last line of defense
after client-side filtering in extract_and_ingest.py and redaction.py.

Gated behind INGEST_SECRET_SCANNING (default: true).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("agentssot.secret_scanner")


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SecretPattern:
    """A named regex pattern that matches a class of secrets."""
    name: str
    pattern: re.Pattern[str]
    description: str


# All patterns are compiled once at module load.
SECRET_PATTERNS: list[SecretPattern] = [
    # ── API key prefixes ──────────────────────────────────────────
    SecretPattern(
        name="openai_api_key",
        pattern=re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
        description="OpenAI API key (sk-...)",
    ),
    SecretPattern(
        name="aws_access_key",
        pattern=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        description="AWS access key ID (AKIA...)",
    ),
    SecretPattern(
        name="aws_secret_key",
        pattern=re.compile(r"(?i)(?:aws_secret_access_key|aws_secret)\s*[:=]\s*[A-Za-z0-9/+=]{30,}"),
        description="AWS secret access key assignment",
    ),
    SecretPattern(
        name="github_pat",
        pattern=re.compile(r"\bghp_[A-Za-z0-9]{36,}\b"),
        description="GitHub personal access token (ghp_...)",
    ),
    SecretPattern(
        name="github_oauth",
        pattern=re.compile(r"\bgho_[A-Za-z0-9]{36,}\b"),
        description="GitHub OAuth token (gho_...)",
    ),
    SecretPattern(
        name="github_app",
        pattern=re.compile(r"\bghs_[A-Za-z0-9]{36,}\b"),
        description="GitHub app token (ghs_...)",
    ),
    SecretPattern(
        name="github_refresh",
        pattern=re.compile(r"\bghr_[A-Za-z0-9]{36,}\b"),
        description="GitHub refresh token (ghr_...)",
    ),
    SecretPattern(
        name="gitlab_pat",
        pattern=re.compile(r"\bglpat-[A-Za-z0-9\-]{20,}\b"),
        description="GitLab personal access token",
    ),
    SecretPattern(
        name="slack_token",
        pattern=re.compile(r"\bxox[bporas]-[A-Za-z0-9\-]{10,}"),
        description="Slack token (xoxb-, xoxp-, etc.)",
    ),
    SecretPattern(
        name="stripe_key",
        pattern=re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{20,}"),
        description="Stripe API key",
    ),
    SecretPattern(
        name="anthropic_key",
        pattern=re.compile(r"\bsk-ant-[A-Za-z0-9\-]{20,}"),
        description="Anthropic API key (sk-ant-...)",
    ),
    SecretPattern(
        name="google_api_key",
        pattern=re.compile(r"\bAIza[A-Za-z0-9_\-]{35}\b"),
        description="Google API key",
    ),
    SecretPattern(
        name="heroku_api_key",
        pattern=re.compile(r"(?i)heroku\s*(?:api[_ ]?key|token)\s*[:=]\s*[A-Za-z0-9\-]{30,}"),
        description="Heroku API key",
    ),
    SecretPattern(
        name="sendgrid_key",
        pattern=re.compile(r"\bSG\.[A-Za-z0-9_\-]{22,}\.[A-Za-z0-9_\-]{22,}"),
        description="SendGrid API key",
    ),
    SecretPattern(
        name="twilio_key",
        pattern=re.compile(r"\bSK[a-f0-9]{32}\b"),
        description="Twilio API key",
    ),
    SecretPattern(
        name="npm_token",
        pattern=re.compile(r"\bnpm_[A-Za-z0-9]{36,}\b"),
        description="npm access token",
    ),
    SecretPattern(
        name="pypi_token",
        pattern=re.compile(r"\bpypi-[A-Za-z0-9\-]{50,}\b"),
        description="PyPI API token",
    ),

    # ── Generic secret assignments ────────────────────────────────
    SecretPattern(
        name="generic_password_assignment",
        pattern=re.compile(
            r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{8,}['\"]?"
        ),
        description="Password assignment (password=...)",
    ),
    SecretPattern(
        name="generic_token_assignment",
        pattern=re.compile(
            r"(?i)(?:api[_ ]?key|api[_ ]?secret|access[_ ]?token|auth[_ ]?token|bearer[_ ]?token|secret[_ ]?key)"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9_\-/.+=]{16,}['\"]?"
        ),
        description="Generic token/key assignment",
    ),
    SecretPattern(
        name="authorization_bearer",
        pattern=re.compile(
            r"(?i)(?:authorization|bearer)\s*[:=]\s*(?:bearer\s+)?[A-Za-z0-9_\-/.+=]{20,}"
        ),
        description="Authorization/Bearer token",
    ),

    # ── Private keys ──────────────────────────────────────────────
    SecretPattern(
        name="private_key_pem",
        pattern=re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
        description="PEM-encoded private key",
    ),
    SecretPattern(
        name="private_key_hex",
        pattern=re.compile(r"\b0x[a-fA-F0-9]{64,}\b"),
        description="Hex-encoded private key (64+ hex chars)",
    ),

    # ── Connection strings ────────────────────────────────────────
    SecretPattern(
        name="database_url",
        pattern=re.compile(
            r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|mssql)"
            r"://[^\s'\"]{10,}"
        ),
        description="Database/service connection string with credentials",
    ),

    # ── .env file values ──────────────────────────────────────────
    SecretPattern(
        name="env_secret_value",
        pattern=re.compile(
            r"(?i)^(?:export\s+)?(?:DATABASE_URL|DB_PASSWORD|SECRET_KEY|JWT_SECRET|ENCRYPTION_KEY"
            r"|PRIVATE_KEY|API_KEY|API_SECRET|AUTH_SECRET|SESSION_SECRET"
            r"|STRIPE_SECRET|OPENAI_API_KEY|ANTHROPIC_API_KEY|AWS_SECRET)"
            r"\s*=\s*['\"]?[^\s'\"]{8,}['\"]?",
            re.MULTILINE,
        ),
        description="Environment variable secret assignment",
    ),

    # ── Project-specific token patterns (from extract_and_ingest.py) ──
    SecretPattern(
        name="project_tokens",
        pattern=re.compile(r"\b(?:cw|moltbook_sk|ssot)_[A-Za-z0-9_\-]{10,}\b"),
        description="Project-specific tokens (cw_, moltbook_sk_, ssot_)",
    ),

    # ── Solana / crypto wallet keys ───────────────────────────────
    SecretPattern(
        name="solana_private_key",
        pattern=re.compile(r"(?i)SOLANA_PRIVATE_KEY\b"),
        description="Solana private key reference",
    ),
    SecretPattern(
        name="base58_private_key",
        pattern=re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{87,88}\b"),
        description="Base58-encoded private key (Solana/ed25519 length)",
    ),
]


# ---------------------------------------------------------------------------
# Scanning API
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """Result of scanning a single text for secrets."""
    has_secrets: bool
    matched_patterns: list[str]  # pattern names, not the actual secret values

    @property
    def reason(self) -> str:
        if not self.matched_patterns:
            return ""
        return f"Content matches secret pattern(s): {', '.join(self.matched_patterns)}"


def scan_text(text: str) -> ScanResult:
    """Scan text for secret patterns.

    Returns a ScanResult indicating whether secrets were found and which
    pattern categories matched. Never includes the actual secret value
    in the result to prevent leaking via error messages.
    """
    if not text:
        return ScanResult(has_secrets=False, matched_patterns=[])

    matched: list[str] = []
    for sp in SECRET_PATTERNS:
        if sp.pattern.search(text):
            matched.append(sp.name)

    return ScanResult(has_secrets=bool(matched), matched_patterns=matched)


def scan_batch(texts: list[str]) -> dict[int, ScanResult]:
    """Scan a list of texts, returning a dict of index→ScanResult for items with secrets.

    Only items that contain secrets are included in the returned dict.
    """
    flagged: dict[int, ScanResult] = {}
    for idx, text in enumerate(texts):
        result = scan_text(text)
        if result.has_secrets:
            flagged[idx] = result
    return flagged


def scan_ingest_payload(payload) -> list[str]:
    """Pre-scan an IngestRequest-like payload for secrets.

    Works with any object that has .knowledge_items, .events, and .requirements
    attributes (duck-typed to avoid importing models/schemas here).

    Returns a list of rejection messages. Empty list means no secrets found.
    """
    rejections: list[str] = []

    for idx, item in enumerate(getattr(payload, "knowledge_items", [])):
        result = scan_text(getattr(item, "content", ""))
        if result.has_secrets:
            rejections.append(
                f"knowledge_items[{idx}]: rejected — {result.reason}"
            )

    for idx, item in enumerate(getattr(payload, "events", [])):
        for field_name in ("title", "body"):
            text = getattr(item, field_name, None) or ""
            if not text:
                continue
            result = scan_text(text)
            if result.has_secrets:
                rejections.append(
                    f"events[{idx}].{field_name}: rejected — {result.reason}"
                )

    for idx, item in enumerate(getattr(payload, "requirements", [])):
        for field_name in ("title", "body", "context_snippet"):
            text = getattr(item, field_name, None) or ""
            if not text:
                continue
            result = scan_text(text)
            if result.has_secrets:
                rejections.append(
                    f"requirements[{idx}].{field_name}: rejected — {result.reason}"
                )

    return rejections
