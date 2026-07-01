"""Secret redaction — strip secrets BEFORE classification, embedding, or storage.

Conservative: favours redacting over leaking. Patterns cover common fake-secret
shapes used in tests and real credentials.
"""

from __future__ import annotations

import re

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"
            r"[\s\S]*?-----END (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"
        ),
    ),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("github_pat", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
    ("bearer", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b", re.IGNORECASE)),
    (
        "password_kv",
        re.compile(r"\b(password|passwd|secret|api[_-]?key|token)\b\s*[:=]\s*\S+", re.IGNORECASE),
    ),
]


def redact_secrets(content: str) -> tuple[str, int, list[str]]:
    """Return (redacted_content, redaction_count, kinds_found)."""
    out = content
    total = 0
    kinds: list[str] = []
    for name, pattern in _SECRET_PATTERNS:
        def _sub(_match: re.Match[str], _name: str = name) -> str:
            nonlocal total
            total += 1
            if _name not in kinds:
                kinds.append(_name)
            return f"[REDACTED:{_name}]"

        out = pattern.sub(_sub, out)
    return out, total, kinds
