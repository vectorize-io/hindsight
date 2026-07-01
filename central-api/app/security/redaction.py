import re
import logging
from typing import Any
from functools import wraps

SECRET_PATTERNS = [
    r'("password":|"api[_-]?key":|"secret":|"token":|"authorization":)\s*"[^"]*"',
    r'(_password=)([^\s&]*)',
    r'(api[_-]?key=)([^\s&]*)',
    r'(Bearer\s+)([^\s]*)',
    r'(mem11_sk_)([^\s"\']*)',
    r'(:mem11_sk_)([^\s"\']*)',
]

def redact_secrets(text: str) -> str:
    """Redact sensitive data from logs."""
    if not isinstance(text, str):
        return str(text)
    for pattern in SECRET_PATTERNS:
        text = re.sub(pattern, r'\1***', text, flags=re.IGNORECASE)
    return text


class SecretRedactionFilter(logging.Filter):
    """Logging filter to redact secrets from log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_secrets(str(record.msg))
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: redact_secrets(str(v)) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(redact_secrets(str(v)) for v in record.args)
        return True


def setup_redaction_filter(logger: logging.Logger) -> None:
    """Add secret redaction filter to logger."""
    redaction_filter = SecretRedactionFilter()
    for handler in logger.handlers:
        handler.addFilter(redaction_filter)


def log_with_redaction(func):
    """Decorator to redact secrets from function return values before logging."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, dict):
            return {k: redact_secrets(str(v)) for k, v in result.items()}
        return redact_secrets(str(result))
    return wrapper
