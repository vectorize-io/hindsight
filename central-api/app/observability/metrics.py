"""Observability metrics and dashboards."""

from datetime import datetime
from typing import Any


class MetricsCollector:
    """Collect runtime metrics."""

    def __init__(self):
        self.metrics: dict[str, Any] = {
            "requests_total": 0,
            "requests_errors": 0,
            "ledger_entries": 0,
            "quarantine_items": 0,
            "approvals_pending": 0,
            "api_latency_ms": 0,
            "db_connections": 0,
        }
        self.started_at = datetime.utcnow()

    def record_request(self, status_code: int, latency_ms: float):
        """Record API request."""
        self.metrics["requests_total"] += 1
        self.metrics["api_latency_ms"] = latency_ms
        if status_code >= 400:
            self.metrics["requests_errors"] += 1

    def record_ledger_entry(self):
        """Record execution ledger entry."""
        self.metrics["ledger_entries"] += 1

    def record_quarantine_item(self):
        """Record quarantine item."""
        self.metrics["quarantine_items"] += 1

    def record_approval_pending(self, count: int):
        """Record pending approvals."""
        self.metrics["approvals_pending"] = count

    def get_dashboard(self) -> dict[str, Any]:
        """Get dashboard snapshot."""
        uptime_seconds = (datetime.utcnow() - self.started_at).total_seconds()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": uptime_seconds,
            "metrics": self.metrics,
            "status": "healthy" if self.metrics["requests_errors"] < self.metrics["requests_total"] * 0.05 else "degraded",
        }


# Global singleton
_collector = MetricsCollector()


def get_collector() -> MetricsCollector:
    """Get global metrics collector."""
    return _collector
