from datetime import datetime


class RetryTaskAt(Exception):
    """Raise from a task handler to schedule a retry at a specific time."""

    def __init__(self, retry_at: datetime, message: str = ""):
        self.retry_at = retry_at
        super().__init__(message)
