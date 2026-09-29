"""
Worker package for distributed task processing.

This package provides:
- WorkerPoller: Polls PostgreSQL for pending tasks and executes them
- main: CLI entry point for hindsight-worker
"""

from .poller import WorkerPoller

__all__ = ["WorkerPoller"]
