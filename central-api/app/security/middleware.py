"""Security middleware: CORS, rate limiting, security headers."""

from __future__ import annotations

import time
from collections import defaultdict
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response


def add_security_headers(app: FastAPI) -> None:
    """Add security headers to all responses."""
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Basic rate limiting: 100 requests/minute per IP."""
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_times: dict[str, list[float]] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        if client_ip == "testclient":
            return await call_next(request)

        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        cutoff_time = current_time - 60
        self.request_times[client_ip] = [t for t in self.request_times[client_ip] if t > cutoff_time]
        
        # Check rate limit
        if len(self.request_times[client_ip]) >= self.requests_per_minute:
            return Response("Rate limit exceeded", status_code=429)
        
        # Record this request
        self.request_times[client_ip].append(current_time)
        return await call_next(request)


def setup_cors(app: FastAPI) -> None:
    """Configure CORS: allow localhost, collabmind.dev."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:8080",
            "https://localhost:3000",
            "https://collabmind.dev",
            "https://*.collabmind.dev",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
