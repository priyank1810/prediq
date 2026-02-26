from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from fastapi import Request, HTTPException, Depends


class RateLimiter:
    def __init__(self):
        self.requests: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str, max_calls: int, window: int):
        now = datetime.utcnow().timestamp()
        # Prune expired entries
        self.requests[key] = [t for t in self.requests[key] if now - t < window]
        if len(self.requests[key]) >= max_calls:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {max_calls} requests per {window}s.",
            )
        self.requests[key].append(now)


_limiter = RateLimiter()


def rate_limit(max_calls: int = 5, window_seconds: int = 60):
    async def dependency(request: Request):
        # Use authenticated user email if available, otherwise IP
        user = getattr(request.state, "user", None)
        key = (getattr(user, "email", None) if user else None) or request.client.host
        _limiter.check(f"{key}:{request.url.path}", max_calls, window_seconds)

    return Depends(dependency)
