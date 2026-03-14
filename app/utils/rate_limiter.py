import time
import threading


class RateLimiter:
    """In-memory sliding-window rate limiter keyed by IP address.

    Thread-safe; old entries are cleaned up automatically every
    ``cleanup_interval`` seconds to prevent memory leaks.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60,
                 cleanup_interval: int = 300):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.cleanup_interval = cleanup_interval

        self._timestamps: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def check_rate_limit(self, ip: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        now = time.monotonic()

        with self._lock:
            # Periodic cleanup of stale entries
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup(now)
                self._last_cleanup = now

            timestamps = self._timestamps.get(ip, [])
            # Prune timestamps outside the current window
            timestamps = [t for t in timestamps if now - t < self.window_seconds]

            if len(timestamps) >= self.max_requests:
                self._timestamps[ip] = timestamps
                return False

            timestamps.append(now)
            self._timestamps[ip] = timestamps
            return True

    def time_until_available(self, ip: str) -> int:
        """Return seconds until the next request would be allowed (for Retry-After)."""
        now = time.monotonic()
        with self._lock:
            timestamps = self._timestamps.get(ip, [])
            timestamps = [t for t in timestamps if now - t < self.window_seconds]
            if len(timestamps) < self.max_requests:
                return 0
            # Oldest timestamp in window determines when a slot opens
            oldest = min(timestamps)
            return max(1, int(self.window_seconds - (now - oldest)) + 1)

    def _cleanup(self, now: float) -> None:
        """Remove IPs whose timestamps are all outside the window (called under lock)."""
        stale_ips = []
        for ip, timestamps in self._timestamps.items():
            active = [t for t in timestamps if now - t < self.window_seconds]
            if not active:
                stale_ips.append(ip)
            else:
                self._timestamps[ip] = active
        for ip in stale_ips:
            del self._timestamps[ip]
