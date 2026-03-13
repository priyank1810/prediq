"""Simple circuit breaker for external API calls.

States: CLOSED (normal) -> OPEN (failing, reject calls) -> HALF_OPEN (test one call)

Usage:
    breaker = CircuitBreaker("yahoo", failure_threshold=5, recovery_timeout=60)

    if breaker.allow_request():
        try:
            result = call_external_api()
            breaker.record_success()
            return result
        except Exception:
            breaker.record_failure()
            raise
    else:
        raise ExternalAPIUnavailable("yahoo")
"""

import time
import threading
import logging

logger = logging.getLogger(__name__)


class CircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
            return self._state

    def allow_request(self) -> bool:
        s = self.state
        if s == self.CLOSED:
            return True
        if s == self.HALF_OPEN:
            return True  # Allow one test request
        return False  # OPEN — reject

    def record_success(self):
        with self._lock:
            self._failure_count = 0
            if self._state == self.HALF_OPEN:
                logger.info(f"Circuit breaker [{self.name}] recovered — closing")
            self._state = self.CLOSED

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                if self._state != self.OPEN:
                    logger.warning(
                        f"Circuit breaker [{self.name}] opened after {self._failure_count} failures. "
                        f"Blocking requests for {self.recovery_timeout}s."
                    )
                self._state = self.OPEN


# Shared breakers for external APIs
yahoo_breaker = CircuitBreaker("yahoo", failure_threshold=5, recovery_timeout=60)
angel_breaker = CircuitBreaker("angel_one", failure_threshold=3, recovery_timeout=90)
nse_breaker = CircuitBreaker("nse", failure_threshold=5, recovery_timeout=120)
