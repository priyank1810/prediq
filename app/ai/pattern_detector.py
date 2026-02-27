import logging

import numpy as np
import pandas as pd

try:
    from scipy.signal import argrelextrema

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_diff(a: float, b: float) -> float:
    """Return the absolute percentage difference between two values."""
    mid = (abs(a) + abs(b)) / 2.0
    if mid == 0:
        return 0.0
    return abs(a - b) / mid


def _slope(values: np.ndarray) -> float:
    """Ordinary-least-squares slope over an index range 0..n-1."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, values, 1)[0])


# ---------------------------------------------------------------------------
# PatternDetector
# ---------------------------------------------------------------------------

class PatternDetector:
    """Detect common chart patterns in OHLCV price data.

    Uses ``scipy.signal.argrelextrema`` to locate local maxima / minima and
    then applies geometric rules on the resulting extrema positions and price
    levels.
    """

    ORDER = 5          # bars on each side for argrelextrema
    LOOKBACK = 60      # analyse the most recent N bars
    TOLERANCE = 0.02   # 2 % tolerance for "similar level" comparisons

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, df: pd.DataFrame) -> list[dict]:
        """Scan *df* for known chart patterns.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with at least ``close``, ``high``, ``low`` and
            ``date`` columns.

        Returns
        -------
        list[dict]
            Each dict has keys ``type``, ``start_date``, ``end_date``,
            ``confidence`` and ``description``.
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy is not installed – pattern detection disabled")
            return []

        if df is None or len(df) < self.ORDER * 2 + 1:
            logger.debug("Not enough data for pattern detection")
            return []

        required = {"close", "high", "low", "date"}
        if not required.issubset(set(df.columns)):
            logger.warning(
                "DataFrame missing required columns for pattern detection: %s",
                required - set(df.columns),
            )
            return []

        # Work on the tail of the data only
        window = df.tail(self.LOOKBACK).reset_index(drop=True)
        close = window["close"].values.astype(float)
        highs = window["high"].values.astype(float)
        lows = window["low"].values.astype(float)
        dates = window["date"].astype(str).values

        # Locate local extrema
        max_idx = argrelextrema(close, np.greater_equal, order=self.ORDER)[0]
        min_idx = argrelextrema(close, np.less_equal, order=self.ORDER)[0]

        if len(max_idx) == 0 and len(min_idx) == 0:
            logger.debug("No extrema found in the lookback window")
            return []

        patterns: list[dict] = []

        patterns.extend(self._detect_double_top(close, dates, max_idx, min_idx))
        patterns.extend(self._detect_double_bottom(close, dates, min_idx, max_idx))
        patterns.extend(self._detect_head_and_shoulders(close, dates, max_idx, min_idx))
        patterns.extend(self._detect_rising_wedge(highs, lows, dates, max_idx, min_idx))
        patterns.extend(self._detect_falling_wedge(highs, lows, dates, max_idx, min_idx))

        logger.info("Detected %d chart pattern(s)", len(patterns))
        return patterns

    # ------------------------------------------------------------------
    # Double Top
    # ------------------------------------------------------------------

    def _detect_double_top(
        self,
        close: np.ndarray,
        dates: np.ndarray,
        max_idx: np.ndarray,
        min_idx: np.ndarray,
    ) -> list[dict]:
        patterns: list[dict] = []
        if len(max_idx) < 2 or len(min_idx) < 1:
            return patterns

        for i in range(len(max_idx) - 1):
            pk1 = max_idx[i]
            pk2 = max_idx[i + 1]
            price1 = close[pk1]
            price2 = close[pk2]

            # Peaks must be at similar levels (within tolerance)
            if _pct_diff(price1, price2) > self.TOLERANCE:
                continue

            # There must be at least one trough between the peaks
            troughs_between = min_idx[(min_idx > pk1) & (min_idx < pk2)]
            if len(troughs_between) == 0:
                continue

            trough_idx = troughs_between[np.argmin(close[troughs_between])]
            trough_price = close[trough_idx]

            # The trough should be meaningfully lower than the peaks
            drop_pct = (((price1 + price2) / 2.0) - trough_price) / ((price1 + price2) / 2.0)
            if drop_pct < 0.01:
                continue

            # Confidence: closer peak levels -> higher confidence
            diff_pct = _pct_diff(price1, price2)
            conf = max(0.5, min(0.95, 0.95 - diff_pct * 10))

            # Boost confidence when the trough is deeper
            if drop_pct > 0.05:
                conf = min(0.95, conf + 0.05)

            patterns.append({
                "type": "Double Top",
                "start_date": str(dates[pk1]),
                "end_date": str(dates[pk2]),
                "confidence": round(conf, 2),
                "description": (
                    f"Two peaks near {price1:.2f} and {price2:.2f} with a trough "
                    f"at {trough_price:.2f} between them. "
                    f"Potential bearish reversal signal."
                ),
            })
        return patterns

    # ------------------------------------------------------------------
    # Double Bottom
    # ------------------------------------------------------------------

    def _detect_double_bottom(
        self,
        close: np.ndarray,
        dates: np.ndarray,
        min_idx: np.ndarray,
        max_idx: np.ndarray,
    ) -> list[dict]:
        patterns: list[dict] = []
        if len(min_idx) < 2 or len(max_idx) < 1:
            return patterns

        for i in range(len(min_idx) - 1):
            tr1 = min_idx[i]
            tr2 = min_idx[i + 1]
            price1 = close[tr1]
            price2 = close[tr2]

            if _pct_diff(price1, price2) > self.TOLERANCE:
                continue

            peaks_between = max_idx[(max_idx > tr1) & (max_idx < tr2)]
            if len(peaks_between) == 0:
                continue

            peak_idx = peaks_between[np.argmax(close[peaks_between])]
            peak_price = close[peak_idx]

            rise_pct = (peak_price - (price1 + price2) / 2.0) / ((price1 + price2) / 2.0)
            if rise_pct < 0.01:
                continue

            diff_pct = _pct_diff(price1, price2)
            conf = max(0.5, min(0.95, 0.95 - diff_pct * 10))

            if rise_pct > 0.05:
                conf = min(0.95, conf + 0.05)

            patterns.append({
                "type": "Double Bottom",
                "start_date": str(dates[tr1]),
                "end_date": str(dates[tr2]),
                "confidence": round(conf, 2),
                "description": (
                    f"Two troughs near {price1:.2f} and {price2:.2f} with a peak "
                    f"at {peak_price:.2f} between them. "
                    f"Potential bullish reversal signal."
                ),
            })
        return patterns

    # ------------------------------------------------------------------
    # Head & Shoulders
    # ------------------------------------------------------------------

    def _detect_head_and_shoulders(
        self,
        close: np.ndarray,
        dates: np.ndarray,
        max_idx: np.ndarray,
        min_idx: np.ndarray,
    ) -> list[dict]:
        patterns: list[dict] = []
        if len(max_idx) < 3:
            return patterns

        for i in range(len(max_idx) - 2):
            left = max_idx[i]
            head = max_idx[i + 1]
            right = max_idx[i + 2]

            left_price = close[left]
            head_price = close[head]
            right_price = close[right]

            # Head must be the highest of the three
            if head_price <= left_price or head_price <= right_price:
                continue

            # Shoulders should be at similar levels
            if _pct_diff(left_price, right_price) > self.TOLERANCE:
                continue

            # Head should be meaningfully higher than the shoulders
            shoulder_avg = (left_price + right_price) / 2.0
            head_rise = (head_price - shoulder_avg) / shoulder_avg
            if head_rise < 0.01:
                continue

            # Check for troughs (neckline) between left-head and head-right
            trough_lh = min_idx[(min_idx > left) & (min_idx < head)]
            trough_hr = min_idx[(min_idx > head) & (min_idx < right)]
            if len(trough_lh) == 0 or len(trough_hr) == 0:
                continue

            neckline1 = close[trough_lh[np.argmin(close[trough_lh])]]
            neckline2 = close[trough_hr[np.argmin(close[trough_hr])]]

            # Confidence based on shoulder symmetry and head prominence
            shoulder_diff = _pct_diff(left_price, right_price)
            neckline_diff = _pct_diff(neckline1, neckline2)

            conf = 0.80
            # Penalise asymmetric shoulders
            conf -= shoulder_diff * 5
            # Penalise uneven neckline
            conf -= neckline_diff * 3
            # Reward prominent head
            if head_rise > 0.03:
                conf += 0.05
            conf = round(max(0.5, min(0.95, conf)), 2)

            neckline_avg = (neckline1 + neckline2) / 2.0
            patterns.append({
                "type": "Head & Shoulders",
                "start_date": str(dates[left]),
                "end_date": str(dates[right]),
                "confidence": conf,
                "description": (
                    f"Left shoulder at {left_price:.2f}, head at {head_price:.2f}, "
                    f"right shoulder at {right_price:.2f}. "
                    f"Neckline near {neckline_avg:.2f}. "
                    f"Potential bearish reversal signal."
                ),
            })
        return patterns

    # ------------------------------------------------------------------
    # Rising Wedge
    # ------------------------------------------------------------------

    def _detect_rising_wedge(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        dates: np.ndarray,
        max_idx: np.ndarray,
        min_idx: np.ndarray,
    ) -> list[dict]:
        patterns: list[dict] = []
        if len(max_idx) < 3 or len(min_idx) < 3:
            return patterns

        # Use the last few extrema to evaluate the wedge
        recent_max = max_idx[-3:]
        recent_min = min_idx[-3:]

        peak_prices = highs[recent_max]
        trough_prices = lows[recent_min]

        # Higher highs
        if not np.all(np.diff(peak_prices) > 0):
            return patterns

        # Higher lows
        if not np.all(np.diff(trough_prices) > 0):
            return patterns

        # Converging: slope of highs must be less than slope of lows
        # (both positive, but the upper boundary rises more slowly
        #  relative to the price range than the lower boundary)
        slope_highs = _slope(peak_prices)
        slope_lows = _slope(trough_prices)

        if slope_highs <= 0 or slope_lows <= 0:
            return patterns

        # The lines converge when the high-slope is smaller than the low-slope
        # *relative to the price range*, i.e. the spread narrows over time.
        spread_start = peak_prices[0] - trough_prices[0]
        spread_end = peak_prices[-1] - trough_prices[-1]
        if spread_start <= 0 or spread_end <= 0:
            return patterns
        if spread_end >= spread_start:
            return patterns

        convergence_ratio = spread_end / spread_start  # < 1 means converging
        conf = max(0.5, min(0.95, 0.90 - convergence_ratio * 0.3))
        conf = round(conf, 2)

        start = min(recent_max[0], recent_min[0])
        end = max(recent_max[-1], recent_min[-1])

        patterns.append({
            "type": "Rising Wedge",
            "start_date": str(dates[start]),
            "end_date": str(dates[end]),
            "confidence": conf,
            "description": (
                f"Higher highs ({peak_prices[0]:.2f} -> {peak_prices[-1]:.2f}) "
                f"and higher lows ({trough_prices[0]:.2f} -> {trough_prices[-1]:.2f}) "
                f"with converging trendlines. "
                f"Potential bearish reversal signal."
            ),
        })
        return patterns

    # ------------------------------------------------------------------
    # Falling Wedge
    # ------------------------------------------------------------------

    def _detect_falling_wedge(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        dates: np.ndarray,
        max_idx: np.ndarray,
        min_idx: np.ndarray,
    ) -> list[dict]:
        patterns: list[dict] = []
        if len(max_idx) < 3 or len(min_idx) < 3:
            return patterns

        recent_max = max_idx[-3:]
        recent_min = min_idx[-3:]

        peak_prices = highs[recent_max]
        trough_prices = lows[recent_min]

        # Lower highs
        if not np.all(np.diff(peak_prices) < 0):
            return patterns

        # Lower lows
        if not np.all(np.diff(trough_prices) < 0):
            return patterns

        slope_highs = _slope(peak_prices)
        slope_lows = _slope(trough_prices)

        if slope_highs >= 0 or slope_lows >= 0:
            return patterns

        # Converging: the spread between highs and lows narrows
        spread_start = peak_prices[0] - trough_prices[0]
        spread_end = peak_prices[-1] - trough_prices[-1]
        if spread_start <= 0 or spread_end <= 0:
            return patterns
        if spread_end >= spread_start:
            return patterns

        convergence_ratio = spread_end / spread_start
        conf = max(0.5, min(0.95, 0.90 - convergence_ratio * 0.3))
        conf = round(conf, 2)

        start = min(recent_max[0], recent_min[0])
        end = max(recent_max[-1], recent_min[-1])

        patterns.append({
            "type": "Falling Wedge",
            "start_date": str(dates[start]),
            "end_date": str(dates[end]),
            "confidence": conf,
            "description": (
                f"Lower highs ({peak_prices[0]:.2f} -> {peak_prices[-1]:.2f}) "
                f"and lower lows ({trough_prices[0]:.2f} -> {trough_prices[-1]:.2f}) "
                f"with converging trendlines. "
                f"Potential bullish reversal signal."
            ),
        })
        return patterns


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
pattern_detector = PatternDetector()
