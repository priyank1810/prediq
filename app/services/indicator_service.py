import numpy as np
import pandas as pd
import ta


class IndicatorService:
    # ── Candlestick Pattern Detection ──
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> dict:
        """Detect 9 candlestick patterns from last 3 bars using raw OHLC math."""
        patterns = []
        if df is None or len(df) < 3:
            return {"patterns": patterns, "score": 0.0}

        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values

        # Helper: body and shadow sizes
        def body(i):
            return abs(c[i] - o[i])

        def upper_shadow(i):
            return h[i] - max(o[i], c[i])

        def lower_shadow(i):
            return min(o[i], c[i]) - l[i]

        def is_bullish(i):
            return c[i] > o[i]

        def candle_range(i):
            return h[i] - l[i] if h[i] != l[i] else 0.001

        i = len(df) - 1  # current bar
        p = i - 1         # previous bar
        pp = i - 2        # two bars ago

        # --- Single-bar patterns ---
        # Doji: body < 10% of range
        if body(i) < 0.10 * candle_range(i):
            patterns.append({"name": "Doji", "type": "neutral", "score": 0})

        # Hammer: small body at top, long lower shadow >= 2x body, small upper shadow
        if (lower_shadow(i) >= 2 * body(i) and
                upper_shadow(i) <= body(i) * 0.5 and
                body(i) > 0.05 * candle_range(i)):
            patterns.append({"name": "Hammer", "type": "bullish", "score": 60})

        # Shooting Star: small body at bottom, long upper shadow >= 2x body
        if (upper_shadow(i) >= 2 * body(i) and
                lower_shadow(i) <= body(i) * 0.5 and
                body(i) > 0.05 * candle_range(i)):
            patterns.append({"name": "Shooting Star", "type": "bearish", "score": -60})

        # --- Two-bar patterns ---
        # Bullish Engulfing: prev bearish, current bullish, current body engulfs prev body
        if (not is_bullish(p) and is_bullish(i) and
                o[i] <= c[p] and c[i] >= o[p] and body(i) > body(p)):
            patterns.append({"name": "Bullish Engulfing", "type": "bullish", "score": 70})

        # Bearish Engulfing: prev bullish, current bearish, current body engulfs prev body
        if (is_bullish(p) and not is_bullish(i) and
                o[i] >= c[p] and c[i] <= o[p] and body(i) > body(p)):
            patterns.append({"name": "Bearish Engulfing", "type": "bearish", "score": -70})

        # Bullish Harami: prev bearish with big body, current bullish inside prev body
        if (not is_bullish(p) and is_bullish(i) and
                body(p) > body(i) and
                o[i] >= c[p] and c[i] <= o[p]):
            patterns.append({"name": "Bullish Harami", "type": "bullish", "score": 40})

        # Bearish Harami: prev bullish with big body, current bearish inside prev body
        if (is_bullish(p) and not is_bullish(i) and
                body(p) > body(i) and
                o[i] <= c[p] and c[i] >= o[p]):
            patterns.append({"name": "Bearish Harami", "type": "bearish", "score": -40})

        # --- Three-bar patterns ---
        # Morning Star: bar[pp] bearish, bar[p] small body (star), bar[i] bullish closes above mid of pp
        mid_pp = (o[pp] + c[pp]) / 2
        if (not is_bullish(pp) and body(pp) > 0.3 * candle_range(pp) and
                body(p) < 0.3 * candle_range(p) and
                is_bullish(i) and c[i] > mid_pp):
            patterns.append({"name": "Morning Star", "type": "bullish", "score": 80})

        # Evening Star: bar[pp] bullish, bar[p] small body (star), bar[i] bearish closes below mid of pp
        if (is_bullish(pp) and body(pp) > 0.3 * candle_range(pp) and
                body(p) < 0.3 * candle_range(p) and
                not is_bullish(i) and c[i] < mid_pp):
            patterns.append({"name": "Evening Star", "type": "bearish", "score": -80})

        # Aggregate score: average of detected pattern scores (or 0 if none)
        if patterns:
            avg_score = sum(p["score"] for p in patterns) / len(patterns)
        else:
            avg_score = 0.0

        return {"patterns": patterns, "score": max(-100, min(100, round(avg_score, 2)))}

    # ── Support/Resistance Levels ──
    def compute_support_resistance(self, df: pd.DataFrame, current_price: float = None) -> dict:
        """Compute pivot points, previous day levels, and Fibonacci retracement."""
        if df is None or len(df) < 20:
            return {"levels": {}, "proximity_signal": 0}

        if current_price is None:
            current_price = float(df["close"].iloc[-1])

        # Determine previous day H/L/C from intraday data
        # Group by date and take the last complete day
        if "datetime_str" in df.columns:
            df_copy = df.copy()
            df_copy["_date"] = df_copy["datetime_str"].str[:10]
            dates = df_copy["_date"].unique()
            if len(dates) >= 2:
                prev_day = df_copy[df_copy["_date"] == dates[-2]]
                prev_high = float(prev_day["high"].max())
                prev_low = float(prev_day["low"].min())
                prev_close = float(prev_day["close"].iloc[-1])
            else:
                prev_high = float(df["high"].iloc[:-1].max()) if len(df) > 1 else float(df["high"].iloc[-1])
                prev_low = float(df["low"].iloc[:-1].min()) if len(df) > 1 else float(df["low"].iloc[-1])
                prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else float(df["close"].iloc[-1])
        else:
            prev_high = float(df["high"].iloc[-2]) if len(df) > 1 else float(df["high"].iloc[-1])
            prev_low = float(df["low"].iloc[-2]) if len(df) > 1 else float(df["low"].iloc[-1])
            prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else float(df["close"].iloc[-1])

        # Classic Pivot Points
        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pivot - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        s1 = 2 * pivot - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)

        # Fibonacci Retracement from 20-bar swing
        swing_high = float(df["high"].tail(20).max())
        swing_low = float(df["low"].tail(20).min())
        fib_range = swing_high - swing_low
        fib_levels = {
            "fib_236": round(swing_high - 0.236 * fib_range, 2),
            "fib_382": round(swing_high - 0.382 * fib_range, 2),
            "fib_500": round(swing_high - 0.500 * fib_range, 2),
            "fib_618": round(swing_high - 0.618 * fib_range, 2),
        }

        levels = {
            "pivot": round(pivot, 2),
            "r1": round(r1, 2), "r2": round(r2, 2), "r3": round(r3, 2),
            "s1": round(s1, 2), "s2": round(s2, 2), "s3": round(s3, 2),
            "prev_high": round(prev_high, 2),
            "prev_low": round(prev_low, 2),
            "prev_close": round(prev_close, 2),
            **fib_levels,
        }

        # Proximity signal: check if price is within 0.5% of any level
        threshold_pct = 0.005
        support_levels = [s1, s2, s3, swing_low, fib_levels["fib_618"], fib_levels["fib_500"]]
        resistance_levels = [r1, r2, r3, swing_high, fib_levels["fib_236"], fib_levels["fib_382"]]

        # Determine trend from last 5 bars
        if len(df) >= 5:
            trend = 1 if float(df["close"].iloc[-1]) > float(df["close"].iloc[-5]) else -1
        else:
            trend = 0

        proximity_signal = 0
        for lvl in support_levels:
            if lvl > 0 and abs(current_price - lvl) / current_price <= threshold_pct:
                proximity_signal += 30 if trend >= 0 else -20
                break
        for lvl in resistance_levels:
            if lvl > 0 and abs(current_price - lvl) / current_price <= threshold_pct:
                proximity_signal += 30 if trend < 0 else -20
                break

        proximity_signal = max(-100, min(100, proximity_signal))

        return {"levels": levels, "proximity_signal": proximity_signal, "trend": "up" if trend >= 0 else "down"}
    def compute_all(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        dates = df["date"].tolist()

        result = {}

        # RSI
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        result["rsi"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in rsi]}

        # MACD
        macd = ta.trend.MACD(close)
        result["macd_line"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in macd.macd()]}
        result["macd_signal"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in macd.macd_signal()]}
        result["macd_histogram"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in macd.macd_diff()]}

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        result["bollinger_upper"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in bb.bollinger_hband()]}
        result["bollinger_middle"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in bb.bollinger_mavg()]}
        result["bollinger_lower"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in bb.bollinger_lband()]}

        # SMA
        sma20 = ta.trend.SMAIndicator(close, window=20).sma_indicator()
        sma50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
        result["sma_20"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in sma20]}
        result["sma_50"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in sma50]}

        # EMA
        ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        result["ema_20"] = {"dates": dates, "values": [None if pd.isna(v) else round(v, 2) for v in ema20]}

        return result

    def compute_intraday_indicators(self, df: pd.DataFrame) -> dict:
        """15-min intraday signal scoring.

        Five indicators only: RSI, Volume Action, Bollinger Band, VWAP, 5/9 MA Crossover.
        """
        if df is None or df.empty or len(df) < 20:
            return {"score": 0, "details": {}, "raw": {}}

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        current_price = float(close.iloc[-1])

        # ── 1. RSI (14) ──
        rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi()
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

        # RSI scoring: oversold (<30) = strong bullish, overbought (>70) = strong bearish
        # Mid-zone (40-60) is neutral; 30-40 and 60-70 provide moderate signal
        if rsi <= 20:
            rsi_score = 100
        elif rsi <= 30:
            rsi_score = 60 + (30 - rsi) * 4  # 60 to 100
        elif rsi <= 40:
            rsi_score = (40 - rsi) * 6  # 0 to 60
        elif rsi <= 60:
            rsi_score = 0  # neutral zone
        elif rsi <= 70:
            rsi_score = -(rsi - 60) * 6  # 0 to -60
        elif rsi <= 80:
            rsi_score = -60 - (rsi - 70) * 4  # -60 to -100
        else:
            rsi_score = -100
        rsi_score = max(-100, min(100, rsi_score))

        # ── 2. Volume Action ──
        # Compare current bar volume to 20-bar average volume
        # A volume surge with price up = bullish confirmation; surge with price down = bearish
        vol_sma20 = float(volume.rolling(window=20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean())
        current_vol = float(volume.iloc[-1])
        vol_ratio = current_vol / vol_sma20 if vol_sma20 > 0 else 1.0

        # Price direction over the last bar
        price_change = float(close.iloc[-1] - close.iloc[-2]) if len(close) >= 2 else 0
        price_dir = 1 if price_change > 0 else (-1 if price_change < 0 else 0)

        # Volume score: high volume amplifies the price direction signal
        # vol_ratio > 1.5 = notable surge, > 2.0 = strong surge
        if vol_ratio > 2.0:
            vol_strength = 100
        elif vol_ratio > 1.5:
            vol_strength = 50 + (vol_ratio - 1.5) * 100  # 50 to 100
        elif vol_ratio > 1.0:
            vol_strength = (vol_ratio - 1.0) * 100  # 0 to 50
        elif vol_ratio > 0.5:
            vol_strength = -(1.0 - vol_ratio) * 40  # 0 to -20 (drying volume)
        else:
            vol_strength = -30  # very low volume — no conviction

        volume_score = max(-100, min(100, vol_strength * price_dir))

        # ── 3. Bollinger Bands (20, 2) ──
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper = float(bb.bollinger_hband().iloc[-1]) if not pd.isna(bb.bollinger_hband().iloc[-1]) else current_price * 1.02
        bb_lower = float(bb.bollinger_lband().iloc[-1]) if not pd.isna(bb.bollinger_lband().iloc[-1]) else current_price * 0.98
        bb_mid = float(bb.bollinger_mavg().iloc[-1]) if not pd.isna(bb.bollinger_mavg().iloc[-1]) else current_price

        bb_range = bb_upper - bb_lower if bb_upper != bb_lower else 1
        bb_pos = (current_price - bb_lower) / bb_range  # 0 = at lower, 1 = at upper

        # Price at/below lower band = bullish (mean reversion); at/above upper = bearish
        # Near middle band = neutral
        if bb_pos <= 0:
            bb_score = 100  # below lower band — strong bullish
        elif bb_pos <= 0.2:
            bb_score = 60 + (0.2 - bb_pos) * 200  # 60 to 100
        elif bb_pos <= 0.4:
            bb_score = (0.4 - bb_pos) * 300  # 0 to 60
        elif bb_pos <= 0.6:
            bb_score = 0  # middle zone — neutral
        elif bb_pos <= 0.8:
            bb_score = -(bb_pos - 0.6) * 300  # 0 to -60
        elif bb_pos < 1.0:
            bb_score = -60 - (bb_pos - 0.8) * 200  # -60 to -100
        else:
            bb_score = -100  # above upper band — strong bearish
        bb_score = max(-100, min(100, bb_score))

        # ── 4. VWAP + Bands (±1σ, ±2σ) ──
        typical_price = (high + low + close) / 3
        cumvol = volume.cumsum()
        vwap_series = (typical_price * volume).cumsum() / cumvol.replace(0, np.nan)
        current_vwap = float(vwap_series.iloc[-1]) if not pd.isna(vwap_series.iloc[-1]) else current_price

        # VWAP standard deviation bands
        vwap_var = ((typical_price - vwap_series) ** 2 * volume).cumsum() / cumvol.replace(0, np.nan)
        vwap_std = np.sqrt(vwap_var)
        vwap_upper_1 = vwap_series + vwap_std
        vwap_lower_1 = vwap_series - vwap_std
        vwap_upper_2 = vwap_series + 2 * vwap_std
        vwap_lower_2 = vwap_series - 2 * vwap_std

        current_std = float(vwap_std.iloc[-1]) if not pd.isna(vwap_std.iloc[-1]) else 0
        vwap_z = ((current_price - current_vwap) / current_std) if current_std > 0 else 0

        # VWAP score: enhanced with band-aware scoring
        vwap_pct = ((current_price - current_vwap) / current_price * 100) if current_price > 0 else 0
        if abs(vwap_z) >= 2:
            # Beyond 2σ — strong mean-reversion signal
            vwap_score = max(-100, min(100, -vwap_z * 50))
        else:
            # Normal: above VWAP = bullish, below = bearish
            vwap_score = max(-100, min(100, vwap_pct * 200))

        # ── 5. EMA Crossover (9 vs 21) ──
        ema9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
        ema9_now = float(ema9.iloc[-1]) if not pd.isna(ema9.iloc[-1]) else current_price
        ema21_now = float(ema21.iloc[-1]) if not pd.isna(ema21.iloc[-1]) else current_price
        ema9_prev = float(ema9.iloc[-2]) if len(ema9) >= 2 and not pd.isna(ema9.iloc[-2]) else ema9_now
        ema21_prev = float(ema21.iloc[-2]) if len(ema21) >= 2 and not pd.isna(ema21.iloc[-2]) else ema21_now

        # Detect crossover: 9 crosses above 21 = bullish, below = bearish
        cross_bullish = ema9_prev <= ema21_prev and ema9_now > ema21_now
        cross_bearish = ema9_prev >= ema21_prev and ema9_now < ema21_now

        # Score: fresh cross = strong signal; ongoing separation = moderate
        ma_spread_pct = ((ema9_now - ema21_now) / current_price * 100) if current_price > 0 else 0
        if cross_bullish:
            ma_score = 80  # fresh bullish cross
        elif cross_bearish:
            ma_score = -80  # fresh bearish cross
        else:
            # Ongoing spread: 9 above 21 = bullish, scaled by distance
            ma_score = max(-100, min(100, ma_spread_pct * 300))

        # ── 6. Candlestick Patterns ──
        candle_result = self._detect_candlestick_patterns(df)
        candle_score = candle_result["score"]

        # ── 7. MACD ──
        macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd_line = float(macd_obj.macd().iloc[-1]) if not pd.isna(macd_obj.macd().iloc[-1]) else 0
        macd_signal_line = float(macd_obj.macd_signal().iloc[-1]) if not pd.isna(macd_obj.macd_signal().iloc[-1]) else 0
        macd_hist = float(macd_obj.macd_diff().iloc[-1]) if not pd.isna(macd_obj.macd_diff().iloc[-1]) else 0

        # MACD scoring: histogram direction + crossover
        macd_prev = float(macd_obj.macd().iloc[-2]) if len(close) >= 2 and not pd.isna(macd_obj.macd().iloc[-2]) else macd_line
        macd_sig_prev = float(macd_obj.macd_signal().iloc[-2]) if len(close) >= 2 and not pd.isna(macd_obj.macd_signal().iloc[-2]) else macd_signal_line
        macd_cross_bull = macd_prev <= macd_sig_prev and macd_line > macd_signal_line
        macd_cross_bear = macd_prev >= macd_sig_prev and macd_line < macd_signal_line

        if macd_cross_bull:
            macd_score = 80
        elif macd_cross_bear:
            macd_score = -80
        elif macd_hist > 0:
            macd_score = min(100, macd_hist / (abs(macd_line) + 0.01) * 200)
        else:
            macd_score = max(-100, macd_hist / (abs(macd_line) + 0.01) * 200)
        macd_score = max(-100, min(100, macd_score))

        # ── 8. ADX ──
        adx_obj = ta.trend.ADXIndicator(high, low, close, window=14)
        adx_val = float(adx_obj.adx().iloc[-1]) if not pd.isna(adx_obj.adx().iloc[-1]) else 0
        plus_di = float(adx_obj.adx_pos().iloc[-1]) if not pd.isna(adx_obj.adx_pos().iloc[-1]) else 0
        minus_di = float(adx_obj.adx_neg().iloc[-1]) if not pd.isna(adx_obj.adx_neg().iloc[-1]) else 0

        # ADX scoring: ADX > 25 = trending, direction from +DI vs -DI
        if adx_val < 20:
            adx_score = 0  # Weak/no trend — neutral
        else:
            trend_strength = min(100, (adx_val - 20) * 2.5)  # 20→0, 60→100
            if plus_di > minus_di:
                adx_score = trend_strength
            else:
                adx_score = -trend_strength
        adx_score = max(-100, min(100, adx_score))

        # ── Weighted composite (8 indicators) ──
        technical_score = (
            0.18 * rsi_score +
            0.14 * volume_score +
            0.14 * bb_score +
            0.13 * vwap_score +
            0.10 * ma_score +
            0.10 * candle_score +
            0.12 * macd_score +
            0.09 * adx_score
        )
        technical_score = max(-100, min(100, round(technical_score, 2)))

        details = {
            "rsi": round(rsi, 2),
            "rsi_score": round(rsi_score, 2),
            "volume_ratio": round(vol_ratio, 2),
            "volume_score": round(volume_score, 2),
            "bb_position": round(bb_pos, 3),
            "bb_score": round(bb_score, 2),
            "vwap": round(current_vwap, 2),
            "vwap_pct": round(vwap_pct, 4),
            "vwap_score": round(vwap_score, 2),
            "vwap_upper_1": round(float(vwap_upper_1.iloc[-1]), 2) if not pd.isna(vwap_upper_1.iloc[-1]) else None,
            "vwap_lower_1": round(float(vwap_lower_1.iloc[-1]), 2) if not pd.isna(vwap_lower_1.iloc[-1]) else None,
            "vwap_upper_2": round(float(vwap_upper_2.iloc[-1]), 2) if not pd.isna(vwap_upper_2.iloc[-1]) else None,
            "vwap_lower_2": round(float(vwap_lower_2.iloc[-1]), 2) if not pd.isna(vwap_lower_2.iloc[-1]) else None,
            "ema9": round(ema9_now, 2),
            "ema21": round(ema21_now, 2),
            "ma_cross": "bullish" if cross_bullish else ("bearish" if cross_bearish else "none"),
            "ma_score": round(ma_score, 2),
            "macd_line": round(macd_line, 4),
            "macd_signal_line": round(macd_signal_line, 4),
            "macd_histogram": round(macd_hist, 4),
            "macd_score": round(macd_score, 2),
            "adx": round(adx_val, 2),
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "adx_score": round(adx_score, 2),
            "current_price": round(current_price, 2),
            "candlestick_patterns": candle_result["patterns"],
            "candlestick_score": candle_result["score"],
        }

        datetimes = df["datetime_str"].tolist() if "datetime_str" in df.columns else list(range(len(df)))
        raw = {
            "datetimes": datetimes,
            "rsi": [None if pd.isna(v) else round(v, 2) for v in rsi_series],
            "vwap": [None if pd.isna(v) else round(v, 2) for v in vwap_series],
            "vwap_upper_1": [None if pd.isna(v) else round(v, 2) for v in vwap_upper_1],
            "vwap_lower_1": [None if pd.isna(v) else round(v, 2) for v in vwap_lower_1],
            "vwap_upper_2": [None if pd.isna(v) else round(v, 2) for v in vwap_upper_2],
            "vwap_lower_2": [None if pd.isna(v) else round(v, 2) for v in vwap_lower_2],
        }

        return {"score": technical_score, "details": details, "raw": raw}


indicator_service = IndicatorService()
