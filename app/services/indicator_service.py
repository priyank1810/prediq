import numpy as np
import pandas as pd
import ta


class IndicatorService:
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

        # ── 4. VWAP ──
        typical_price = (high + low + close) / 3
        cumvol = volume.cumsum()
        vwap_series = (typical_price * volume).cumsum() / cumvol.replace(0, np.nan)
        current_vwap = float(vwap_series.iloc[-1]) if not pd.isna(vwap_series.iloc[-1]) else current_price

        # VWAP score: distance from VWAP as % of price
        vwap_pct = ((current_price - current_vwap) / current_price * 100) if current_price > 0 else 0
        # Above VWAP = bullish, below = bearish. Scale: ±0.5% = strong signal
        vwap_score = max(-100, min(100, vwap_pct * 200))

        # ── 5. MA Crossover (5 vs 9) ──
        ma5 = ta.trend.SMAIndicator(close, window=5).sma_indicator()
        ma9 = ta.trend.SMAIndicator(close, window=9).sma_indicator()
        ma5_now = float(ma5.iloc[-1]) if not pd.isna(ma5.iloc[-1]) else current_price
        ma9_now = float(ma9.iloc[-1]) if not pd.isna(ma9.iloc[-1]) else current_price
        ma5_prev = float(ma5.iloc[-2]) if len(ma5) >= 2 and not pd.isna(ma5.iloc[-2]) else ma5_now
        ma9_prev = float(ma9.iloc[-2]) if len(ma9) >= 2 and not pd.isna(ma9.iloc[-2]) else ma9_now

        # Detect crossover: 5 crosses above 9 = bullish, below = bearish
        cross_bullish = ma5_prev <= ma9_prev and ma5_now > ma9_now
        cross_bearish = ma5_prev >= ma9_prev and ma5_now < ma9_now

        # Score: fresh cross = strong signal; ongoing separation = moderate
        ma_spread_pct = ((ma5_now - ma9_now) / current_price * 100) if current_price > 0 else 0
        if cross_bullish:
            ma_score = 80  # fresh bullish cross
        elif cross_bearish:
            ma_score = -80  # fresh bearish cross
        else:
            # Ongoing spread: 5 above 9 = bullish, scaled by distance
            ma_score = max(-100, min(100, ma_spread_pct * 300))

        # ── Weighted composite ──
        technical_score = (
            0.25 * rsi_score +
            0.20 * volume_score +
            0.20 * bb_score +
            0.20 * vwap_score +
            0.15 * ma_score
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
            "ma5": round(ma5_now, 2),
            "ma9": round(ma9_now, 2),
            "ma_cross": "bullish" if cross_bullish else ("bearish" if cross_bearish else "none"),
            "ma_score": round(ma_score, 2),
            "current_price": round(current_price, 2),
        }

        datetimes = df["datetime_str"].tolist() if "datetime_str" in df.columns else list(range(len(df)))
        raw = {
            "datetimes": datetimes,
            "rsi": [None if pd.isna(v) else round(v, 2) for v in rsi_series],
        }

        return {"score": technical_score, "details": details, "raw": raw}


indicator_service = IndicatorService()
